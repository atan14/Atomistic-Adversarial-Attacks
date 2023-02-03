import torch
import numpy as np
from torch import nn
from torch.nn import ModuleDict, Sequential, Linear
from torch.nn.functional import softmax

from .layers import MessagePassingModule, GaussianSmearing, Dense
from .activations import ShiftedSoftplus
from .construct import construct_module_dict, layer_types
from evi.utils import scatter_add, compute_grad, make_directed


EPSILON = 1e-15
DEFAULT_BONDPRIOR_PARAM = {"k": 20.0}


def get_offsets(batch, offset_key, nbr_key='nbr_list'):
    nxyz = batch['nxyz']
    zero = torch.Tensor([0]).to(nxyz.device)
    offsets = batch.get(offset_key, zero)
    if isinstance(offsets, torch.Tensor) and offsets.is_sparse:
        offsets = offsets.to_dense()
    return offsets


def get_rij(xyz,
            batch,
            nbrs,
            cutoff):

    offsets = get_offsets(batch, 'offsets')
    # + offsets not - offsets because it's r_j - r_i,
    # whereas for schnet we've coded it as r_i - r_j
    r_ij = xyz[nbrs[:, 1]] - xyz[nbrs[:, 0]] + offsets

    # originally, nbrs given is directed, so r_ij computation
    # is more expensive. Since r_ij for a two-way directed
    # nbr is the same, concatenating rij and -rij is the same
    # as supplying directed nbr list. This would save a bit of
    # calculation time.
    nbrs, directed = make_directed(nbrs)
    if not directed:
        r_ij = torch.cat([r_ij, -r_ij], dim=0)

    # remove nbr skin (extra distance added to cutoff
    # to catch atoms that become neighbors between nbr
    # list updates)
    dist = (r_ij.detach() ** 2).sum(-1) ** 0.5

    if type(cutoff) == torch.Tensor:
        dist = dist.to(cutoff.device)
    use_nbrs = (dist <= cutoff)

    r_ij = r_ij[use_nbrs]
    nbrs = nbrs[use_nbrs]

    return r_ij, nbrs


def add_stress(batch,
               all_results,
               nbrs,
               r_ij):
    """
    Add stress as output. Needs to be divided by lattice volume to get actual stress.
    For batching for loop seemed unavoidable. will change later.
    stress considers both for crystal and molecules.
    For crystals need to divide by lattice volume.
    r_ij considers offsets which is different for molecules and crystals.
    """
    Z = compute_grad(output=all_results['energy'],
                     inputs=r_ij)
    if batch['num_atoms'].shape[0] == 1:
        all_results['stress_volume'] = torch.matmul(Z.t(), r_ij)
    else:
        allstress = []
        for j in range(batch['nxyz'].shape[0]):
            allstress.append(
                torch.matmul(
                    Z[torch.where(nbrs[:, 0] == j)].t(),
                    r_ij[torch.where(nbrs[:, 0] == j)]
                )
            )
        allstress = torch.stack(allstress)
        N = batch["num_atoms"].detach().cpu().tolist()
        split_val = torch.split(allstress, N)
        all_results['stress_volume'] = torch.stack([i.sum(0)
                                                    for i in split_val])
    return all_results


class SchNetConv(MessagePassingModule):

    """The convolution layer with filter.

    Attributes:
        moduledict (TYPE): Description
    """

    def __init__(
        self,
        n_atom_basis,
        n_filters,
        n_gaussians,
        cutoff,
        trainable_gauss,
        dropout_rate,
    ):
        super(SchNetConv, self).__init__()
        self.moduledict = ModuleDict(
            {
                "message_edge_filter": Sequential(
                    GaussianSmearing(
                        start=0.0,
                        stop=cutoff,
                        n_gaussians=n_gaussians,
                        trainable=trainable_gauss,
                    ),
                    Dense(
                        in_features=n_gaussians,
                        out_features=n_gaussians,
                        dropout_rate=dropout_rate,
                    ),
                    ShiftedSoftplus(),
                    Dense(
                        in_features=n_gaussians,
                        out_features=n_filters,
                        dropout_rate=dropout_rate,
                    ),
                ),
                "message_node_filter": Dense(
                    in_features=n_atom_basis,
                    out_features=n_filters,
                    dropout_rate=dropout_rate,
                ),
                "update_function": Sequential(
                    Dense(
                        in_features=n_filters,
                        out_features=n_atom_basis,
                        dropout_rate=dropout_rate,
                    ),
                    ShiftedSoftplus(),
                    Dense(
                        in_features=n_atom_basis,
                        out_features=n_atom_basis,
                        dropout_rate=dropout_rate,
                    ),
                ),
            }
        )

    def message(self, r, e, a, aggr_wgt=None):
        """The message function for SchNet convoltuions
        Args:
            r (TYPE): node inputs
            e (TYPE): edge inputs
            a (TYPE): neighbor list
            aggr_wgt (None, optional): Description

        Returns:
            TYPE: message should a pair of message and
        """
        # update edge feature
        e = self.moduledict["message_edge_filter"](e)
        # convection: update
        r = self.moduledict["message_node_filter"](r)

        # soft aggr if aggr_wght is provided
        if aggr_wgt is not None:
            r = r * aggr_wgt

        # combine node and edge info
        message = r[a[:, 0]] * e, r[a[:, 1]] * e  # (ri [] eij) -> rj, []: *, +, (,)
        return message

    def update(self, r):
        return self.moduledict["update_function"](r)


class NodeMultiTaskReadOut(nn.Module):
    """Stack Multi Task outputs

        example multitaskdict:

        multitaskdict = {
            'myenergy_0': [
                {'name': 'linear', 'param' : { 'in_features': 5, 'out_features': 20}},
                {'name': 'linear', 'param' : { 'in_features': 20, 'out_features': 1}}
            ],
            'myenergy_1': [
                {'name': 'linear', 'param' : { 'in_features': 5, 'out_features': 20}},
                {'name': 'linear', 'param' : { 'in_features': 20, 'out_features': 1}}
            ],
            'muliken_charges': [
                {'name': 'linear', 'param' : { 'in_features': 5, 'out_features': 20}},
                {'name': 'linear', 'param' : { 'in_features': 20, 'out_features': 1}}
            ]
        }

        example post_readout:

        def post_readout(predict_dict, readoutdict):
            sorted_keys = sorted(list(readoutdict.keys()))
            sorted_ens = torch.sort(torch.stack([predict_dict[key] for key in sorted_keys]))[0]
            sorted_dic = {key: val for key, val in zip(sorted_keys, sorted_ens) }
            return sorted_dic
    """

    def __init__(self, multitaskdict, post_readout=None):
        """Summary

        Args:
            multitaskdict (dict): dictionary that contains model information
        """
        super(NodeMultiTaskReadOut, self).__init__()
        # construct moduledict
        self.readout = construct_module_dict(multitaskdict)
        self.post_readout = post_readout
        self.multitaskdict = multitaskdict

    def forward(self, r):
        predict_dict = dict()

        # for backwards compatability
        if not hasattr(self, "readout"):
            self.readout = construct_module_dict(self.multitaskdict["atom_readout"])
            self.readout.to(r.device)

        for key in self.readout:
            predict_dict[key] = self.readout[key](r)

        if getattr(self, "post_readout", None) is not None:
            predict_dict = self.post_readout(predict_dict)

        ###
        # predict_dict['energy_0'] = predict_dict['d0']
        # predict_dict['energy_1'] = predict_dict['d1']
        ###

        return predict_dict


def sum_and_grad(batch,
                 xyz,
                 r_ij,
                 nbrs,
                 atomwise_output,
                 grad_keys,
                 out_keys=None,
                 mean=False):

    N = batch["num_atoms"].detach().cpu().tolist()
    results = {}
    if out_keys is None:
        out_keys = list(atomwise_output.keys())

    for key, val in atomwise_output.items():
        if key not in out_keys:
            continue

        mol_idx = torch.arange(len(N)).repeat_interleave(
            torch.LongTensor(N)).to(val.device)
        dim_size = mol_idx.max() + 1

        if val.reshape(-1).shape[0] == mol_idx.shape[0]:
            use_val = val.reshape(-1)

        # summed atom features
        elif val.shape[0] == mol_idx.shape[0]:
            use_val = val.sum(-1)

        else:
            raise Exception(("Don't know how to handle val shape "
                             "{} for key {}" .format(val.shape, key)))

        pooled_result = scatter_add(use_val,
                                    mol_idx,
                                    dim_size=dim_size)
        if mean:
            pooled_result = pooled_result / torch.Tensor(N).to(val.device)

        results[key] = pooled_result

    # compute gradients
    for key in grad_keys:

        # pooling has already been done to add to total props for each system
        # but batch still contains multiple systems
        # so need to be careful to do things in batched fashion
        if key == 'stress':
            output = results['energy']
            grad_ = compute_grad(output=output,
                                 inputs=r_ij)
            allstress = []
            for i in range(batch['nxyz'].shape[0]):
                allstress.append(
                    torch.matmul(
                        grad_[torch.where(nbrs[:, 0] == i)].t(),
                        r_ij[torch.where(nbrs[:, 0] == i)]
                    )
                )
            allstress = torch.stack(allstress)
            split_val = torch.split(allstress, N)
            grad_ = torch.stack([i.sum(0) for i in split_val])
            if 'cell' in batch.keys():
                cell = torch.stack(torch.split(batch['cell'], 3, dim=0))
            elif 'lattice' in batch.keys():
                cell = torch.stack(torch.split(batch['lattice'], 3, dim=0))
            volume = (torch.Tensor(np.abs(np.linalg.det(cell.cpu().numpy())))
                                                        .to(grad_.get_device()))
            grad = grad_*(1/volume[:, None, None])
            grad = torch.flatten(grad, start_dim=0, end_dim=1)

        else:
            output = results[key.replace("_grad", "")]
            grad = compute_grad(output=output,
                                inputs=xyz)

        results[key] = grad

    return results


class SumPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                batch,
                xyz,
                r_ij,
                nbrs,
                atomwise_output,
                grad_keys,
                out_keys=None):
        results = sum_and_grad(batch=batch,
                               xyz=xyz,
                               r_ij=r_ij,
                               nbrs=nbrs,
                               atomwise_output=atomwise_output,
                               grad_keys=grad_keys,
                               out_keys=out_keys)
        return results


class MeanPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                batch,
                xyz,
                atomwise_output,
                grad_keys,
                out_keys=None):
        results = sum_and_grad(batch=batch,
                               xyz=xyz,
                               atomwise_output=atomwise_output,
                               grad_keys=grad_keys,
                               out_keys=out_keys,
                               mean=True)
        return results


def att_readout_probs(name):
    if name.lower() == "softmax":
        def func(output):
            weights = softmax(output, dim=0)
            return weights

    elif name.lower() == "square":
        def func(output):
            weights = ((output ** 2 / (output ** 2).sum()))
            return weights
    else:
        raise NotImplementedError

    return func


class AttentionPool(nn.Module):
    """
    Compute output quantities using attention, rather than a sum over
    atomic quantities. There are two methods to do this:
    (1): "atomwise": Learn the attention weights from atomic fingerprints,
    get atomwise quantities from a network applied to the fingeprints,
    and sum them with attention weights.
    (2) "mol_fp": Learn the attention weights from atomic fingerprints,
    multiply the fingerprints by these weights, add the fingerprints
    together to get a molecular fingerprint, and put the molecular
    fingerprint through a network that predicts the output.

    This one uses `mol_fp`, since it seems more expressive (?)
    """

    def __init__(self,
                 prob_func,
                 feat_dim,
                 att_act,
                 mol_fp_act,
                 num_out_layers,
                 out_dim,
                 **kwargs):
        """

        """
        super().__init__()

        self.w_mat = Linear(in_features=feat_dim,
                               out_features=feat_dim,
                               bias=False)

        self.att_weight = torch.nn.Parameter(torch.rand(1, feat_dim))
        nn.init.xavier_uniform_(self.att_weight, gain=1.414)
        self.prob_func = att_readout_probs(prob_func)
        self.att_act = layer_types[att_act]()

        # reduce the number of features by the same factor in each layer
        feat_num = [int(feat_dim / num_out_layers ** m)
                    for m in range(num_out_layers)]

        # make layers followed by an activation for all but the last
        # layer
        mol_fp_layers = [Dense(in_features=feat_num[i],
                               out_features=feat_num[i+1],
                               activation=layer_types[mol_fp_act]())
                         for i in range(num_out_layers - 1)]

        # use no activation for the last layer
        mol_fp_layers.append(Dense(in_features=feat_num[-1],
                                   out_features=out_dim,
                                   activation=None))

        # put together in readout network
        self.mol_fp_nn = Sequential(*mol_fp_layers)

    def forward(self,
                batch,
                xyz,
                atomwise_output,
                grad_keys,
                out_keys):
        """
        Args:
            feats (torch.Tensor): n_atom x feat_dim atomic features,
                after convolutions are finished.
        """

        N = batch["num_atoms"].detach().cpu().tolist()
        results = {}

        for key in out_keys:

            batched_feats = atomwise_output['features']

            # split the outputs into those of each molecule
            split_feats = torch.split(batched_feats, N)
            # sum the results for each molecule

            all_outputs = []
            learned_feats = []

            for feats in split_feats:
                weights = self.prob_func(
                    self.att_act(
                        (self.att_weight * self.w_mat(feats)).sum(-1)
                    )
                )

                mol_fp = (weights.reshape(-1, 1) * self.w_mat(feats)).sum(0)

                output = self.mol_fp_nn(mol_fp)
                all_outputs.append(output)
                learned_feats.append(mol_fp)

            results[key] = torch.stack(all_outputs).reshape(-1)
            results[f"{key}_features"] = torch.stack(learned_feats)

        for key in grad_keys:
            output = results[key.replace("_grad", "")]
            grad = compute_grad(output=output,
                                inputs=xyz)
            results[key] = grad

        return results


class MolFpPool(nn.Module):
    def __init__(self,
                 feat_dim,
                 mol_fp_act,
                 num_out_layers,
                 out_dim,
                 **kwargs):

        super().__init__()

        # reduce the number of features by the same factor in each layer
        feat_num = [int(feat_dim / num_out_layers ** m)
                    for m in range(num_out_layers)]

        # make layers followed by an activation for all but the last
        # layer
        mol_fp_layers = [Dense(in_features=feat_num[i],
                               out_features=feat_num[i+1],
                               activation=layer_types[mol_fp_act]())
                         for i in range(num_out_layers - 1)]

        # use no activation for the last layer
        mol_fp_layers.append(Dense(in_features=feat_num[-1],
                                   out_features=out_dim,
                                   activation=None))

        # put together in readout network
        self.mol_fp_nn = Sequential(*mol_fp_layers)

    def forward(self,
                batch,
                xyz,
                atomwise_output,
                grad_keys,
                out_keys):
        """
        Args:
            feats (torch.Tensor): n_atom x feat_dim atomic features,
                after convolutions are finished.
        """

        N = batch["num_atoms"].detach().cpu().tolist()
        results = {}

        for key in out_keys:

            batched_feats = atomwise_output['features']

            # split the outputs into those of each molecule
            split_feats = torch.split(batched_feats, N)
            # sum the results for each molecule

            all_outputs = []
            learned_feats = []

            for feats in split_feats:
                mol_fp = feats.sum(0)
                output = self.mol_fp_nn(mol_fp)
                all_outputs.append(output)
                learned_feats.append(mol_fp)

            results[key] = torch.stack(all_outputs).reshape(-1)
            results[f"{key}_features"] = torch.stack(learned_feats)

        for key in grad_keys:
            output = results[key.replace("_grad", "")]
            grad = compute_grad(output=output,
                                inputs=xyz)
            results[key] = grad

        return results
