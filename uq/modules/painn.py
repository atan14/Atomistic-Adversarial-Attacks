import torch
from torch import nn
import numpy as np

from .layers import Dense, ScaleShift
from .construct import layer_types
from evi.utils.graphop import scatter_add


def norm(vec, eps=1e-15):
    result = ((vec ** 2 + eps).sum(-1)) ** 0.5
    return result


def preprocess_r(r_ij):
    """
    r_ij (n_nbrs x 3): tensor of interatomic vectors (r_j - r_i)
    """

    dist = norm(r_ij)
    unit = r_ij / dist.reshape(-1, 1)

    return dist, unit


def to_module(activation):
    return layer_types[activation]()


class CosineEnvelope(nn.Module):
    # Behler, J. Chem. Phys. 134, 074106 (2011)
    def __init__(self, cutoff):
        super().__init__()

        self.cutoff = cutoff

    def forward(self, d):

        output = 0.5 * (torch.cos((np.pi * d / self.cutoff)) + 1)
        exclude = d >= self.cutoff
        output[exclude] = 0

        return output


class PainnRadialBasis(nn.Module):
    def __init__(self,
                 n_gaussians,
                 cutoff,
                 trainable_gauss):
        super().__init__()

        self.n = torch.arange(1, n_gaussians + 1).float()
        if trainable_gauss:
            self.n = nn.Parameter(self.n)

        self.cutoff = cutoff

    def forward(self, dist):
        """
        Args:
            d (torch.Tensor): tensor of distances
        """

        shape_d = dist.unsqueeze(-1)
        n = self.n.to(dist.device)
        coef = n * np.pi / self.cutoff
        device = shape_d.device

        # replace divide by 0 with limit of sinc function
        denom = torch.where(shape_d == 0,
                            torch.tensor(1.0, device=device, dtype=shape_d.dtype),
                            shape_d)
        num = torch.where(shape_d == 0,
                          coef.to(shape_d.dtype),
                          torch.sin(coef * shape_d))

        output = torch.where(shape_d >= self.cutoff,
                             torch.tensor(0.0, device=device, dtype=shape_d.dtype),
                             num / denom)

        return output


class InvariantDense(nn.Module):
    def __init__(self,
                 dim,
                 dropout,
                 activation='swish'):
        super().__init__()
        self.layers = nn.Sequential(Dense(in_features=dim,
                                          out_features=dim,
                                          bias=True,
                                          dropout_rate=dropout,
                                          activation=to_module(activation)),
                                    Dense(in_features=dim,
                                          out_features=3 * dim,
                                          bias=True,
                                          dropout_rate=dropout))

    def forward(self, s_j):
        output = self.layers(s_j)
        return output


class DistanceEmbed(nn.Module):
    def __init__(self,
                 n_gaussians,
                 cutoff,
                 n_atom_basis,
                 trainable_gauss,
                 dropout):

        super().__init__()
        rbf = PainnRadialBasis(n_gaussians=n_gaussians,
                               cutoff=cutoff,
                               trainable_gauss=trainable_gauss)

        dense = Dense(in_features=n_gaussians,
                      out_features=3 * n_atom_basis,
                      bias=True,
                      dropout_rate=dropout)
        self.block = nn.Sequential(rbf, dense)
        self.f_cut = CosineEnvelope(cutoff=cutoff)

    def forward(self, dist):
        rbf_feats = self.block(dist)
        envelope = self.f_cut(dist).reshape(-1, 1)
        output = rbf_feats * envelope

        return output


class InvariantMessage(nn.Module):
    def __init__(self,
                 n_atom_basis,
                 activation,
                 n_gaussians,
                 cutoff,
                 trainable_gauss,
                 dropout):
        super().__init__()

        self.inv_dense = InvariantDense(dim=n_atom_basis,
                                        activation=activation,
                                        dropout=dropout)
        self.dist_embed = DistanceEmbed(n_gaussians=n_gaussians,
                                        cutoff=cutoff,
                                        n_atom_basis=n_atom_basis,
                                        trainable_gauss=trainable_gauss,
                                        dropout=dropout)

    def forward(self,
                s_j,
                dist,
                nbrs):

        phi = self.inv_dense(s_j)[nbrs[:, 1]]
        w_s = self.dist_embed(dist)
        output = phi * w_s

        # split into three components, so the tensor now has
        # shape n_atoms x 3 x n_atom_basis

        n_atom_basis = s_j.shape[-1]
        out_reshape = output.reshape(output.shape[0], 3, n_atom_basis)

        return out_reshape


class MessageBase(nn.Module):

    def forward(self,
                s_j,
                v_j,
                r_ij,
                nbrs):

        dist, unit = preprocess_r(r_ij)
        inv_out = self.inv_message(s_j=s_j,
                                   dist=dist,
                                   nbrs=nbrs)

        split_0 = inv_out[:, 0, :].unsqueeze(-1)
        split_1 = inv_out[:, 1, :]
        split_2 = inv_out[:, 2, :].unsqueeze(-1)

        unit_add = split_2 * unit.unsqueeze(1)
        delta_v_ij = unit_add + split_0 * v_j[nbrs[:, 1]]
        delta_s_ij = split_1

        # add results from neighbors of each node

        graph_size = s_j.shape[0]
        delta_v_i = scatter_add(src=delta_v_ij,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

        delta_s_i = scatter_add(src=delta_s_ij,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

        return delta_s_i, delta_v_i


class MessageBlock(MessageBase):
    def __init__(self,
                 n_atom_basis,
                 activation,
                 n_gaussians,
                 cutoff,
                 trainable_gauss,
                 dropout,
                 **kwargs):
        super().__init__()
        self.inv_message = InvariantMessage(n_atom_basis=n_atom_basis,
                                            activation=activation,
                                            n_gaussians=n_gaussians,
                                            cutoff=cutoff,
                                            trainable_gauss=trainable_gauss,
                                            dropout=dropout)

    def forward(self,
                s_j,
                v_j,
                r_ij,
                nbrs,
                **kwargs):

        dist, unit = preprocess_r(r_ij)
        inv_out = self.inv_message(s_j=s_j,
                                   dist=dist,
                                   nbrs=nbrs)

        split_0 = inv_out[:, 0, :].unsqueeze(-1)
        split_1 = inv_out[:, 1, :]
        split_2 = inv_out[:, 2, :].unsqueeze(-1)

        unit_add = split_2 * unit.unsqueeze(1)
        delta_v_ij = unit_add + split_0 * v_j[nbrs[:, 1]]
        delta_s_ij = split_1

        # add results from neighbors of each node

        graph_size = s_j.shape[0]
        delta_v_i = scatter_add(src=delta_v_ij,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

        delta_s_i = scatter_add(src=delta_s_ij,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

        return delta_s_i, delta_v_i


class UpdateBlock(nn.Module):
    def __init__(self,
                 n_atom_basis,
                 activation,
                 dropout):
        super().__init__()
        self.u_mat = Dense(in_features=n_atom_basis,
                           out_features=n_atom_basis,
                           bias=False)
        self.v_mat = Dense(in_features=n_atom_basis,
                           out_features=n_atom_basis,
                           bias=False)
        self.s_dense = nn.Sequential(Dense(in_features=2*n_atom_basis,
                                           out_features=n_atom_basis,
                                           bias=True,
                                           dropout_rate=dropout,
                                           activation=to_module(activation)),
                                     Dense(in_features=n_atom_basis,
                                           out_features=3*n_atom_basis,
                                           bias=True,
                                           dropout_rate=dropout))

    def forward(self,
                s_i,
                v_i):

        # v_i = (num_atoms, num_feats, 3)
        # v_i.transpose(1, 2).reshape(-1, v_i.shape[1])
        # = (num_atoms, 3, num_feats).reshape(-1, num_feats)
        # = (num_atoms * 3, num_feats)
        # -> So the same u gets applied to each atom
        # and for each of the three dimensions, but differently
        # for the different feature dimensions

        v_tranpose = v_i.transpose(1, 2).reshape(-1, v_i.shape[1])

        # now reshape it to (num_atoms, 3, num_feats) and transpose
        # to get (num_atoms, num_feats, 3)

        num_feats = v_i.shape[1]
        u_v = (self.u_mat(v_tranpose).reshape(-1, 3, num_feats)
               .transpose(1, 2))
        v_v = (self.v_mat(v_tranpose).reshape(-1, 3, num_feats)
               .transpose(1, 2))

        v_v_norm = norm(v_v)
        s_stack = torch.cat([s_i, v_v_norm], dim=-1)

        split = (self.s_dense(s_stack)
                 .reshape(s_i.shape[0], 3, -1))

        # delta v update
        a_vv = split[:, 0, :].unsqueeze(-1)
        delta_v_i = u_v * a_vv

        # delta s update
        a_sv = split[:, 1, :]
        a_ss = split[:, 2, :]

        inner = (u_v * v_v).sum(-1)
        delta_s_i = inner * a_sv + a_ss

        return delta_s_i, delta_v_i


class EmbeddingBlock(nn.Module):
    def __init__(self,
                 n_atom_basis):

        super().__init__()
        self.atom_embed = nn.Embedding(100, n_atom_basis, padding_idx=0)
        self.n_atom_basis = n_atom_basis

    def forward(self,
                z_number,
                **kwargs):

        num_atoms = z_number.shape[0]
        s_i = self.atom_embed(z_number)
        v_i = (torch.zeros(num_atoms, self.n_atom_basis, 3)
               .to(s_i.device))

        return s_i, v_i


class ReadoutBlock(nn.Module):
    def __init__(self,
                 n_atom_basis,
                 output_keys,
                 activation,
                 dropout,
                 means=None,
                 stddevs=None):
        super().__init__()

        self.readoutdict = nn.ModuleDict(
            {key: nn.Sequential(
                Dense(in_features=n_atom_basis,
                      out_features=n_atom_basis//2,
                      bias=True,
                      dropout_rate=dropout,
                      activation=to_module(activation)),
                Dense(in_features=n_atom_basis//2,
                      out_features=1,
                      bias=True,
                      dropout_rate=dropout))
             for key in output_keys}
        )

        self.scale_shift = ScaleShift(means=means,
                                      stddevs=stddevs)

    def forward(self, s_i):
        """
        Note: no atomwise summation. That's done in the model itself
        """

        results = {}

        for key, readoutdict in self.readoutdict.items():
            output = readoutdict(s_i)
            output = self.scale_shift(output, key)
            results[key] = output

        return results
