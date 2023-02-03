from torch import nn
from evi.modules import (
    SchNetConv,
    NodeMultiTaskReadOut,
    get_default_readout,
    SumPool,
    MeanPool,
    AttentionPool,
    MolFpPool,
    get_rij,
    add_stress,
)


POOL_DIC = {
    "sum": SumPool,
    "mean": MeanPool,
    "attention": AttentionPool,
    "mol_fp": MolFpPool,
}


class SchNet(nn.Module):
    def __init__(self, model_params):
        nn.Module.__init__(self)

        n_atom_basis = model_params["n_atom_basis"]
        n_filters = model_params["n_filters"]
        n_gaussians = model_params["n_gaussians"]
        n_convolutions = model_params["n_convolutions"]
        cutoff = model_params["cutoff"]
        activation = model_params.get("activation", "shifted_softplus")
        trainable_gauss = model_params.get("trainable_gauss", False)
        dropout_rate = model_params.get("dropout_rate", 0.0)
        num_readout_layer = model_params.get("num_readout_layer", {'energy': 1})
        pool_dic = model_params.get("pool_dic")

        self.output_keys = model_params["output_keys"]
        self.grad_keys = model_params["grad_keys"]

        self.atom_embed = nn.Embedding(100, n_atom_basis, padding_idx=0)

        readoutdict = model_params.get(
            "readoutdict",
            get_default_readout(
                n_atom_basis=n_atom_basis,
                num_readout_layer=num_readout_layer,
                output_keys=self.output_keys,
                activation=activation,
            ),
        )
        post_readout = model_params.get("post_readout", None)

        # convolutions
        self.convolutions = nn.ModuleList(
            [
                SchNetConv(
                    n_atom_basis=n_atom_basis,
                    n_filters=n_filters,
                    n_gaussians=n_gaussians,
                    cutoff=cutoff,
                    trainable_gauss=trainable_gauss,
                    dropout_rate=dropout_rate,
                )
                for _ in range(n_convolutions)
            ]
        )

        # ReadOut
        self.atomwisereadout = NodeMultiTaskReadOut(
            multitaskdict=readoutdict, post_readout=post_readout
        )

        if pool_dic is None:
            self.pool_dic = {key: SumPool() for key in self.output_keys}
        else:
            self.pool_dic = nn.ModuleDict({})
            for out_key, sub_dic in pool_dic.items():
                if out_key not in self.output_keys:
                    continue
                pool_name = sub_dic["name"].lower()
                kwargs = sub_dic["param"]
                pool_class = POOL_DIC[pool_name]
                self.pool_dic[out_key] = pool_class(**kwargs)

        self.device = None
        self.cutoff = cutoff

    def set_cutoff(self):
        if hasattr(self, "cutoff"):
            return
        gauss_centers = (
            self.convolutions[0].moduledict["message_edge_filter"][0].offsets
        )
        self.cutoff = gauss_centers[-1] - gauss_centers[0]

    def convolve(self, batch, xyz=None):
        if xyz is None:
            xyz = batch["nxyz"][:, 1:4]
            if not xyz.requires_grad and xyz.grad_fn is None:
                xyz.requires_grad = True

        r = batch["nxyz"][:, 0]
        N = batch["num_atoms"].reshape(-1).tolist()
        a = batch["nbr_list"]

        # get r_ij including offsets and excluding
        # anything in the neighbor skin
        self.set_cutoff()
        r_ij, a = get_rij(xyz=xyz, batch=batch, nbrs=a, cutoff=self.cutoff)
        dist = r_ij.pow(2).sum(1).sqrt()
        e = dist[:, None]

        # ensuring image atoms have the same vectors of their corresponding
        # atom inside the unit cell
        r = self.atom_embed(r.long()).squeeze()

        # update function includes periodic boundary conditions
        for i, conv in enumerate(self.convolutions):
            dr = conv(r=r, e=e, a=a)
            r = r + dr

        return r, N, xyz, r_ij, a

    def activation(self, results):
        activation = nn.Softplus()
        for key, value in results.items():
            if key in ["energy", "energy_grad", "stress"]:
                continue
            results[key] = activation(value)
            if key == 'alpha':
                results[key] = results[key] + 1

        return results

    def pool(self, batch, atomwise_output, xyz, r_ij, nbrs):
        all_results = {}

        for key in self.output_keys:
            if key not in self.pool_dic.keys():
                all_results[key] = atomwise_output[key]
            else:
                pool_obj = self.pool_dic[key]
                grad_key = f"{key}_grad"
                grad_keys = [grad_key] if (grad_key in self.grad_keys) else []
                if "stress" in self.grad_keys and "stress" not in all_results:
                    grad_keys.append("stress")
                results = pool_obj(
                    batch=batch,
                    xyz=xyz,
                    r_ij=r_ij,
                    nbrs=nbrs,
                    atomwise_output=atomwise_output,
                    grad_keys=grad_keys,
                    out_keys=[key],
                )
                all_results.update(results)

        return all_results, xyz

    def forward(self, batch, xyz=None, requires_stress=False, **kwargs):
        r, N, xyz, r_ij, nbrs = self.convolve(batch, xyz)
        r = self.atomwisereadout(r)

        results, xyz = self.pool(
            batch=batch,
            atomwise_output=r,
            xyz=xyz,
            r_ij=r_ij,
            nbrs=nbrs,
        )
        if requires_stress:
            results = add_stress(batch=batch, all_results=results, nbrs=nbrs, r_ij=r_ij)
        results = self.activation(results)

        return results
