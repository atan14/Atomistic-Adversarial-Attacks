from torch import nn
from uq.utils.graphop import scatter_add
from uq.modules.painn import MessageBlock, UpdateBlock, EmbeddingBlock, ReadoutBlock
from uq.modules.schnet import (
    AttentionPool,
    SumPool,
    MolFpPool,
    MeanPool, get_rij,
    add_stress,
)


POOL_DIC = {
    "sum": SumPool,
    "mean": MeanPool,
    "attention": AttentionPool,
    "mol_fp": MolFpPool,
}


class PaiNN(nn.Module):
    def __init__(self, model_params):
        """
        Args:
            model_params (dict): dictionary of model parameters
        """

        super().__init__()

        n_atom_basis = model_params["n_atom_basis"]
        activation = model_params["activation"]
        n_gaussians = model_params["n_gaussians"]
        cutoff = model_params["cutoff"]
        n_convolutions = model_params["n_convolutions"]
        output_keys = model_params["output_keys"]
        trainable_gauss = model_params.get("trainable_gauss", False)
        dropout_rate = model_params.get("dropout_rate", 0)
        means = model_params.get("means")
        stddevs = model_params.get("stddevs")
        pool_dic = model_params.get("pool_dic")

        self.excl_vol = model_params.get("excl_vol", False)
        if self.excl_vol:
            self.power = model_params["V_ex_power"]
            self.sigma = model_params["V_ex_sigma"]

        self.grad_keys = model_params["grad_keys"]
        self.embed_block = EmbeddingBlock(n_atom_basis=n_atom_basis)
        self.message_blocks = nn.ModuleList(
            [
                MessageBlock(
                    n_atom_basis=n_atom_basis,
                    activation=activation,
                    n_gaussians=n_gaussians,
                    cutoff=cutoff,
                    trainable_gauss=trainable_gauss,
                    dropout=dropout_rate,
                )
                for _ in range(n_convolutions)
            ]
        )
        self.update_blocks = nn.ModuleList(
            [
                UpdateBlock(
                    n_atom_basis=n_atom_basis, activation=activation, dropout=dropout_rate
                )
                for _ in range(n_convolutions)
            ]
        )

        self.output_keys = output_keys
        # no skip connection in original paper
        self.skip = model_params.get(
            "skip_connection", {key: False for key in self.output_keys}
        )

        num_readouts = n_convolutions if any(self.skip.values()) else 1
        self.readout_blocks = nn.ModuleList(
            [
                ReadoutBlock(
                    n_atom_basis=n_atom_basis,
                    output_keys=output_keys,
                    activation=activation,
                    dropout=dropout_rate,
                    means=means,
                    stddevs=stddevs,
                )
                for _ in range(num_readouts)
            ]
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

        self.compute_delta = model_params.get("compute_delta", False)
        self.cutoff = cutoff

    def set_cutoff(self):
        if hasattr(self, "cutoff"):
            return
        msg = self.message_blocks[0]
        dist_embed = msg.inv_message.dist_embed
        self.cutoff = dist_embed.f_cut.cutoff

    def atomwise(self, batch, xyz=None):

        # for backwards compatability
        if isinstance(self.skip, bool):
            self.skip = {key: self.skip for key in self.output_keys}

        nbrs = batch["nbr_list"]
        nxyz = batch["nxyz"]

        if xyz is None:
            xyz = nxyz[:, 1:]
            if not xyz.requires_grad:
                xyz.requires_grad = True

        z_numbers = nxyz[:, 0].long()

        # get r_ij including offsets and excluding
        # anything in the neighbor skin
        self.set_cutoff()
        r_ij, nbrs = get_rij(xyz=xyz, batch=batch, nbrs=nbrs, cutoff=self.cutoff)

        s_i, v_i = self.embed_block(z_numbers, nbrs=nbrs, r_ij=r_ij)
        results = {}

        for i, message_block in enumerate(self.message_blocks):
            update_block = self.update_blocks[i]
            ds_message, dv_message = message_block(
                s_j=s_i, v_j=v_i, r_ij=r_ij, nbrs=nbrs
            )

            s_i = s_i + ds_message
            v_i = v_i + dv_message

            ds_update, dv_update = update_block(s_i=s_i, v_i=v_i)

            s_i = s_i + ds_update
            v_i = v_i + dv_update

            if not any(self.skip.values()):
                continue

            readout_block = self.readout_blocks[i]
            new_results = readout_block(s_i=s_i)
            for key, skip in self.skip.items():
                if not skip:
                    continue
                if key not in new_results:
                    continue
                if key in results:
                    results[key] += new_results[key]
                else:
                    results[key] = new_results[key]

        if not all(self.skip.values()):
            first_readout = self.readout_blocks[0]
            new_results = first_readout(s_i=s_i)
            for key, skip in self.skip.items():
                if key not in new_results:
                    continue
                if not skip:
                    results[key] = new_results[key]

        results["embedding"] = s_i

        return results, xyz, r_ij, nbrs

    def pool(self, batch, atomwise_out, xyz, r_ij, nbrs, inference=False):

        # import here to avoid circular imports
        from evi.utils.cuda import batch_detach

        if not hasattr(self, "output_keys"):
            self.output_keys = list(self.readout_blocks[0].readoutdict.keys())

        if not hasattr(self, "pool_dic"):
            self.pool_dic = {key: SumPool() for key in self.output_keys}

        all_results = {}

        for key in self.output_keys:
            if key not in self.pool_dic.keys():
                all_results[key] = atomwise_out[key]
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
                    atomwise_output=atomwise_out,
                    grad_keys=grad_keys,
                    out_keys=[key],
                )

                if inference:
                    results = batch_detach(results)
                all_results.update(results)

        return all_results, xyz

    def add_delta(self, all_results):
        for i, e_i in enumerate(self.output_keys):
            if i == 0:
                continue
            e_j = self.output_keys[i - 1]
            key = f"{e_i}_{e_j}_delta"
            all_results[key] = all_results[e_i] - all_results[e_j]
        return all_results

    def activation(self, results):
        activation = nn.Softplus()
        for key, value in results.items():
            if key in ["energy", "energy_grad", "stress", "embedding"]:
                continue
            results[key] = activation(value)
            if key == 'alpha':
                results[key] = results[key] + 1

        return results

    def V_ex(self, r_ij, nbr_list, xyz):

        dist = (r_ij).pow(2).sum(1).sqrt()
        potential = (dist.reciprocal() * self.sigma).pow(self.power)

        return scatter_add(potential, nbr_list[:, 0], dim_size=xyz.shape[0])[:, None]

    def run(self, batch, xyz=None, requires_stress=False, inference=False):

        atomwise_out, xyz, r_ij, nbrs = self.atomwise(batch=batch, xyz=xyz)

        if getattr(self, "excl_vol", None):
            # Excluded Volume interactions
            r_ex = self.V_ex(r_ij, nbrs, xyz)
            atomwise_out["energy"] += r_ex

        all_results, xyz = self.pool(
            batch=batch,
            atomwise_out=atomwise_out,
            xyz=xyz,
            r_ij=r_ij,
            nbrs=nbrs,
            inference=inference,
        )

        if requires_stress:
            all_results = add_stress(
                batch=batch, all_results=all_results, nbrs=nbrs, r_ij=r_ij
            )

        if getattr(self, "compute_delta", False):
            all_results = self.add_delta(all_results)

        all_results = self.activation(all_results)

        return all_results, xyz

    def forward(
        self, batch, xyz=None, requires_stress=False, inference=False, **kwargs
    ):
        """
        Call the model
        Args:
            batch (dict): batch dictionary
        Returns:
            results (dict): dictionary of predictions
        """

        results, _ = self.run(
            batch=batch, xyz=xyz, requires_stress=requires_stress, inference=inference
        )

        return results
