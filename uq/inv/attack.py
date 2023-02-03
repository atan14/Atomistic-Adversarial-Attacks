import os
import sys
import json
import torch
import argparse
import warnings
import numpy as np
import pickle5 as pickle
from pymatgen.core import Structure
from ase import Atoms
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.mixture import GaussianMixture

from evi.data import Dataset, concatenate_dict, collate_dicts, wrap_cell, AtomsBatch, densify_tensor
from evi.models import SchNet, PaiNN, Ensemble
from evi.utils import make_dir, batch_to, batch_detach
from adv import AdvEvidential, AdvMVE, AdvEnsemble, AdvGMM


ADV_DICT = {
    "evidential": AdvEvidential,
    "mve": AdvMVE,
    "ensemble": AdvEnsemble,
    "gmm": AdvGMM,
}


class Attack:
    """
    Class to do adversarial attack
    """

    def __init__(
        self,
        sampling_params,
        dset,
        model_dir,
        model=None,
        device="cuda:1",
        batch_size=8,
        cutoff=5.0,
        directed=False,
        offset_key="offsets",
        nbr_key="nbr_list",
        min_value=None,
        pbc=torch.tensor([1, 1, 1]),
        file_type=None,
        ignore_error=False,
        use_pymatgen_nbrlist=False,
    ):
        """
        Args:
            sampling_params (dict): parameters for the adversarial sampling method
            dset (evi.data.Dataset): initial dataset as seed for attack points
            model (torch.nn.modules): a PyTorch neural network model object to be attacked
            model_dir (str): directory where the model and log files are stored
            device (str, optional): the device to run the model and adversarial sampling on
            batch_size (int, optional): batch size to use when evaluating the model
            cutoff (float, optional): cutoff radius for determining the neighbors of each atom
            directed (bool, optional): whether direction of bonds matter (for graphs)
            offset_key (str, optional): key for the offset data in the dataset
            nbr_key (str, optional): key for the neighbor list data in the dataset
            min_value (float, optional): a minimum value for evidential parameters to prevent gradient explosion (only used for deep evidential regression)
            pbc (torch.tensor or bool): periodic boundary conditions. A value of True will give periodic boundary conditions along all 3 axes. Give sequence of three bool values to specify pbc along specific axes.
            file_type (str, optional): file format (xyz or cif) to use when saving the adversarial examples
            ignore_error (bool, optional): indicate whether to ignore errors that occur when performing adversarial sampling
            use_pymatgen_nbrlist (bool, optional): whether to use pymatgen library to calculate the neighbor list (slower, but does not give CUDA out of memory error)
        """
        self.sampling_params = sampling_params
        self.model_dir = model_dir
        self.device = device
        self.dset = dset
        self.model = model
        self.adv_fn = self.get_adv_fn()
        self.batch_size = batch_size
        self.cutoff = cutoff
        self.directed = directed
        self.offset_key = offset_key
        self.nbr_key = nbr_key
        self.min_value = min_value
        self.pbc = pbc
        self.system_type = self.check_if_toy()
        self.file_type = self.check_file_type(file_type)
        self.ignore_error = ignore_error
        self.use_pymatgen_nbrlist = use_pymatgen_nbrlist

    def check_if_toy(self):
        """
        method to check if the system being attacked is a toy or a materials system
        """
        if all([self.nbr_key in d for d in self.dset]) and all([self.offset_key in d for d in self.dset]):
            return "mat"
        else:
            return "toy"

    def check_file_type(self, file_type):
        """
        method to check if the specified file format for saving the adversarial examples is supported
        """
        if (self.system_type == "mat") and (file_type is not None) and (file_type not in ["xyz", "cif"]):
            warnings.warn(
                f"{file_type} not available. Attack systems will be saved as default file types"
            )
        return file_type

    def log(self, log, dry_run=False):
        """
        method to log the progress of the attack
        """
        if not dry_run:
            f = open(f"{self.model_dir}/attack.out", "a+")
            write = lambda x: f.write(x + "\n")
            write(log)
        print(log)

    def get_adv_fn(self):
        """
        method to get the adversarial sampling function based on the method specified in the `sampling_params` dictionary
        """
        return ADV_DICT[self.sampling_params["adv_fn"]](
            train=self.dset, **self.sampling_params
        )

    def get_gmm_model(self):
        """
        method to fit a Gaussian mixture model to the embedding of the model on the training data
        """
        split_inds = pickle.load(open(f"{self.model_dir}/split_inds.pkl", "rb"))
        train = self.dset[split_inds['train']]
        train = Dataset(concatenate_dict(*train))
        train_loader = DataLoader(train, batch_size=self.batch_size, collate_fn=collate_dicts)

        train_embedding = []
        for batch in train_loader:
            batch = batch_to(batch, device=self.device)
            pred = self.model(batch)
            pred = batch_detach(pred)
            batch = batch_detach(batch)
            train_embedding.append(pred['embedding'].squeeze().detach().cpu())
            del pred
            del batch
        train_embedding = torch.cat(train_embedding, dim=0)

        gm_model = GaussianMixture(n_components=self.sampling_params['n_clusters'])
        gm_model.fit(train_embedding)
        return gm_model

    def get_seed_loader(self):
        """
        method to get seed for the attack using a random number generator
        """
        randperm = torch.randperm(len(self.dset))[: self.sampling_params["num_attacks"]]
        seed_dset = self.dset[randperm.tolist()]
        seed_dset = Dataset(concatenate_dict(*seed_dset))

        loader = DataLoader(
            seed_dset,
            batch_size=self.batch_size,
            collate_fn=collate_dicts,
        )
        return loader

    def get_model(self):
        """
        method to load trained model from the specified `model_dir`
        """
        if self.model is None:
            try:
                self.model = torch.load(
                    f"{self.model_dir}/best_model", map_location=self.device
                )
            except:
                from evi.train import TrainPipeline
                params = json.load(open(f"{self.model_dir}/params.json", "r"))
                trainer = TrainPipeline(params)
                state_dict = torch.load(f"{self.model_dir}/best_model.pth.tar")
                trainer._load_model_state_dict(state_dict['model'])
                self.model = trainer._model.to(self.device)

    def get_delta(self, batch):
        """
        method to return a zero-filled delta (perturbation) tensor
        """
        delta = torch.zeros_like(
            batch["nxyz"][:, 1:],
            requires_grad=True,
            device=self.device,
        )
        return delta

    def split_batched_props(self, batch, key):
        """
        method to split batched properties into individual properties
        """
        if "num_atoms" not in batch:
            raise Exception("Key 'num_atoms' does not exist in batch")

        num_atoms = batch['num_atoms'].tolist()
        num_systems = len(num_atoms)

        if key not in batch:
            warnings.warn(f"{key} does not exist!")
            return [None] * num_systems

        if key == "lattice":
            return torch.split(batch[key], [3] * num_systems)
        if key in ["nbr_list", "offsets"]:
            nbr_list = batch["nbr_list"]
            cumulative_atoms = np.cumsum([0] + num_atoms)
            nbr_inds = [0]
            for i, j in zip(cumulative_atoms[:-1], cumulative_atoms[1:]):
                isin = ((nbr_list > i) & (nbr_list < j)).sum(1)
                inds = np.sort(torch.where(isin > 0)[0])
                nbr_inds.append((inds.max() + 1))

            new_list = []
            for i, (idx_i, idx_j) in enumerate(zip(nbr_inds[:-1], nbr_inds[1:])):
                if key == "nbr_list":
                    b = batch[key][idx_i:idx_j]
                    b = b - cumulative_atoms[i]
                if key == "offsets":
                    b = batch[key][idx_i:idx_j]
                new_list.append(b)

            return new_list

        return torch.split(batch[key], num_atoms)

    def get_atoms_list(self, all_batches):
        """
        method to get a list of individual materials systems where each element contains the properties of one system
        """
        systems_list = []
        idx = 0
        for i, batch in enumerate(all_batches):
            d = {}
            for prop in ['nxyz', 'lattice', 'nbr_list', 'offsets']:
                if prop in batch:
                    d[prop] = self.split_batched_props(batch, prop)
            for j, n in enumerate(batch['num_atoms'].tolist()):
                idtf = f"{self.model_dir[self.model_dir.rfind('models/')+7:]}/{idx:03d}"
                data = {
                    "nxyz": d["nxyz"][j],
                    "num_atoms": torch.tensor([n]),
                    "identifier": idtf,
                }
                for prop in ['lattice', 'nbr_list', 'offsets']:
                    if prop in d:
                        data[prop] = d[prop][j]

                systems_list.append(data)
                idx += 1
        return systems_list

    def reindex_nbrlist(self, batch, nbr_list):
        """
        method to reorganize index of neighbor list when batched properties are splitted into individual properties
        """
        num_atoms = batch['num_atoms'].tolist()
        cumulative_atoms = np.cumsum([0] + num_atoms)
        for i, (n, nbr) in enumerate(zip(cumulative_atoms, nbr_list)):
            nbr_list[i] = nbr + int(n)
        return nbr_list

    def _get_neighbor_list(self, batch):
        """
        method to get neighbor list for the materials systems
        """
        if "lattice" not in batch:
            return batch

        nxyz_list = self.split_batched_props(batch, "nxyz")
        lattice_list = self.split_batched_props(batch, "lattice")

        nbr_list, offsets_list = [], []
        for nxyz, lattice in zip(nxyz_list, lattice_list):
            atoms = AtomsBatch(
                numbers=nxyz[:, 0].long().detach().cpu(),
                positions=nxyz[:, 1:].detach().cpu(),
                cell=lattice.detach().cpu(),
                pbc=self.pbc.tolist(),
                cutoff=self.cutoff,
                directed=self.directed,
                props={},
                device=self.device,
                use_pymatgen_nbrlist=self.use_pymatgen_nbrlist,
            )
            nbrs, offs = atoms.update_nbr_list()
            if offs.layout == torch.sparse_coo:
                offs = densify_tensor(offs)
            nbr_list.append(nbrs)
            offsets_list.append(offs)

        nbr_list = self.reindex_nbrlist(batch, nbr_list)
        batch[self.nbr_key] = torch.cat(nbr_list, 0)
        batch[self.offset_key] = torch.cat(offsets_list, 0)

        return batch

    def _wrap_cell(self, batch):
        """
        method to wrap positions of atoms in the lattice cell
        """
        if "lattice" not in batch:
            return batch

        nxyz_list = self.split_batched_props(batch, "nxyz")
        lattice_list = self.split_batched_props(batch, "lattice")

        new_nxyz_list = []
        for nxyz, lattice in zip(nxyz_list, lattice_list):
            n = nxyz[:, 0]
            new_xyz = wrap_cell(
                lattice=lattice, xyz=nxyz[:, 1:], pbc=self.pbc.to(self.device)
            )
            new_nxyz = torch.cat([n.reshape(-1, 1), new_xyz], 1)
            new_nxyz_list.append(new_nxyz)
        batch["nxyz"] = torch.cat(new_nxyz_list, 0)
        return batch

    def update_batch(self, batch, delta):
        """
        method to update coordinates and neighbor list of system after every iteration of adversarial attack
        """
        batch["nxyz"] = batch["nxyz"] + torch.cat(
            [torch.zeros((len(batch["nxyz"]), 1), device=self.device), delta], dim=1
        )
        if self.system_type == "mat":
            batch = self._wrap_cell(batch)
            batch = self._get_neighbor_list(batch)
        batch = batch_to(batch, self.device)
        return batch

    def get_adv_dir(self):
        """
        method to return the directory where files of adversarial examples can be stored
        """
        unique_id = self.model_dir[self.model_dir.rfind("/")+1:self.model_dir.rfind("_")]
        gen = int(self.model_dir[self.model_dir.rfind("_")+1:])
        adv_dir = self.model_dir.split("models/")[0] + f"adv/inbox/{unique_id}_{gen}"

        return make_dir(adv_dir)

    def save_xyz_files(self, system, path):
        """
        method to save materials systems in the xyz file format
        """
        nxyz = system["nxyz"]
        mol = Atoms(
            numbers=nxyz[:, 0],
            positions=nxyz[:, 1:],
        )
        mol.write(path)
        self.log(f"Saved to {path}")

    def save_cif_files(self, system, path):
        """
        method to save materials system in the CIF file format
        """
        struc = Structure(
            species=system["nxyz"][:, 0],
            coords=system["nxyz"][:, 1:],
            lattice=system["lattice"],
            coords_are_cartesian=True,
        )
        with open(path, "w") as f:
            f.write(struc.to("cif"))
            f.close()
        self.log(f"Saved to {path}")

    def save_individual_files(self, systems_list):
        """
        method to loop through the list of materials systems and save to specified file format (xyz or CIF)
        """
        adv_dir = self.get_adv_dir()
        for i, system in enumerate(systems_list):
            path = f"{adv_dir}_{i:03d}"
            if self.file_type and (self.file_type in ["xyz", "cif"]):
                if self.file_type == "xyz":
                    self.save_xyz_files(system, f"{path}.xyz")
                elif self.file_type == "cif":
                    self.save_cif_files(system, f"{path}.cif")
            else:
                if "lattice" in system:
                    self.save_cif_files(system, f"{path}.cif")
                else:
                    self.save_xyz_files(system, f"{path}.xyz")
        return systems_list

    def save_as_dset(self, systems_list):
        """
        method to save dataset containing adversarial examples
        """
        dset = Dataset(concatenate_dict(*systems_list))
        dset.save(f"{self.model_dir}/attacks.pth.tar")
        self.log(f"Attack dataset saved to {self.model_dir}/attacks.pth.tar")

    def attack(self, dry_run=False):
        """
        method to perform adversarial attack on the model
        """
        self.log("Epoch |   AdvLoss   |   Delta   | Energy Mean | Force Mean ", dry_run)

        loader = self.get_seed_loader()
        self.get_model()
        self.model.eval()

        scaler = GradScaler()

        if self.sampling_params['adv_fn'] == 'gmm':
            self.adv_fn.gm_model = self.get_gmm_model()

        all_batches = []
        for i, batch in enumerate(loader):
            batch = batch_to(batch, self.device)

            delta = self.get_delta(batch)
            opt = torch.optim.Adam([delta], lr=self.sampling_params["lr"])

            for epoch in range(self.sampling_params["n_epochs"]):
                opt.zero_grad()

                with autocast():
                    batch = self.update_batch(batch, delta)
                    results = self.model(batch)

                    loss = self.adv_fn(
                        results,
                        num_atoms=batch['num_atoms'],
                        min_value=self.min_value,
                    )

                scaler.scale(loss).backward(retain_graph=True)
                scaler.step(opt)
                scaler.update()

                results = batch_detach(results)

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    if self.ignore_error:
                        batch = None
                        break
                    else:
                        raise Exception("Loss contains NaN or Inf values")

                if torch.allclose(loss, torch.zeros_like(loss), atol=1e-8):
                    if self.ignore_error:
                        break
                    else:
                        raise Exception("Loss is exactly zero for all points")

                if epoch % 5 == 0:
                    self.log(
                        "{:>5d} | {:>1.4e} | {:>1.3e} | {:>1.4e} | {:>1.3e} ".format(
                            epoch,
                            loss.item(),
                            delta.abs().sum().item(),
                            results["energy"].mean().item(),
                            results["energy_grad"].mean().item(),
                        ),
                        dry_run=dry_run,
                    )

                del results

            if batch is not None:
                batch = batch_detach(batch)
                all_batches.append(batch)

                del batch

        self.model = self.model.cpu()

        if not dry_run:
            if self.system_type == "mat":
                systems_list = self.get_atoms_list(all_batches)
                systems_list = self.save_individual_files(systems_list)
            self.save_as_dset(systems_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("params_path", type=str, help="Path to parameter file to run")
    parser.add_argument("--cutoff", type=float, default=5.0)
    #parser.add_argument("--directed", action="store_false")
    parser.add_argument("--offset_key", type=str, default="offsets")
    parser.add_argument("--nbr_key", type=str, default="nbr_list")
    parser.add_argument("--min_value", type=float, default=1e-6)
    parser.add_argument("--pbc", nargs="+", type=int, default=[1, 1, 1])
    parser.add_argument("--file_type", type=str, default=None, choices=[None, "xyz", "cif"])
    parser.add_argument("--dry_run", action='store_true', default=False)
    parser.add_argument("--ignore_error", action="store_true", default=False)
    parser.add_argument("--use_pymatgen_nbrlist", action='store_true', default=False)
    args = parser.parse_args()

    params = json.load(open(args.params_path, "r"))
    sampling_params = params['sampling']
    dset = Dataset.from_file(params['dset']['path'])
    device = params['train']['device']
    model_dir = params['model']['outdir']
    batch_size = params['dset']['batch_size']
    pbc = torch.Tensor(args.pbc)

    print(f"Start adversarial attack on {args.params_path}.")
    attacker = Attack(
        sampling_params=sampling_params,
        dset=dset,
        model=None,
        model_dir=model_dir,
        device=device,
        batch_size=batch_size,
        cutoff=args.cutoff,
        #directed=args.directed,
        directed=True,
        offset_key=args.offset_key,
        nbr_key=args.nbr_key,
        min_value=args.min_value,
        pbc=pbc,
        file_type=args.file_type,
        ignore_error=args.ignore_error,
        use_pymatgen_nbrlist=args.use_pymatgen_nbrlist,
    )
    attacker.attack(args.dry_run)

    if not args.dry_run:
        if "running" in args.params_path:
            new_params_path = args.params_path.replace("running", "pending")
        else:
            idx = args.params_path.rfind("/")+1
            new_params_path = f"{args.params_path[:idx]}/pending/{args.params_path[idx:]}"
        os.rename(args.params_path, make_dir(new_params_path))

    print(f"Done attacking on {args.params_path}.")
    sys.exit(0)
