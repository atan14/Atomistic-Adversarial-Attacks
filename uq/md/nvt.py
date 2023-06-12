#!/usr/bin/env python
import os
import json
import time
import torch
import argparse
import numpy as np

from uq.data import Dataset, AtomsBatch
from uq.utils import make_dir, write_params

from ase.md.verlet import VelocityVerlet

from nff.io import NeuralFF, EnsembleNFF
from nff.md.nve import Dynamics
from nff.md.nvt import NoseHoover


THERMOSTAT = {
    "velocityverlet": VelocityVerlet,
    "nosehoover": NoseHoover,
}


class MdRunner:
    def __init__(
        self,
        dataset,
        model_dir,
        outdir=None,
        idx=None,
        temperature=300,  # K
        time=5,  # ps
        device="cuda:2",
        cutoff=5.0,
        save_freq=40,
        timestep=0.5,  # fs
        cutoff_skin=1.0,
        ttime=25.0,
        nbr_update_freq=5,
        requires_large_offsets=False,
        stop_if_error=True,
    ):
        self.temperature = temperature
        self.time = time
        self.device = device
        self.cutoff = cutoff
        self.cutoff_skin = cutoff_skin
        self.save_freq = save_freq
        self.timestep = timestep
        self.numsteps = self.get_num_steps()
        self.dset = self.load_dataset(dataset)
        self.idx = self.get_idx(idx)
        self.model_dir = model_dir
        self.calculator = self.get_calc(model_dir)
        self.outdir = self.get_outdir(outdir)
        self.check_directed()
        self.ttime = ttime
        self.nbr_update_freq = nbr_update_freq
        self.requires_large_offsets = requires_large_offsets
        self.stop_if_error = stop_if_error

    def load_dataset(self, dataset):
        return Dataset.from_file(dataset)

    def get_idx(self, idx):
        if idx is None:
            return np.random.choice(len(self.dset))
        else:
            return idx

    def get_num_steps(self):
        return int(self.time * 1000 / self.timestep)

    def get_calc(self, model_dir):
        model = torch.load(f"{model_dir}/best_model", map_location=self.device)
        calculator = EnsembleNFF(model, device=self.device)
        return calculator

    def get_outdir(self, outdir):
        if outdir is None:
            basedir = self.model_dir.replace("models", "md/nvt")
            idx = basedir.find("emae_f")
            basedir = basedir.replace(basedir[idx : idx + 10], "")
            basedir = os.path.join(basedir, f"{self.temperature}K_{self.time:.1f}ps")

            count = 0
            outdir = os.path.join(basedir, f"{count:03d}")
            while os.path.exists(f"{outdir}.log"):
                count += 1
                outdir = os.path.join(basedir, f"{count:03d}")

        return make_dir(outdir)

    def get_md_params(self):
        md_params = {
            "thermostat": "nosehoover",  # or Langevin or NPT or NVT or Thermodynamic Integration
            "thermostat_params": {
                "timestep": self.timestep,  # fs
                "temperature": self.temperature,
                "ttime": self.ttime,
            },
            "T_init": self.temperature,
            "steps": self.numsteps,
            "save_frequency": self.save_freq,
            "nbr_list_update_freq": self.nbr_update_freq,
            "thermo_filename": f"{self.outdir}.log",
            "traj_filename": f"{self.outdir}.traj",
            "skip": 0,
            "dset_idx": self.idx,
            "stop_if_error": self.stop_if_error,
        }
        write_params(md_params, f"{self.outdir}.json")
        md_params["thermostat"] = THERMOSTAT[md_params["thermostat"]]
        return md_params

    def check_directed(self):
        params = json.load(open(f"{self.model_dir}/params.json", "r"))
        model_type = params["model"]["model_type"]
        if model_type in ["painn"]:
            self.directed = True
        else:
            self.directed = False

    def get_md_atoms(self):
        props = self.dset[self.idx]

        positions = props["nxyz"][:, 1:]
        numbers = props["nxyz"][:, 0]
        lattice = props.get("lattice", None)
        pbc = lattice is not None

        atoms = AtomsBatch(
            positions=positions,
            numbers=numbers,
            cell=lattice,
            pbc=pbc,
            cutoff=self.cutoff,
            cutoff_skin=self.cutoff_skin,
            props={"energy": 0, "energy_grad": []},
            calculator=self.calculator,
            directed=self.directed,
            requires_large_offsets=self.requires_large_offsets,
        )
        _ = atoms.update_nbr_list()

        return atoms

    def run(self):
        atoms = self.get_md_atoms()
        md_params = self.get_md_params()
        dyn = Dynamics(atoms, md_params)
        dyn.run()


def get_args():
    parser = argparse.ArgumentParser(description="Run NFF MD several times")

    parser.add_argument(
        "-D",
        "--dataset",
        type=str,
        help="dataset path (default: %(default)s)",
    )
    parser.add_argument(
        "-i",
        "--idx",
        type=int,
        default=None,
        help="index to the configuration in dataset",
    )
    parser.add_argument(
        "-m",
        "--model_dir",
        help="path from where the models will be loaded",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=None,
        help="path where the data will be stored",
    )
    parser.add_argument(
        "-T",
        "--temperature",
        type=float,
        default=300,  # K
        help="Initial temperature (in K) (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=float,
        default=5,  # ps
        help="simulation time (in ps) (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda",
        type=str,
        help="device (default: %(default))",
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        type=float,
        default=5.0,
        help="cutoff for neighbor list (default: %(default)A)",
    )
    parser.add_argument(
        "-f",
        "--save_freq",
        type=int,
        default=40,
        help="The save frequency of the trajectory",
    )
    parser.add_argument(
        "-s",
        "--timestep",
        type=float,
        default=0.5,
        help="Timestep for running md (default: %(default)fs)",
    )
    parser.add_argument(
        "-n",
        "--nbr_update_freq",
        type=int,
        default=5,
        help="Frequency of updating neighbors list (default: %(default))",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    md = MdRunner(
        dataset=args.dataset,
        idx=args.idx,
        model_dir=args.model_dir,
        outdir=args.outdir,
        temperature=args.temperature,  # K
        time=args.time,  # ps
        device=args.device,
        cutoff=args.cutoff,
        save_freq=args.save_freq,
        timestep=args.timestep,  # fs
    )
    start_time = time.time()
    md.run()
    end_time = time.time()
    duration = end_time - start_time
    print(f"Duration: {duration}")
    print("Done.")
