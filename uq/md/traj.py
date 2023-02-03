import json
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import Trajectory
from pymatgen.core import Structure
from json import JSONDecodeError


class MDTraj:
    def __init__(
        self,
        filename,
    ):
        self.filename = filename
        self.mdparams = self.read_mdparams()
        self.trajectory = self.read_trajectory()
        self.log = self.read_trajlog()
        self.pbc = np.array(self.trajectory.pbc).any()

    def __len__(self):
        return len(self.trajectory)

    def __getitem__(self, index):
        return self.trajectory[index]

    def read_mdparams(self):
        mdparams = json.load(open(f"{self.filename}.json", "r"))
        return mdparams

    def read_trajectory(self):
        traj = Trajectory(f"{self.filename}.traj")
        return traj

    def read_trajlog(self):
        log = pd.read_csv(f"{self.filename}.log", sep="\s+")
        try:
            log = log.astype("float")
        except ValueError:
            mask = pd.to_numeric(log["Time[ps]"], errors="coerce").isna()
            log = log.drop(mask[mask].index)
        log = log.replace({"NaN": None})
        log = log.astype(float)
        return log

    def get_atoms(self, atoms):
        if self.pbc:
            atoms = Structure(
                lattice=atoms.cell,
                coords=atoms.get_positions(),
                species=atoms.get_atomic_numbers(),
                coords_are_cartesian=True,
            )
        return atoms

    def get_all_distances(self, atoms):
        if isinstance(atoms, Atoms):
            return atoms.get_all_distances(mic=True)
        if isinstance(atoms, Structure):
            return atoms.distance_matrix
        raise Exception("Atoms is neither ase.Atoms or pymatgen.Structure object")

    def _get_traj_explosion(
        self,
        min_r=0.75,
        max_r=2.25,
        nbr_list=None,
        max_idx=None,
    ):
        """
        min_r (float): Minimum bond threshold to determine if system exploded.
        max_r (float): Maximum bond threshold to determine if system exploded.
        """
        for frame in range(len(self.trajectory)):
            time = self.log.loc[frame]['Time[ps]']
            if frame > max_idx:
                return max_idx, time
            try:
                atoms = self.get_atoms(self.trajectory[frame])
                dist = self.get_all_distances(atoms)
                mask = (dist <= min_r) | (dist >= max_r)
                np.fill_diagonal(mask, 0)
                if mask.any():
                    return frame, time
            except (MemoryError, ValueError, JSONDecodeError) as error:
                print(error, "at frame", frame)
                return None
        frame = self.log.index[-1]
        return frame, time

    def _get_energy_explosion(
        self,
        min_ke=1.0,
        max_ke=400.0,
        min_pe=1.0,
        max_pe=400.0,
    ):
        pe, ke = self.log["Epot[eV]"], self.log["Ekin[eV]"]
        ke_i = ke.where((ke < min_ke) | (ke > max_ke)).first_valid_index()
        if ke_i is None:
            ke_i = self.log.index[-1]
        pe_i = pe.where((pe < min_pe) | (pe > max_pe)).first_valid_index()
        if pe_i is None:
            pe_i = self.log.index[-1]
        min_idx = np.min([ke_i, pe_i])
        time = self.log.loc[min_idx]["Time[ps]"]
        return min_idx, time

    def get_explosion_time(
        self,
        min_ke=1.0,
        max_ke=400.0,
        min_pe=1.0,
        max_pe=400.0,
        min_r=0.75,
        max_r=2.25,
        nbr_list=None,
    ):
        min_idx, energy_t = self._get_energy_explosion(
            min_ke=min_ke,
            max_ke=max_ke,
            min_pe=min_pe,
            max_pe=max_pe,
        )
        frame_i, traj_t = self._get_traj_explosion(
            min_r=min_r, max_r=max_r, max_idx=min_idx, nbr_list=nbr_list
        )
        min_idx = np.min([min_idx, frame_i])
        return min_idx, self.log.loc[min_idx]["Time[ps]"]

    def get_time(self):
        return self.log["Time[ps]"].to_numpy()

    def get_temperature(self):
        return self.log["T[K]"].to_numpy()

    def get_total_energy(self):
        return self.log["Etot[eV]"].to_numpy()

    def get_potential_energy(self):
        return self.log["Epot[eV]"].to_numpy()

    def get_kinetic_energy(self):
        return self.log["Ekin[eV]"].to_numpy()
