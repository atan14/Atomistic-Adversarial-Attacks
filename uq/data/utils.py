import numpy as np
import torch

from ase import Atoms, units
from pymatgen.core import Structure

from evi.utils.graphop import split_and_sum


DEFAULT_CUTOFF = 5.0
DEFAULT_DIRECTED = False
DEFAULT_SKIN = 1.0


def sparsify_tensor(tensor):
    """Convert a torch.Tensor into a torch.sparse.FloatTensor

    Args:
        tensor (torch.Tensor)

    returns:
        sparse (torch.sparse.Tensor)
    """
    ij = tensor.nonzero(as_tuple=False)

    if len(ij) > 0:
        v = tensor[ij[:, 0], ij[:, 1]]
        return torch.sparse.FloatTensor(ij.t(), v, tensor.size())
    else:
        return 0


def sparsify_array(array):
    """Convert a np.array into a torch.sparse.FloatTensor

    Args:
        array (np.array)

    returns:
        sparse (torch.sparse.Tensor)
    """
    return sparsify_tensor(torch.FloatTensor(array))


def densify_tensor(tensor):
    try:
        return tensor.to_dense()
    except RuntimeError:
        return tensor


def densify_dataset(dataset):
    try:
        dataset.props['offsets'] = [densify_tensor(d['offsets']) for d in dataset]
    except RuntimeError:
        print("Could not densify offsets values of dataset")
    return dataset


def check_directed(model, atoms):
    model_cls = model.__class__.__name__
    msg = f"{model_cls} needs a directed neighbor list"
    assert atoms.directed, msg


def wrap_cell(
    lattice: torch.tensor, xyz: torch.tensor, pbc: torch.tensor
) -> torch.tensor:
    """
    NOTE: this function is copied from the NNP package (https://aiqm.github.io/nnp-test-docs/code/pbc.html)
    Map atoms outside the unit cell into the cell using PBC.

    Arguments:
        lattice: tensor of shape ``(3, 3)`` of the three vectors defining unit cell:

            .. code-block:: python

                tensor([[x1, y1, z1],
                        [x2, y2, z2],
                        [x3, y3, z3]])

        xyz: Tensor of shape ``(atoms, 3)`` or ``(molecules, atoms, 3)``.
        pbc: boolean vector of size 3 storing if pbc is enabled for that direction.

    Returns:
        coordinates of atoms mapped back to unit lattice.
    """
    # Step 1: convert xyz from standard cartesian coordinate to unit cell xyz
    try:
        inv_lattice = torch.linalg.inv(lattice)
    except:
        inv_lattice = torch.linalg.inv(lattice.cpu())
        inv_lattice = inv_lattice.to(xyz.device)
    xyz_lattice = xyz @ inv_lattice
    # Step 2: wrap lattice coordinates into [0, 1)
    xyz_lattice -= xyz_lattice.floor() * pbc.to(xyz_lattice.dtype)
    # Step 3: convert from lattice xyz back to standard cartesian coordinate
    return xyz_lattice @ lattice


def lattice_points_in_supercell(supercell_matrix):
    """Adapted from ASE to find all lattice points contained in a supercell.

    Adapted from pymatgen, which is available under MIT license:
    The MIT License (MIT) Copyright (c) 2011-2012 MIT & The Regents of the
    University of California, through Lawrence Berkeley National Laboratory
    """

    diagonals = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
            [0, -1, 1],
            [1, 0, -1],
            [1, -1, 0],
            [1, -1, 1],
            [-1, -1, 1],
            [1, 1, -1],
            [1, -1, -1],
        ]
    )

    d_points = np.dot(diagonals, supercell_matrix)

    mins = np.min(d_points, axis=0)
    maxes = np.max(d_points, axis=0) + 1

    ar = np.arange(mins[0], maxes[0])[:, None] * np.array([1, 0, 0])[None, :]
    br = np.arange(mins[1], maxes[1])[:, None] * np.array([0, 1, 0])[None, :]
    cr = np.arange(mins[2], maxes[2])[:, None] * np.array([0, 0, 1])[None, :]

    all_points = ar[:, None, None] + br[None, :, None] + cr[None, None, :]
    all_points = all_points.reshape((-1, 3))

    frac_points = np.dot(all_points, np.linalg.inv(supercell_matrix))

    tvects = frac_points[np.all(abs(frac_points) < 1 - 1e-10, axis=1)]

    return tvects


def clean_matrix(matrix, eps=1e-12):
    """ clean from small values"""
    matrix = np.array(matrix)
    for ij in np.ndindex(matrix.shape):
        if abs(matrix[ij]) < eps:
            matrix[ij] = 0
    return matrix


def get_nonperiodic_neighbor_list(xyz, cutoff=5, directed=False):
    """Get neighbor list from xyz positions of atoms.

    Args:
        xyz (torch.Tensor or np.array): (N, 3) array with positions
            of the atoms.
        cutoff (float): maximum distance to consider atoms as
            connected.

    Returns:
        nbr_list (torch.Tensor): (num_edges, 2) array with the
            indices of connected atoms.
    """

    if torch.is_tensor(xyz) is False:
        xyz = torch.Tensor(xyz)
    n = xyz.size(0)

    # calculating distances
    dist = (
        (xyz.expand(n, n, 3) - xyz.expand(n, n, 3).transpose(0, 1))
        .pow(2)
        .sum(dim=2)
        .sqrt()
    )

    # neighbor list
    mask = dist <= cutoff
    mask[np.diag_indices(n)] = 0
    nbr_list = mask.nonzero(as_tuple=False)

    if not directed:
        nbr_list = nbr_list[nbr_list[:, 1] > nbr_list[:, 0]]

    return nbr_list


def get_torch_nbr_list(
    atomsobject,
    cutoff,
    device="cuda:1",
    directed=True,
    requires_large_offsets=True,
):
    """Pytorch implementations of nbr_list for minimum image convention, the offsets are only limited to 0, 1, -1:
    it means that no pair interactions is allowed for more than 1 periodic box length. It is so much faster than
    neighbor_list algorithm in ase.
    It is similar to the output of neighbor_list("ijS", atomsobject, cutoff) but a lot faster
    Args:
        atomsobject (TYPE): ase.Atoms class
        cutoff (float): cutoff for neighbors list
        device (str, optional): cuda device
        requires_large_offsets: to get offsets beyond -1,0,1
    Returns:
        i, j, cutoff: just like ase.neighborlist.neighbor_list
    """

    if any(atomsobject.pbc):
        # check if sufficiently large to run the "fast" nbr_list function
        # also check if orthorhombic
        # otherwise, default to the "robust" nbr_list function below for small cells
        if (
            np.all(2 * cutoff < atomsobject.cell.cellpar()[:3])
            and not np.count_nonzero(
                atomsobject.cell.T - np.diag(np.diagonal(atomsobject.cell.T))
            )
            != 0
        ):
            # "fast" nbr_list function for large cells (pbc)
            xyz = torch.Tensor(atomsobject.get_positions(wrap=False)).to(device)
            dis_mat = xyz[None, :, :] - xyz[:, None, :]
            cell_dim = torch.Tensor(atomsobject.get_cell().tolist()).diag().to(device)
            if requires_large_offsets:
                shift = torch.round(torch.divide(dis_mat, cell_dim))
                offsets = -shift
            else:
                offsets = -dis_mat.ge(0.5 * cell_dim).to(torch.float) + dis_mat.lt(
                    -0.5 * cell_dim
                ).to(torch.float)

            dis_mat = dis_mat + offsets * cell_dim
            dis_sq = dis_mat.pow(2).sum(-1)
            mask = (dis_sq < cutoff ** 2) & (dis_sq != 0)
            nbr_list = mask.nonzero(as_tuple=False)
            offsets = (
                offsets[nbr_list[:, 0], nbr_list[:, 1], :].detach().to("cpu").numpy()
            )

        else:
            # "robust" nbr_list function for all cells (pbc)
            xyz = torch.Tensor(atomsobject.get_positions(wrap=True)).to(device)

            # since we are not wrapping
            # retrieve the shift vectors that would be equivalent to wrapping
            positions = atomsobject.get_positions(wrap=True)
            unwrapped_positions = atomsobject.get_positions(wrap=False)
            shift = positions - unwrapped_positions
            cell = atomsobject.cell
            cell = np.broadcast_to(
                cell.T, (shift.shape[0], cell.shape[0], cell.shape[1])
            )
            shift = np.linalg.solve(cell, shift).round().astype(int)

            # estimate getting close to the cutoff with supercell expansion
            cell = atomsobject.cell
            a_mul = int(np.ceil(cutoff / np.linalg.norm(cell[0]))) + 1
            b_mul = int(np.ceil(cutoff / np.linalg.norm(cell[1]))) + 1
            c_mul = int(np.ceil(cutoff / np.linalg.norm(cell[2]))) + 1
            supercell_matrix = np.array([[a_mul, 0, 0], [0, b_mul, 0], [0, 0, c_mul]])
            supercell = clean_matrix(supercell_matrix @ cell)

            # cartesian lattice points
            lattice_points_frac = lattice_points_in_supercell(supercell_matrix)
            lattice_points = np.dot(lattice_points_frac, supercell)
            # need to get all negative lattice translation vectors but remove duplicate 0 vector
            zero_idx = np.where(
                np.all(lattice_points.__eq__(np.array([0, 0, 0])), axis=1)
            )[0][0]
            lattice_points = np.concatenate(
                [lattice_points[zero_idx:, :], lattice_points[:zero_idx, :]]
            )

            N = len(lattice_points)
            # perform lattice translation vectors on positions
            lattice_points_T = torch.tile(
                torch.from_numpy(lattice_points),
                (len(xyz),) + (1,) * (len(lattice_points.shape) - 1),
            ).to(device)
            lattice_points = None
            lattice_points_frac = None
            xyz_T = torch.repeat_interleave(xyz.view(-1, 1, 3), N, dim=1)
            xyz_T = xyz_T + lattice_points_T.view(xyz_T.shape)
            diss = xyz_T[None, :, None, :, :] - xyz_T[:, None, :, None, :]
            diss = diss[:, :, 0, :, :]
            dis_sq = diss.pow(2).sum(-1)
            mask = (dis_sq < cutoff ** 2) & (dis_sq != 0)
            nbr_list = mask.nonzero(as_tuple=False)[:, :2]
            offsets = lattice_points_T.view(xyz_T.shape)[
                mask.nonzero(as_tuple=False)[:, 1], mask.nonzero(as_tuple=False)[:, 2]
            ]
            xyz_T = None
            lattice_points_T = None


            # get offsets as original integer multiples of lattice vectors
            cell = np.broadcast_to(
                cell.T, (offsets.shape[0], cell.shape[0], cell.shape[1])
            )
            offsets = offsets.detach().to("cpu").numpy()
            offsets = np.linalg.solve(cell, offsets).round().astype(int)

            # add shift to offsets with the right indices according to pairwise nbr_list
            offsets = torch.from_numpy(offsets).int().to(device)
            shift = torch.from_numpy(shift).int().to(device)

            # index shifts by atom but then apply shifts to pairwise interactions
            # get shifts for each atom i and j that would be equivalent to wrapping
            # convention is j - i for get_rij with NNs
            shift_i = shift[nbr_list[:, 0]]
            shift_j = shift[nbr_list[:, 1]]
            offsets = (shift_j - shift_i + offsets).detach().to("cpu").numpy()

    else:
        xyz = torch.Tensor(atomsobject.get_positions(wrap=False)).to(device)
        nbr_list = get_nonperiodic_neighbor_list(xyz=xyz, cutoff=cutoff, directed=directed)

    if not directed:
        nbr_list = nbr_list[nbr_list[:, 1] > nbr_list[:, 0]]

    i, j = (
        nbr_list[:, 0].detach().to("cpu").numpy(),
        nbr_list[:, 1].detach().to("cpu").numpy(),
    )

    if any(atomsobject.pbc):
        offsets = offsets
    else:
        offsets = np.zeros((nbr_list.shape[0], 3))

    return i, j, offsets


class AtomsBatch(Atoms):
    """Class to deal with the Neural Force Field and batch several
       Atoms objects.
    """

    def __init__(
        self,
        *args,
        props=None,
        cutoff=DEFAULT_CUTOFF,
        directed=DEFAULT_DIRECTED,
        requires_large_offsets=False,
        use_pymatgen_nbrlist=False,
        cutoff_skin=DEFAULT_SKIN,
        device="cuda:1",
        **kwargs,
    ):
        """

        Args:
            *args: Description
            nbr_list (None, optional): Description
            pbc_index (None, optional): Description
            cutoff (TYPE, optional): Description
            cutoff_skin (float): extra distance added to cutoff
                            to ensure we don't miss neighbors between nbr
                            list updates.
            **kwargs: Description
        """
        super().__init__(*args, **kwargs)

        if props is None:
            props = {}

        self.props = props
        self.nbr_list = props.get("nbr_list", None)
        self.offsets = props.get("offsets", None)
        self.directed = directed
        self.num_atoms = props.get("num_atoms", torch.LongTensor([len(self)])).reshape(
            -1
        )
        self.props["num_atoms"] = self.num_atoms
        self.cutoff = cutoff
        self.cutoff_skin = cutoff_skin
        self.device = device
        self.requires_large_offsets = requires_large_offsets
        self.use_pymatgen_nbrlist = use_pymatgen_nbrlist
        self.mol_nbrs, self.mol_idx = self.get_mol_nbrs()

    def get_mol_nbrs(self):
        """
        Dense directed neighbor list for each molecule, in case that's needed
        in the model calculation
        """

        # Not yet implemented for PBC
        if self.offsets is not None and (self.offsets != 0).any():
            return None, None

        counter = 0
        nbrs = []

        for atoms in self.get_list_atoms():
            nxyz = np.concatenate(
                [
                    atoms.get_atomic_numbers().reshape(-1, 1),
                    atoms.get_positions().reshape(-1, 3),
                ],
                axis=1,
            )

            n = nxyz.shape[0]
            idx = torch.arange(n)
            x, y = torch.meshgrid(idx, idx, indexing="xy")

            # undirected neighbor list
            these_nbrs = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1)], dim=1)
            these_nbrs = these_nbrs[these_nbrs[:, 0] != these_nbrs[:, 1]]

            nbrs.append(these_nbrs + counter)
            counter += n

        nbrs = torch.cat(nbrs)
        mol_idx = torch.cat(
            [torch.zeros(num) + i for i, num in enumerate(self.num_atoms)]
        ).long()

        return nbrs, mol_idx

    def get_nxyz(self):
        """Gets the atomic number and the positions of the atoms
           inside the unit cell of the system.
        Returns:
            nxyz (np.array): atomic numbers + cartesian coordinates
                             of the atoms.
        """
        nxyz = np.concatenate(
            [
                self.get_atomic_numbers().reshape(-1, 1),
                self.get_positions().reshape(-1, 3),
            ],
            axis=1,
        )

        return nxyz

    def get_batch(self):
        """Uses the properties of Atoms to create a batch
           to be sent to the model.
           Returns:
              batch (dict): batch with the keys 'nxyz',
                            'num_atoms', 'nbr_list' and 'offsets'
        """

        if self.nbr_list is None or self.offsets is None:
            self.update_nbr_list()

        self.props["nbr_list"] = self.nbr_list
        self.props["offsets"] = self.offsets
        if self.pbc.any():
            self.props["cell"] = self.cell

        self.props["nxyz"] = torch.Tensor(self.get_nxyz())
        if self.props.get("num_atoms") is None:
            self.props["num_atoms"] = torch.LongTensor([len(self)])

        if self.mol_nbrs is not None:
            self.props["mol_nbrs"] = self.mol_nbrs

        if self.mol_idx is not None:
            self.props["mol_idx"] = self.mol_idx

        return self.props

    def get_list_atoms(self):

        if self.props.get("num_atoms") is None:
            self.props["num_atoms"] = torch.LongTensor([len(self)])

        mol_split_idx = self.props["num_atoms"].tolist()

        positions = torch.Tensor(self.get_positions())
        Z = torch.LongTensor(self.get_atomic_numbers())

        positions = list(positions.split(mol_split_idx))
        Z = list(Z.split(mol_split_idx))
        masses = list(torch.Tensor(self.get_masses()).split(mol_split_idx))

        Atoms_list = []

        for i, molecule_xyz in enumerate(positions):
            atoms = Atoms(
                Z[i].tolist(), molecule_xyz.numpy(), cell=self.cell, pbc=self.pbc
            )

            # in case you artificially changed the masses
            # of any of the atoms
            atoms.set_masses(masses[i])

            Atoms_list.append(atoms)

        return Atoms_list

    def update_nbr_list(self):
        """Update neighbor list and the periodic reindexing
           for the given Atoms object.
           Args:
           cutoff(float): maximum cutoff for which atoms are
                                          considered interacting.
           Returns:
           nbr_list(torch.LongTensor)
           offsets(torch.Tensor)
           nxyz(torch.Tensor)
        """

        Atoms_list = self.get_list_atoms()

        ensemble_nbr_list = []
        ensemble_offsets_list = []

        for i, atoms in enumerate(Atoms_list):
            if not self.use_pymatgen_nbrlist:
                edge_from, edge_to, offsets = get_torch_nbr_list(
                    atoms,
                    (self.cutoff + self.cutoff_skin),
                    device=self.device,
                    directed=self.directed,
                    requires_large_offsets=self.requires_large_offsets,
                )
            else:
                struc = Structure(
                    lattice=atoms.get_cell(),
                    coords=atoms.get_positions(),
                    species=atoms.get_atomic_numbers(),
                    coords_are_cartesian=True,
                )
                edge_from, edge_to, offsets, _ = struc.get_neighbor_list(
                    r=self.cutoff + self.cutoff_skin,
                )
                if not self.directed:
                    inds = edge_to > edge_from
                    edge_from = edge_from[inds]
                    edge_to = edge_to[inds]
                    offsets = offsets[inds]

            nbr_list = torch.LongTensor(np.stack([edge_from, edge_to], axis=1))
            these_offsets = sparsify_array(offsets.dot(self.get_cell()))

            # non-periodic
            if isinstance(these_offsets, int):
                these_offsets = torch.Tensor(offsets)

            ensemble_nbr_list.append(self.props["num_atoms"][:i].sum() + nbr_list)
            ensemble_offsets_list.append(these_offsets)

        ensemble_nbr_list = torch.cat(ensemble_nbr_list)

        if all([isinstance(i, int) for i in ensemble_offsets_list]):
            ensemble_offsets_list = torch.Tensor(ensemble_offsets_list)
        else:
            ensemble_offsets_list = torch.cat(ensemble_offsets_list)

        self.nbr_list = ensemble_nbr_list
        self.offsets = ensemble_offsets_list

        return ensemble_nbr_list, ensemble_offsets_list

    def get_batch_energies(self):

        if self._calc is None:
            raise RuntimeError("Atoms object has no calculator.")

        if not hasattr(self._calc, "get_potential_energies"):
            raise RuntimeError(
                "The calculator for atomwise energies is not implemented"
            )

        energies = self.get_potential_energies()

        batched_energies = split_and_sum(
            torch.Tensor(energies), self.props["num_atoms"].tolist()
        )

        return batched_energies.detach().cpu().numpy()

    def get_batch_kinetic_energy(self):

        if self.get_momenta().any():
            atomwise_ke = torch.Tensor(
                0.5 * self.get_momenta() * self.get_velocities()
            ).sum(-1)
            batch_ke = split_and_sum(atomwise_ke, self.props["num_atoms"].tolist())
            return batch_ke.detach().cpu().numpy()

        else:
            print("No momenta are set for atoms")

    def get_batch_T(self):

        T = self.get_batch_kinetic_energy() / (
            1.5 * units.kB * self.props["num_atoms"].detach().cpu().numpy()
        )
        return T

    def batch_properties():
        pass

    def batch_virial():
        pass

    @classmethod
    def from_atoms(cls, atoms, **kwargs):
        return cls(
            atoms, positions=atoms.positions, numbers=atoms.numbers, props={}, **kwargs
        )
