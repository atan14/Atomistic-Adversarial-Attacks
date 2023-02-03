import torch
import numbers
import numpy as np
import copy
from copy import deepcopy
from sklearn.utils import shuffle as skshuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset

from .utils import get_nonperiodic_neighbor_list, AtomsBatch


class Dataset(TorchDataset):
    """Dataset to deal with NFF calculations.

    Attributes:
        props (dict of lists): dictionary, where each key is the name of a property and
            each value is a list. The element of each list is the properties of a single
            geometry, whose coordinates are given by
            `nxyz`.

            Keys are the name of the property and values are the properties. Each value
            is given by `props[idx][key]`. The only mandatory key is 'nxyz'. If inputting
            energies, forces or hessians of different electronic states, the quantities
            should be distinguished with a "_n" suffix, where n = 0, 1, 2, ...
            Whatever name is given to the energy of state n, the corresponding force name
            must be the exact same name, but with "energy" replaced by "force".

            Example:

                props = {
                    'nxyz': [np.array([[1, 0, 0, 0], [1, 1.1, 0, 0]]),
                             np.array([[1, 3, 0, 0], [1, 1.1, 5, 0]])],
                    'energy_0': [1, 1.2],
                    'energy_0_grad': [np.array([[0, 0, 0], [0.1, 0.2, 0.3]]),
                                      np.array([[0, 0, 0], [0.1, 0.2, 0.3]])],
                    'energy_1': [1.5, 1.5],
                    'energy_1_grad': [np.array([[0, 0, 1], [0.1, 0.5, 0.8]]),
                                      np.array([[0, 0, 1], [0.1, 0.5, 0.8]])],
                    'dipole_2': [3, None]
                }

            Periodic boundary conditions must be specified through the 'offset' key in
                props. Once the neighborlist is created, distances between
                atoms are computed by subtracting their xyz coordinates
                and adding to the offset vector. This ensures images
                of atoms outside of the unit cell have different
                distances when compared to atoms inside of the unit cell.
                This also bypasses the need for a reindexing.

        units (str): units of the energies, forces etc.

    """

    def __init__(self, props, units="kcal/mol", check_props=True, do_copy=True):
        """Constructor for Dataset class.

        Args:
            props (dictionary of lists): dictionary containing the
                properties of the system. Each key has a list, and
                all lists have the same length.
            units (str): units of the system.
        """
        if check_props:
            if do_copy:
                self.props = self._check_dictionary(deepcopy(props))
            else:
                self.props = self._check_dictionary(props)
        else:
            self.props = props
        self.units = units

    def __len__(self):
        return len(self.props["nxyz"])

    def __getitem__(self, idx):
        if isinstance(idx, list) or isinstance(idx, tuple):
            return [{key: val[i] for key, val in self.props.items()} for i in idx]
        else:
            return {key: val[idx] for key, val in self.props.items()}

    def __add__(self, other):
        new_props = self.props
        keys = list(new_props.keys())
        for key in keys:
            if key not in other.props:
                new_props.pop(key)
                continue
            val = other.props[key]
            if type(val) is list:
                new_props[key] += val
            else:
                old_val = new_props[key]
                new_props[key] = torch.cat([old_val, val.to(old_val.dtype)])
        self.props = new_props

        return copy.deepcopy(self)

    def _check_dictionary(self, props):
        """Check the dictionary or properties to see if it has the
        specified format.
        """

        assert "nxyz" in props.keys()
        n_atoms = [len(x) for x in props["nxyz"]]
        n_geoms = len(props["nxyz"])

        if "num_atoms" not in props.keys():
            props["num_atoms"] = torch.LongTensor(n_atoms)
        else:
            props["num_atoms"] = torch.LongTensor(props["num_atoms"])

        for key, val in props.items():

            if val is None:
                props[key] = to_tensor([np.nan] * n_geoms)

            elif any([x is None for x in val]):
                bad_indices = [i for i, item in enumerate(val) if item is None]
                good_indices = [
                    index for index in range(len(val)) if index not in bad_indices
                ]
                if len(good_indices) == 0:
                    nan_list = np.array([float("NaN")]).tolist()
                else:
                    good_index = good_indices[0]
                    nan_list = (np.array(val[good_index]) * float("NaN")).tolist()
                for index in bad_indices:
                    props[key][index] = nan_list
                props.update({key: to_tensor(val)})

            else:
                assert len(val) == n_geoms, (
                    f"length of {key} is not "
                    f"compatible with {n_geoms} "
                    "geometries"
                )
                props[key] = to_tensor(val)

        return props

    def generate_neighbor_list(
        self, cutoff, directed=False, key="nbr_list", offset_key="offsets"
    ):
        """Generates a neighbor list for each one of the atoms in the dataset.
            By default, does not consider periodic boundary conditions.

        Args:
            cutoff (float): distance up to which atoms are considered bonded.
            directed (bool, optional): Description

        Returns:
            TYPE: Description
        """
        if "lattice" not in self.props:
            self.props[key] = [
                get_nonperiodic_neighbor_list(nxyz[:, 1:4], cutoff, directed)
                for nxyz in self.props["nxyz"]
            ]
            self.props[offset_key] = [
                torch.sparse.FloatTensor(nbrlist.shape[0], 3)
                for nbrlist in self.props[key]
            ]
        else:
            self._get_periodic_neighbor_list(
                cutoff=cutoff, directed=directed, offset_key=offset_key, nbr_key=key
            )
            return self.props[key], self.props[offset_key]

        return self.props[key]

    def _get_periodic_neighbor_list(
        self, cutoff, directed=True, offset_key="offsets", nbr_key="nbr_list"
    ):
        nbrlist = []
        offsets = []
        for nxyz, lattice in zip(self.props["nxyz"], self.props["lattice"]):
            atoms = AtomsBatch(
                nxyz[:, 0].long(),
                props={"num_atoms": torch.LongTensor([len(nxyz[:, 0])])},
                positions=nxyz[:, 1:],
                cell=lattice,
                pbc=True,
                cutoff=cutoff,
                directed=directed,
            )
            nbrs, offs = atoms.update_nbr_list()
            nbrlist.append(nbrs)
            offsets.append(offs)

        self.props[nbr_key] = nbrlist
        self.props[offset_key] = offsets
        return

    def copy(self):
        """Copies the current dataset

        Returns:
            TYPE: Description
        """
        return Dataset(self.props, self.units)

    def change_idx(self, idx):
        """
        Change the dataset so that the properties are ordered by the
        indices `idx`. If `idx` does not contain all of the original
        indices in the dataset, then this will reduce the size of the
        dataset.
        """

        for key, val in self.props.items():
            if isinstance(val, list):
                self.props[key] = [val[i] for i in idx]
            else:
                self.props[key] = val[idx]

    def shuffle(self):
        """Summary

        Returns:
            TYPE: Description
        """
        idx = list(range(len(self)))
        reindex = skshuffle(idx)
        self.change_idx(reindex)

    def save(self, path):
        """Summary

        Args:
            path (TYPE): Description
        """

        # to deal with the fact that sparse tensors can't be pickled
        offsets = self.props.get("offsets", torch.LongTensor([0]))
        old_offsets = copy.deepcopy(offsets)

        # check if it's a sparse tensor. The first two conditions
        # Are needed for backwards compatability in case it's a float
        # or empty list

        if all([hasattr(offsets, "__len__"), len(offsets) > 0]):
            if isinstance(offsets[0], torch.sparse.FloatTensor):
                self.props["offsets"] = [val.to_dense() for val in offsets]

        torch.save(self, path)
        if "offsets" in self.props:
            self.props["offsets"] = old_offsets

    @classmethod
    def from_file(cls, path):
        """Summary

        Args:
            path (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            TypeError: Description
        """
        obj = torch.load(path)
        if isinstance(obj, cls):
            return obj
        else:
            raise TypeError("{} is not an instance from {}".format(path, type(cls)))


def force_to_energy_grad(dataset):
    """
    Converts forces to energy gradients in a dataset. This conforms to
        the notation that a key with `_grad` is the gradient of the
        property preceding it. Modifies the database in-place.

    Args:
        dataset (TYPE): Description

    Returns:
        success (bool): if True, forces were removed and energy_grad
            became the new key.
    """
    if "forces" not in dataset.props.keys():
        return False
    else:
        dataset.props["energy_grad"] = [-x for x in dataset.props.pop("forces")]
        return True


def convert_nan(x):
    """
    If a list has any elements that contain nan, convert its contents
    to the right form so that it can eventually be converted to a tensor.
    Args:
        x (list): any list with floats, ints, or Tensors.
    Returns:
        new_x (list): updated version of `x`
    """

    new_x = []
    # whether any of the contents have nan
    has_nan = any([np.isnan(y).any() for y in x])
    for y in x:

        if has_nan:
            # if one is nan then they will have to become float tensors
            if type(y) in [int, float]:
                new_x.append(torch.Tensor([y]))
            elif isinstance(y, torch.Tensor):
                new_x.append(y.float())
            elif isinstance(y, list):
                new_x.append(torch.Tensor(y))
            else:
                msg = (
                    "Don't know how to convert sub-components of type "
                    f"{type(x)} when components might contain nan"
                )
                raise Exception(msg)
        else:
            # otherwise they can be kept as is
            new_x.append(y)

    return new_x


def to_tensor(x, stack=False):
    """
    Converts input `x` to torch.Tensor.
    Args:
        x (list of lists): input to be converted. Can be: number, string, list, array, tensor
        stack (bool): if True, concatenates torch.Tensors in the batching dimension
    Returns:
        torch.Tensor or list, depending on the type of x
    Raises:
        TypeError: Description
    """

    # a single number should be a list
    if isinstance(x, numbers.Number):
        return torch.Tensor([x])

    if isinstance(x, str):
        return [x]

    if isinstance(x, torch.Tensor):
        return x

    if type(x) is list and type(x[0]) != str:
        if not isinstance(x[0], torch.sparse.FloatTensor):
            x = convert_nan(x)

    # all objects in x are tensors
    if isinstance(x, list) and all([isinstance(y, torch.Tensor) for y in x]):

        # list of tensors with zero or one effective dimension
        # flatten the tensor

        if all([len(y.shape) < 1 for y in x]):
            return torch.cat([y.view(-1) for y in x], dim=0)

        elif stack:
            return torch.cat(x, dim=0)

        # list of multidimensional tensors
        else:
            return x

    # some objects are not tensors
    elif isinstance(x, list):

        # list of strings
        if all([isinstance(y, str) for y in x]):
            return x

        # list of ints
        if all([isinstance(y, int) for y in x]):
            return torch.LongTensor(x)

        # list of floats
        if all([isinstance(y, numbers.Number) for y in x]):
            return torch.Tensor(x)

        # list of arrays or other formats
        if any([isinstance(y, (list, np.ndarray)) for y in x]):
            return [torch.Tensor(y) for y in x]

    raise TypeError("Data type not understood")


def concatenate_dict(*dicts):
    """Concatenates dictionaries as long as they have the same keys.
        If one dictionary has one key that the others do not have,
        the dictionaries lacking the key will have that key replaced by None.
    Args:
        *dicts: Description
        *dicts (any number of dictionaries)
            Example:
                dict_1 = {
                    'nxyz': [...],
                    'energy': [...]
                }
                dict_2 = {
                    'nxyz': [...],
                    'energy': [...]
                }
                dicts = [dict_1, dict_2]
    Returns:
        TYPE: Description
    """

    assert all(
        [type(d) == dict for d in dicts]
    ), "all arguments have to be dictionaries"

    # Old method
    # keys = set(sum([list(d.keys()) for d in dicts], []))

    # New method
    keys = set()
    for dic in dicts:
        for key in dic.keys():
            if key not in keys:
                keys.add(key)

    # While less pretty, the new method is MUCH faster. For example,
    # for a dataset of size 600,000, the old method literally
    # takes hours, while the new method takes 250 ms

    def is_list_of_lists(value):
        if isinstance(value, list):
            return isinstance(value[0], list)
        return False

    def get_length(value):
        if is_list_of_lists(value):
            if is_list_of_lists(value[0]):
                return len(value)
            return 1

        elif isinstance(value, list):
            return len(value)

        return 1

    def get_length_of_values(dict_):
        if "nxyz" in dict_:
            return get_length(dict_["nxyz"])
        return min([get_length(v) for v in dict_.values()])

    def flatten_val(value):
        """Given a value, which can be a number, a list or
            a torch.Tensor, return its flattened version
            to be appended to a list of values
        """
        if is_list_of_lists(value):
            if is_list_of_lists(value[0]):
                return value
            else:
                return [value]

        elif isinstance(value, list):
            return value

        elif isinstance(value, torch.Tensor):
            if len(value.shape) == 0:
                return [value]
            elif len(value.shape) == 1:
                return [item for item in value]
            else:
                return [value]

        elif get_length(value) == 1:
            return [value]

        return [value]

    # we have to see how many values the properties of each dictionary has.
    values_per_dict = [get_length_of_values(d) for d in dicts]

    # creating the joint dicionary
    joint_dict = {}
    for key in keys:
        # flatten list of values
        values = []
        for num_values, d in zip(values_per_dict, dicts):
            val = d.get(key, ([None] * num_values if num_values > 1 else None))
            values += flatten_val(val)
        joint_dict[key] = values

    return joint_dict


def split_train_test(
    dataset, test_size=0.2, random_state=0, shuffle=True,
):
    idx = list(range(len(dataset)))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=random_state, shuffle=shuffle
    )

    train = Dataset(
        props={key: [val[i] for i in idx_train] for key, val in dataset.props.items()},
        units=dataset.units,
    )
    test = Dataset(
        props={key: [val[i] for i in idx_test] for key, val in dataset.props.items()},
        units=dataset.units,
    )
    return (train, test, (idx_train, idx_test))


def split_train_validation_test(
    dataset,
    val_size=0.2,
    test_size=0.2,
    random_state=0,
    return_indices=False,
    shuffle_train_test=True,
    **kwargs,
):
    """
    Args:
        random_state (int): seed for shuffling
        return_indices (bool): whether indices of all splits are returned
        shuffle_train_test (bool): whether data are shuffle for (train&val) vs
            test splits, since sometimes you want the test data to be very
            different to assess models extrapolation power. Note that data in
            train vs val splits is always shuffled.
    """
    train, test, (train_idx, test_idx) = split_train_test(
        dataset,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle_train_test,
    )
    train, validation, (train_idx_, val_idx_) = split_train_test(
        train,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        shuffle=True,
    )
    if return_indices:
        train_idx, val_idx = [train_idx[i] for i in train_idx_], [
            train_idx[i] for i in val_idx_
        ]
        return train, validation, test, (train_idx, val_idx, test_idx)

    return train, validation, test
