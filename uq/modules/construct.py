import collections
import torch
from torch.nn import ModuleDict, Sequential

from .activations import ShiftedSoftplus


layer_types = {
    "linear": torch.nn.Linear,
    "Tanh": torch.nn.Tanh,
    "ReLU": torch.nn.ReLU,
    "shifted_softplus": ShiftedSoftplus,
    "sigmoid": torch.nn.Sigmoid,
    "Dropout": torch.nn.Dropout,
    "LeakyReLU": torch.nn.LeakyReLU,
    "ELU": torch.nn.ELU,
    "swish": torch.nn.SiLU,
}


def get_default_readout(
    n_atom_basis,
    num_readout_layer={"energy": 2},
    output_keys=["energy"],
    activation="shifted_softplus",
):
    start_layer = {
        "name": "linear",
        "param": {"in_features": n_atom_basis, "out_features": n_atom_basis // 2},
    }
    mid_layer = {
        "name": "linear",
        "param": {"in_features": n_atom_basis // 2, "out_features": n_atom_basis // 2},
    }
    end_layer = {
        "name": "linear",
        "param": {"in_features": n_atom_basis // 2, "out_features": 1},
    }
    act_layer = {"name": activation, "param": {}}

    readoutdict = {}
    for key in output_keys:
        if num_readout_layer[key] == 1:
            readoutdict[key] = [start_layer, act_layer, end_layer]
        else:
            readoutdict[key] = (
                [start_layer, act_layer]
                + [mid_layer, act_layer] * (num_readout_layer[key] - 1)
                + [end_layer]
            )

    return readoutdict


def construct_sequential(layers):
    """Construct a sequential model from list of params

    Args:
        layers (list): list to describe the stacked layer params. Example:
            layers = [
                {'name': 'linear', 'param' : {'in_features': 10, 'out_features': 20}},
                {'name': 'linear', 'param' : {'in_features': 10, 'out_features': 1}}
            ]

    Returns:
        Sequential: Stacked Sequential Model
    """
    return Sequential(
        collections.OrderedDict(
            [layer["name"] + str(i), layer_types[layer["name"]](**layer["param"])]
            for i, layer in enumerate(layers)
        )
    )


def construct_module_dict(moduledict):
    """construct moduledict from a dictionary of layers

    Args:
        moduledict (dict): Description

    Returns:
        ModuleDict: Description
    """
    models = ModuleDict()
    for key in moduledict:
        models[key] = construct_sequential(moduledict[key])
    return models
