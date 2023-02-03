import torch
from pint import UnitRegistry


def convert_to(values, orig_unit, target_unit):
    if orig_unit == target_unit:
        return values

    ureg = UnitRegistry()
    a = 1 * ureg(orig_unit)
    mag = a.to(target_unit).magnitude
    try:
        converted_val = values * float(mag)
    except TypeError:
        converted_val = [v * float(mag) for v in values]

    return converted_val


def convert_dataset_units(dset, target_dset_unit):
    """
    Args:
        dset (evi.data.Dataset): dataset to convert units for
        target_unit (str): desired target unit for dset. Choices are
            ['kcal/mol', 'eV', 'atomic']
    Return:
        dset (evi.data.Dataset)
    """
    xyz_unit = {
        "kcal/mol": "angstrom",
        "atomic": "bohr",
        "eV": "angstrom",
    }
    energy_unit = {
        "kcal/mol": "kcal / mol",
        "atomic": "Eh * N_A",
        "eV": "eV * N_A",
    }
    forces_unit = {
        "kcal/mol": "kcal / mol /angstrom",
        "atomic": "Eh * N_A / bohr",
        "eV": "eV * N_A / angstrom",
    }

    xyz = [nxyz[:, 1:] for nxyz in dset.props['nxyz']]
    xyz = convert_to(xyz, xyz_unit[dset.units], xyz_unit[target_dset_unit])
    dset.props['nxyz'] = [
        torch.cat([nxyz[:, 0].reshape(-1, 1), x], dim=1) for nxyz, x in zip(dset.props['nxyz'], xyz)
    ]

    dset.props['energy'] = convert_to(dset.props['energy'], energy_unit[dset.units], energy_unit[target_dset_unit])
    dset.props['energy_grad'] = convert_to(dset.props['energy_grad'], forces_unit[dset.units], forces_unit[target_dset_unit])

    dset.units = target_dset_unit

    return dset
