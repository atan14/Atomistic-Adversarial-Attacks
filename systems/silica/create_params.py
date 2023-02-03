import os
import json
import uuid
import argparse
import warnings
from glob import glob
import numpy as np
from evi.utils.output import make_dir, write_params


DATA_DIR = f"{os.getenv('STORAGE')}/data"
MODEL_DIR = f"{os.getenv('STORAGE')}/models"
PARAMS_DIR = f"{os.getenv('STORAGE')}/params/inbox"

DEFAULT_PARAMS = {
    "generation": None,
    "model": {
        "n_atom_basis": 128,
        "n_filters": 128,
        "n_gaussians": 20,
        "n_convolutions": 3,
        "cutoff": 5.0,
        "trainable_gauss": False,
        "dropout_rate": 0.0,
        "activation": "shifted_softplus",
        "num_readout_layer": {
            "energy": 1,
        },
        "pool_dic": {
            "energy": {
                "name": "sum",
                "param": {},
            }
        },
        "num_networks": 1,
        "uncertainty_type": "ensemble",
        "model_type": "schnet",
        "output_keys": ['energy'],
        "grad_keys": ['energy_grad'],
        "outdir": "",
    },
    "train": {
        "lr": 1e-4,
        "device": "cuda:2",
        "n_epochs": 1000,
        "every_n_epochs": 20,
        "checkpoint_interval": 10,
        "lr_factor": 0.5,
        "min_lr": 1e-07,
    },
    "loss": {
        "lambda": 0.5,
        "epsilon": 0.0,
        "energy_loss": "mae",
        "forces_loss": "mae",
        "energy_coef": 0.1,
        "forces_coef": 1.0,
        "clamp_min": 1e-5,
    },
    "dset": {
        "test_size": 0.2,
        "val_size": 0.2,
        "shuffle_train_test": True,
        "batch_size": 8,
        "random_state": 0,
        "path": None,
    },
    "sampling": {
        "adv_fn": "",
        "num_attacks": 100,
        "lr": 1e-4,
        "n_epochs": 80,
        "kT": 0.7,  # kcal/mol
        "uncertainty_source": "epistemic",
        "n_clusters": 2,
    },
    "dft": {
        "group_name": "uncertainty",
        "parentconfig": "nn_evidential_attack_crystal",
        "method": "nn_evidential_attack",
        "method_description": "from addxyz",
        "childconfig": "pbe_paw_engrad_vasp",
        "childconfig_method": "dft_paw_gga_pbe",
        "childconfig_method_description": "DFT PBE with PAW",
    },
}


def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_generations", type=int, default=3)
    for key, parameters in DEFAULT_PARAMS.items():
        if not isinstance(parameters, dict):
            continue
        for param, val in parameters.items():
            if isinstance(val, dict) or isinstance(val, list):
                continue
            parser.add_argument(
                f"--{key}__{param}", type=type(val), default=val,
            )
    parser.add_argument("-f", "--force", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    args = parser.parse_args()
    return args


def check_input_args(args):
    if args.total_generations is None:
        raise Exception("No generation is given")

    if args.model__uncertainty_type == "ensemble":
        if args.loss__energy_loss != "mae" and args.loss__forces_loss != "mae":
            raise Exception(
                "Ensemble model does not have MAE for energy or forces loss"
            )
        if args.model__num_networks < 2:
            args.model__num_networks = 4
            warnings.warn("Ensemble model only has one network, changed to 4")

        args.model__output_keys = ['energy']
        args.model__grad_keys = ['energy_grad']

    if args.model__uncertainty_type == "mve":
        if args.loss__energy_loss != "nll" and args.loss__forces_loss != "nll":
            raise Exception("MVE model does not have nll loss function used")
        if args.model__num_networks > 1:
            args.model__num_networks = 1
            warnings.warn("MVE model has more than 1 network, changed to 1")
        args.model__output_keys = ['energy', 'var']
        args.model__grad_keys = ['energy_grad']
        args.model__num_readout_layer = {
            "energy": 1,
            "var": 2,
        }

    if args.model__uncertainty_type == "evidential":
        if (
            args.loss__energy_loss != "evidential"
            and args.loss__forces_loss != "evidential"
        ):
            raise Exception(
                "Evidential model does not have evidential loss function used"
            )
        if args.model__num_networks > 1:
            args.model__num_networks = 1
            warnings.warn("Evidential model has more than 1 network, changed to 1")

        args.sampling__n_epochs = 60
        args.model__output_keys = ['energy', 'v', 'alpha', 'beta']
        args.model__grad_keys = ['energy_grad']
        args.model__num_readout_layer = {
            "energy": 1,
            "v": 2,
            "alpha": 2,
            "beta": 2,
        }

    if args.model__uncertainty_type == "gmm":
        if (args.loss__energy_loss != "mae" and args.loss__forces_loss != "mae"):
            raise Exception("GMM uncertainty model does not use MAE loss function specified")
        if args.model__num_networks > 1:
            args.model__num_networks = 1
            warnings.warn("GMM has more than 1 network, changed to 1")

        args.model__output_keys = ['energy', "embedding"]
        args.model__grad_keys = ['energy_grad']

    if len(args.sampling__adv_fn) == 0:
        args.sampling__adv_fn = args.model__uncertainty_type

    args.model__pool_dic = {
        "energy": {
            "name": "sum",
            "param": {},
        }
    }

    return args


def check_model_params(args):
    if args.model__model_type == 'schnet':
        param_names = ['n_atom_basis', 'n_filters', 'n_gaussians', 'n_convolutions', 'cutoff', 'output_keys', 'grad_keys']
    if args.model__model_type == 'painn':
        param_names = ['n_atom_basis', 'activation', 'n_gaussians', 'n_convolutions', 'cutoff', 'output_keys', 'grad_keys']
    for p in param_names:
        assert f"model__{p}" in args, f"Parameter {p} not given for {args.model__model_type} model"

    return args


def get_unique_id(args):
    args.id = str(uuid.uuid4())[:8]
    return args


def get_random_device(args):
    """
    Just to prevent hogging too much space in one GPU
    """
    device_id = np.random.choice(4) # not use cuda:0 since it's always full
    device = f"cuda:{device_id}"
    args.train__device = device
    return args


def get_dset_path(params):
    if params["dset"]["path"] is None:
        if params["generation"] > 0:
            params['dset']['path'] = f"{DATA_DIR}/{params['model']['uncertainty_type']}/e{params['loss']['energy_loss'][:3]}_f{params['loss']['forces_loss'][:3]}/{params['id']}_gen{params['generation']}.pth.tar"
        else:
            params['dset']['path'] = f"{DATA_DIR}/gen0.pth.tar"

    return params


def get_model_dir(params):
    if len(params["model"]["outdir"]) == 0 or params["model"]["outdir"] is None:
        params['model']['outdir'] = f"{MODEL_DIR}/{params['model']['uncertainty_type']}/e{params['loss']['energy_loss'][:3]}_f{params['loss']['forces_loss'][:3]}/{params['id']}_{params['generation']}"
    return params


def parse_args(args):
    parameters = {}
    args = vars(args)
    for gen in range(args['total_generations']):
        params = {
            "generation": gen,
            "model": {},
            "train": {},
            "loss": {},
            "dset": {},
            "dft": {},
            "sampling": {},
        }
        for key, val in args.items():
            if key in ["total_generations", "force", "dry_run"]:
                continue
            if key == 'id':
                params[key] = val
                continue
            k, p = key.split("__")
            params[k][p] = val
        parameters[gen] = params
    return parameters


def two_params_equal(params1, params2):
    for key in params1.keys():
        if key in ['generation', 'id']:
            continue
        for k in params1[key].keys():
            if (key == 'model') and (k == 'outdir'):
                continue
            if (key == 'train') and (k == 'device'):
                continue
            if (key == 'dset') and (k == 'path'):
                continue
            if params1[key][k] != params2[key][k]:
                return False
    return True


def check_duplicated_params(params, force=False):
    """
    function to check whether same parameters have been ran before
    """
    params_dir = PARAMS_DIR.replace("/inbox", "")
    all_paramsfile = glob(f"{params_dir}/*/*.json")
    unique_ids = [p[p.rfind("/")+1:p.rfind("_")] for p in all_paramsfile]
    _, unique_idx = np.unique(unique_ids, return_index=True)
    for i in unique_idx:
        existing_paramsfile = all_paramsfile[i]
        existing_params = json.load(open(existing_paramsfile, "r"))
        if two_params_equal(params, existing_params):
            if force:
                warnings.warn(f"This parameter {params['id']} is the same as existing parameter {existing_params['id']}")
            else:
                raise Exception(f"This parameter {params['id']} is the same as existing parameter {existing_params['id']}")


if __name__ == "__main__":
    args = add_args()
    args = check_input_args(args)
    args = check_model_params(args)
    args = get_random_device(args)
    args = get_unique_id(args)
    parameters = parse_args(args)
    check_duplicated_params(parameters[0], force=args.force)

    for gen, params in parameters.items():
        params = get_dset_path(params)
        params = get_model_dir(params)

        if not args.dry_run:
            params_path = make_dir(
                f"{PARAMS_DIR}/{params['model']['outdir'].split('/')[-1]}.json"
            )

            model_dir = make_dir(params["model"]["outdir"])
            write_params(params=params, jsonpath=params_path)
        else:
            params_path = f"{PARAMS_DIR}/{params['model']['outdir'].split('/')[-1]}.json"
            print(params_path)
