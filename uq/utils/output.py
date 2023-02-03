import os
import shutil
import json
import pickle5 as pickle


def make_dir(filepath):
    if not os.path.exists(os.path.dirname(filepath)):
        try:
            os.makedirs(os.path.dirname(filepath))
        except:
            pass
    return filepath


def get_outdir(model_dir, params=None):
    outdir = f"{model_dir}"
    if os.path.exists(outdir):
        newpath = f"{model_dir}_backup"
        if os.path.exists(newpath):
            shutil.rmtree(newpath)

        shutil.move(outdir, newpath)

    os.makedirs(outdir)

    if params:
        write_params(params=params, jsonpath=f"{outdir}/params.json")

    return outdir


def write_params(params, jsonpath):
    json.dump(
        params, open(f"{jsonpath}", "w"), indent=4, sort_keys=True,
    )


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(data, path):
    path = make_dir(path)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_json(data, path, **kwargs):
    path = make_dir(path)
    with open(path, "w") as f:
        json.dump(data, f, **kwargs)
