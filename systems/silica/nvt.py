import time
import argparse
import numpy as np
from evi.md.nvt import MdRunner


def get_args():
    parser = argparse.ArgumentParser(description="Run NFF MD several times")

    parser.add_argument(
        "-D",
        "--dataset",
        type=str,
        help="dataset path (default: %(default)s)",
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
        default=100,  # ps
        help="simulation time (in ps) (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda",
        type=str,
        help="device (default: %(default)s)",
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
        default=50,
        help="The save frequency of the trajectory",
    )
    parser.add_argument(
        "-s",
        "--timestep",
        type=float,
        default=0.25,  # fs
        help="timestep for running md (default: %(default)fs)",
    )
    parser.add_argument(
        "-n",
        "--nbr_update_freq",
        type=int,
        default=1,
        help="Frequency of updating neighbors list (default: %(default))",
    )
    parser.add_argument(
        "--continue_if_error",
        action='store_true',
        help="Whether to stop the simulation when the system exploded",
    )

    return parser.parse_args()


def get_idx(temperature):
    indices = {
        300: np.arange(497, 597).tolist(),
        500: np.arange(689, 789).tolist(),
        1000: np.arange(0, 100).tolist(),
        1500: np.arange(100, 200).tolist(),
        2000: np.arange(200, 300).tolist(),
        2500: np.arange(300, 399).tolist(),
        3000: np.arange(399, 497).tolist(),
        3500: np.arange(597, 689).tolist(),
    }
    if temperature not in indices.keys():
        # find closest temperature if temperature not in dft data
        temperature = min(indices.keys(), key=lambda x: abs(x - temperature))
    inds = indices[temperature]
    return inds[np.random.choice(len(inds))]


if __name__ == "__main__":
    args = get_args()

    md = MdRunner(
        dataset=args.dataset,
        idx=get_idx(args.temperature),
        model_dir=args.model_dir,
        outdir=args.outdir,
        temperature=args.temperature,  # K
        time=args.time,  # ps
        device=args.device,
        cutoff=args.cutoff,
        save_freq=args.save_freq,
        timestep=args.timestep,  # fs
        nbr_update_freq=args.nbr_update_freq,
        ttime=20,
        stop_if_error=not args.continue_if_error,
    )
    start_time = time.time()
    md.run()
    end_time = time.time()
    duration = end_time - start_time
    print(f"Duration: {duration}")
    print("Done.")
