import os
import argparse
from evi.md.traj import MDTraj

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("md_path", type=str)
    parser.add_argument("--out_path", type=str, default=f"{os.getenv('POOL')}/projects/evidential/systems/ammonia/md/nvt/explosion.txt")
    parser.add_argument("--min_r",  type=float, default=0.7)
    parser.add_argument("--max_r", type=float, default=2.25)
    args = parser.parse_args()

    if args.md_path.endswith(".traj"):
        args.md_path = args.md_path.replace(".traj", "")
    if args.md_path.endswith(".log"):
        args.md_path = args.md_path.replace(".log", "")
    if args.md_path.endswith(".json"):
        args.md_path = args.md_path.replace(".json", "")

    traj = MDTraj(args.md_path)
    explosion = traj.get_explosion_time(min_r=args.min_r, max_r=args.max_r)
    with open(args.out_path, 'a+') as f:
        f.write(f"{args.md_path},{explosion}\n")
        f.close()
