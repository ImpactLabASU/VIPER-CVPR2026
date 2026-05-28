#!/usr/bin/env python3
"""
Generate u(x,t) simulations for all PDEs and seeds.
Saves to data/simulations/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml
from simp_video.utils.pde_solvers import solve_pde


def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "experiment_configs.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Generate all PDEs, all seeds")
    parser.add_argument("--pde", type=str, choices=["kdv", "burgers", "ks", "schrodinger", "nls"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    out_dir = Path(__file__).parent.parent / "data" / "simulations"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdes = config["pde_configs"]
    seeds = config["seeds"]

    if args.pde:
        pde_names = [args.pde]
    else:
        pde_names = list(pdes.keys())

    for pde_name in pde_names:
        cfg = pdes[pde_name]
        L = cfg["domain"]["L"]
        nx = cfg["domain"]["nx"]
        T = cfg["time"]["T"]
        dt = cfg["time"]["dt"]
        true_coeffs = cfg["true_coeffs"]

        x = np.linspace(-L / 2, L / 2, nx)
        t = np.arange(0, T + dt / 2, dt)

        run_seeds = [args.seed] if args.seed is not None else seeds
        for seed in run_seeds:
            u = solve_pde(pde_name, x, t, true_coeffs, seed=seed)
            fname = out_dir / f"{pde_name}_seed{seed}.npz"
            np.savez(fname, u=u, x=x, t=t, true_coeffs=true_coeffs, pde=pde_name)
            print(f"Saved {fname}")

    print("Done.")


if __name__ == "__main__":
    main()
