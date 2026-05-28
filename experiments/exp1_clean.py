#!/usr/bin/env python3
"""
Experiment 1: Clean Video Benchmarks
Recover PDE coefficients from noise-free synthetic videos.
5 seeds per PDE, 5 PDEs.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml
from simp_video.utils.pde_solvers import solve_pde
from simp_video.utils.metrics import coefficient_error
from simp_video.models.simp_v import train_simp_v
from simp_video.models.baselines.pde_find import pde_find_video
from simp_video.models.baselines.pinn_inverse import pinn_inverse_video
from simp_video.models.baselines.deepmod_video import deepmod_video


def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "experiment_configs.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_single(pde_name: str, seed: int, config: dict, methods: list) -> dict:
    cfg = config["pde_configs"][pde_name]
    L = cfg["domain"]["L"]
    nx = cfg["domain"]["nx"]
    T = cfg["time"]["T"]
    dt = cfg["time"]["dt"]
    true_coeffs = cfg["true_coeffs"]
    coeff_order = cfg.get("coeff_order", list(true_coeffs.keys()))

    x = np.linspace(-L / 2, L / 2, nx)
    t = np.arange(0, T + dt / 2, dt)
    u = solve_pde(pde_name, x, t, true_coeffs, seed=seed)

    results = {}
    theta_true = np.array([true_coeffs[k] for k in coeff_order])

    if "simp_v" in methods:
        r = train_simp_v(
            u, x, t, pde_name, true_coeffs, coeff_order,
            max_epochs=config["training_configs"]["max_epochs"],
            lr=config["training_configs"]["learning_rate"],
            log_every=config["training_configs"].get("log_every", 0),
            log_prefix=f"{pde_name} seed={seed}",
        )
        results["simp_v"] = float(r["coefficient_error_pct"])

    if "pde_find" in methods:
        r = pde_find_video(u, x, t, pde_name, true_coeffs, coeff_order)
        results["pde_find"] = float(r["coefficient_error_pct"])

    if "pinn" in methods:
        r = pinn_inverse_video(u, x, t, pde_name, true_coeffs, coeff_order)
        results["pinn"] = float(r["coefficient_error_pct"])

    if "deepmod" in methods:
        r = deepmod_video(u, x, t, pde_name, true_coeffs, coeff_order)
        results["deepmod"] = float(r["coefficient_error_pct"])

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--methods", type=str, default="simp_v,pde_find,pinn,deepmod")
    parser.add_argument("--log-every", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    if args.log_every is not None:
        config.setdefault("training_configs", {})
        config["training_configs"]["log_every"] = args.log_every
    seeds = config["seeds"][:5]
    pdes = list(config["pde_configs"].keys())
    methods = args.methods.split(",")

    if args.pde:
        pdes = [args.pde]
    if args.seed is not None:
        seeds = [args.seed]

    out_dir = Path(__file__).parent.parent / "results" / "exp1_clean"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {pde: {m: [] for m in methods} for pde in pdes}

    for pde in pdes:
        for seed in seeds:
            r = run_single(pde, seed, config, methods)
            for m, err in r.items():
                all_results[pde][m].append(err)
            print(f"{pde} seed={seed}: {r}")

    summary = {}
    for pde in pdes:
        summary[pde] = {}
        for m in methods:
            vals = all_results[pde][m]
            summary[pde][m] = f"{np.mean(vals):.2f} ± {np.std(vals):.2f}" if vals else "N/A"

    with open(out_dir / "results.json", "w") as f:
        json.dump({"raw": all_results, "summary": summary}, f, indent=2)

    print("\n--- Summary ---")
    for pde in pdes:
        print(pde, summary[pde])


if __name__ == "__main__":
    main()
