#!/usr/bin/env python3
"""
Experiment 2: Noisy Video Benchmarks
Test robustness to Gaussian, salt-pepper, blur, compression.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml
from simp_video.utils.pde_solvers import solve_pde
from simp_video.models.simp_v import train_simp_v
from simp_video.models.baselines.pde_find import pde_find_video


def add_noise(u: np.ndarray, noise_type: str, level: float) -> np.ndarray:
    """Add noise to u."""
    u = u.copy()
    if noise_type == "gaussian":
        σ = level * np.std(u)
        u = u + σ * np.random.randn(*u.shape)
    elif noise_type == "salt_pepper":
        mask = np.random.rand(*u.shape) < level
        u[mask] = np.random.choice([u.min(), u.max()], size=mask.sum())
    elif noise_type == "blur":
        from scipy.ndimage import uniform_filter1d
        k = int(level)
        if k > 0:
            for i in range(u.shape[0]):
                u[i] = uniform_filter1d(u[i], size=k, mode="wrap")
    return u


def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "experiment_configs.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_single(pde_name: str, seed: int, noise_type: str, level: float, config: dict, max_epochs: int = None) -> dict:
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
    u = add_noise(u, noise_type, level)

    r_simp = train_simp_v(
        u, x, t, pde_name, true_coeffs, coeff_order,
        max_epochs=max_epochs or config["training_configs"]["max_epochs"],
    )
    r_pdefind = pde_find_video(u, x, t, pde_name, true_coeffs, coeff_order)

    return {
        "simp_v": float(r_simp["coefficient_error_pct"]),
        "pde_find": float(r_pdefind["coefficient_error_pct"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", type=str, default=None)
    parser.add_argument("--noise", type=str, default="gaussian")
    parser.add_argument("--level", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    pdes = list(config["pde_configs"].keys()) if args.pde is None else [args.pde]
    seeds = config["seeds"][: args.seeds]

    noise_levels = {"gaussian": [0.01, 0.05, 0.10], "blur": [3, 5, 7], "salt_pepper": [0.01, 0.05]}
    levels = [args.level] if args.level else noise_levels.get(args.noise, [0.05])

    out_dir = Path(__file__).parent.parent / "results" / "exp2_noisy"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for pde in pdes:
        all_results[pde] = {}
        for level in levels:
            key = f"{args.noise}_{level}"
            all_results[pde][key] = {"simp_v": [], "pde_find": []}
            for seed in seeds:
                r = run_single(pde, seed, args.noise, level, config, max_epochs=args.max_epochs)
                all_results[pde][key]["simp_v"].append(r["simp_v"])
                all_results[pde][key]["pde_find"].append(r["pde_find"])
            print(f"{pde} {key}: SIMP-V={np.mean(all_results[pde][key]['simp_v']):.2f}%  PDE-FIND={np.mean(all_results[pde][key]['pde_find']):.2f}%")

    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
