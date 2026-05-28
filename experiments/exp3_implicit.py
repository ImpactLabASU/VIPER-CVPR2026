#!/usr/bin/env python3
"""
Experiment 3: Implicit Dynamics (Partial Observability)
Recover coefficients when parts of spatial domain are occluded.
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


def apply_occlusion(u: np.ndarray, x: np.ndarray, coverage: float, pattern: str = "random", seed: int = 42) -> np.ndarray:
    """Occlude columns to achieve given coverage. Unobserved = NaN (will interpolate or mask)."""
    np.random.seed(seed)
    nx = u.shape[1]
    n_keep = max(1, int(nx * coverage))

    if pattern == "random":
        idx = np.random.permutation(nx)[:n_keep]
        mask = np.zeros(nx, dtype=bool)
        mask[idx] = True
    elif pattern == "block":
        start = np.random.randint(0, nx - n_keep + 1) if n_keep < nx else 0
        mask = np.zeros(nx, dtype=bool)
        mask[start : start + n_keep] = True
    else:
        # periodic
        step = max(1, nx // n_keep)
        mask = np.zeros(nx, dtype=bool)
        mask[::step] = True
        mask = mask[:nx]

    u_occ = u.copy()
    u_occ[:, ~mask] = np.nan
    return u_occ, mask


def interpolate_missing(u: np.ndarray) -> np.ndarray:
    """Fill NaN with interpolation along x."""
    from scipy import interpolate
    u_out = u.copy()
    for t in range(u.shape[0]):
        valid = ~np.isnan(u[t])
        if valid.sum() < 2:
            u_out[t] = np.nanmean(u)
            continue
        x_valid = np.where(valid)[0]
        f = interpolate.interp1d(x_valid, u[t, valid], kind="linear", fill_value="extrapolate")
        u_out[t] = f(np.arange(u.shape[1]))
    return u_out


def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "experiment_configs.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_single(
    pde_name: str,
    seed: int,
    coverage: float,
    config: dict,
    log_every: int = 0,
    max_epochs: int = None,
) -> dict:
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
    u_occ, _ = apply_occlusion(u, x, coverage, "random", seed)
    u_filled = interpolate_missing(u_occ)

    r_simp = train_simp_v(
        u_filled, x, t, pde_name, true_coeffs, coeff_order,
        max_epochs=max_epochs or config["training_configs"]["max_epochs"],
        log_every=log_every,
        log_prefix=f"{pde_name} cov={int(coverage*100)}% seed={seed}",
    )
    r_pdefind = pde_find_video(u_filled, x, t, pde_name, true_coeffs, coeff_order)

    return {
        "simp_v": float(r_simp["coefficient_error_pct"]),
        "pde_find": float(r_pdefind["coefficient_error_pct"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", type=str, default=None)
    parser.add_argument("--coverage", type=float, default=None)
    parser.add_argument("--coverages", type=str, default=None, help="Comma-separated list, e.g., 0.2,0.4")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    if args.log_every is not None:
        config.setdefault("training_configs", {})
        config["training_configs"]["log_every"] = args.log_every
    pdes = list(config["pde_configs"].keys())
    seeds = config["seeds"][: args.seeds]
    coverages = config["implicit_configs"]["coverage"]

    if args.pde:
        pdes = [args.pde]
    if args.coverages:
        coverages = [float(x) for x in args.coverages.split(",") if x.strip()]
    if args.coverage is not None:
        coverages = [args.coverage]

    out_dir = Path(__file__).parent.parent / "results" / "exp3_implicit"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for pde in pdes:
        all_results[pde] = {}
        for cov in coverages:
            all_results[pde][f"{int(cov*100)}%"] = {"simp_v": [], "pde_find": []}
            for seed in seeds:
                r = run_single(
                    pde, seed, cov, config,
                    log_every=config["training_configs"].get("log_every", 0),
                    max_epochs=args.max_epochs,
                )
                all_results[pde][f"{int(cov*100)}%"]["simp_v"].append(r["simp_v"])
                all_results[pde][f"{int(cov*100)}%"]["pde_find"].append(r["pde_find"])
            print(f"{pde} {int(cov*100)}%: SIMP-V={np.mean(all_results[pde][f'{int(cov*100)}%']['simp_v']):.2f}%")

    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
