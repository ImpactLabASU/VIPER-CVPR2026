#!/usr/bin/env python3
"""
Experiment 4 & 5: Ablation Studies
A1: LTC vs GRU/LSTM/Transformer
A2: Loss components
A3: Hidden units
A4: Video resolution (via subsampling)
A5: FPS (temporal subsampling)
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


def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "experiment_configs.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_ablation(pde_name: str, seed: int, variant: str, value, config: dict) -> float:
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

    if variant == "resolution":
        factor = int(value)  # 64, 128, 256, 512
        if factor < nx:
            step = nx // factor
            u = u[:, ::step]
            x = x[::step]
    elif variant == "fps":
        # subsample time
        target_nt = int(value)  # ~fps * duration
        if len(t) > target_nt:
            idx = np.linspace(0, len(t) - 1, target_nt).astype(int)
            u = u[idx]
            t = t[idx]
            dt = t[1] - t[0]
    elif variant == "model":
        pass

    model_type = value if variant == "model" else "ltc"
    r = train_simp_v(
        u, x, t, pde_name, true_coeffs, coeff_order,
        max_epochs=config["training_configs"]["max_epochs"],
        model_type=model_type,
    )
    return r["coefficient_error_pct"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, choices=["hidden", "resolution", "fps", "model"], default="hidden")
    parser.add_argument("--pde", type=str, default="burgers")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list for model ablation, e.g., ltc,lstm,node")
    args = parser.parse_args()

    config = load_config()
    seeds = config["seeds"][: args.seeds]

    if args.ablation == "hidden":
        values = [16, 32, 64, 128]
    elif args.ablation == "resolution":
        values = [64, 128, 256]
    elif args.ablation == "fps":
        values = [50, 100, 200, 400]  # nt values
    elif args.ablation == "model":
        values = ["ltc", "lstm", "node"]
        if args.models:
            values = [m.strip() for m in args.models.split(",") if m.strip()]

    out_dir = Path(__file__).parent.parent / "results" / "exp4_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {str(v): [] for v in values}
    for v in values:
        for seed in seeds:
            err = run_ablation(args.pde, seed, args.ablation, v, config)
            results[str(v)].append(err)
        print(f"{args.ablation}={v}: {np.mean(results[str(v)]):.2f}%")

    suffix = ""
    if args.ablation == "model" and args.models:
        safe = "_".join(values)
        suffix = f"_{safe}"
    with open(out_dir / f"ablation_{args.ablation}_{args.pde}{suffix}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
