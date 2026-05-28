#!/usr/bin/env python3
"""
Run 2 additional seeds for Experiment 3 (Implicit Dynamics) and merge with existing results.
Produces updated CSV with 3 seeds total and computes std.
Saves incrementally after each run so progress is visible.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml
from simp_video.utils.pde_solvers import solve_pde
from simp_video.models.simp_v import train_simp_v
from simp_video.models.baselines.pde_find import pde_find_video

# Reuse occlusion/interpolation from exp3
from experiments.exp3_implicit import apply_occlusion, interpolate_missing


def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "experiment_configs.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_single(pde_name, seed, coverage, config):
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
        max_epochs=config["training_configs"]["max_epochs"],
        log_every=0,
        log_prefix=f"{pde_name} cov={int(coverage*100)}% seed={seed}",
    )
    r_pdefind = pde_find_video(u_filled, x, t, pde_name, true_coeffs, coeff_order)

    return {
        "simp_v": float(r_simp["coefficient_error_pct"]),
        "pde_find": float(r_pdefind["coefficient_error_pct"]),
    }


def load_existing_rows(csv_path):
    rows = []
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def save_csv(rows, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pde", "coverage", "seed_idx", "simp_v", "pde_find"])
        writer.writeheader()
        writer.writerows(rows)


def save_summary(rows, out_dir):
    from collections import defaultdict
    grouped = defaultdict(lambda: {"simp_v": [], "pde_find": []})
    for row in rows:
        key = (row["pde"], row["coverage"])
        grouped[key]["simp_v"].append(float(row["simp_v"]))
        grouped[key]["pde_find"].append(float(row["pde_find"]))

    summary_csv = out_dir / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pde", "coverage", "simp_v_mean", "simp_v_std", "pde_find_mean", "pde_find_std", "improvement_x", "n_seeds"])
        for (pde, cov), vals in sorted(grouped.items()):
            sv = np.array(vals["simp_v"])
            pf = np.array(vals["pde_find"])
            imp = np.mean(pf) / max(np.mean(sv), 1e-6)
            writer.writerow([
                pde, cov,
                f"{np.mean(sv):.4f}", f"{np.std(sv):.4f}",
                f"{np.mean(pf):.4f}", f"{np.std(pf):.4f}",
                f"{imp:.2f}", len(sv),
            ])

    json_data = {}
    for (pde, cov), vals in sorted(grouped.items()):
        if pde not in json_data:
            json_data[pde] = {}
        json_data[pde][cov] = vals
    with open(out_dir / "results.json", "w") as f:
        json.dump(json_data, f, indent=2)


def main():
    config = load_config()
    pdes = list(config["pde_configs"].keys())
    coverages = config["implicit_configs"]["coverage"]  # [0.2, 0.4, 0.6, 0.8, 1.0]

    all_seeds = config["seeds"]  # [42, 123, 456, 789, 1011]
    new_seed_indices = [1, 2]
    new_seeds = [all_seeds[i] for i in new_seed_indices]

    out_dir = Path(__file__).parent.parent / "results" / "exp3_implicit"
    csv_path = out_dir / "results.csv"

    # Load existing rows
    all_rows = load_existing_rows(csv_path)

    # Check which runs already exist (to allow resuming)
    existing_keys = set()
    for row in all_rows:
        existing_keys.add((row["pde"], row["coverage"], row["seed_idx"]))

    total = len(pdes) * len(coverages) * len(new_seeds)
    done = 0
    skipped = 0
    for pde in pdes:
        for cov in coverages:
            for seed_idx, seed in zip(new_seed_indices, new_seeds):
                done += 1
                key = (pde, f"{int(cov*100)}%", str(seed_idx))
                if key in existing_keys:
                    skipped += 1
                    print(f"[{done}/{total}] SKIP {pde} cov={int(cov*100)}% seed={seed} (already done)", flush=True)
                    continue

                print(f"[{done}/{total}] {pde} cov={int(cov*100)}% seed={seed} (idx={seed_idx})", flush=True)
                r = run_single(pde, seed, cov, config)
                new_row = {
                    "pde": pde,
                    "coverage": f"{int(cov*100)}%",
                    "seed_idx": str(seed_idx),
                    "simp_v": str(r["simp_v"]),
                    "pde_find": str(r["pde_find"]),
                }
                all_rows.append(new_row)
                print(f"  VIPER={r['simp_v']:.2f}%  PDE-FIND={r['pde_find']:.2f}%", flush=True)

                # Save incrementally
                save_csv(all_rows, csv_path)
                save_summary(all_rows, out_dir)
                print(f"  Saved ({len(all_rows)} rows total)", flush=True)

    print(f"\nDone! {total - skipped} new runs, {skipped} skipped. Total rows: {len(all_rows)}", flush=True)


if __name__ == "__main__":
    main()
