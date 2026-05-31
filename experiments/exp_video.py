#!/usr/bin/env python3
"""
End-to-end video pipeline: extract u(x,t) from rendered RGB video, then
recover PDE coefficients.

Unlike exp1-exp4 (which feed the raw solver field directly into recovery),
this script exercises the README stage-1 "Video-to-Field Extraction":

    solve_pde -> render to MP4 -> extract field from MP4 (colormap inversion)
              -> recover coefficients (VIPER) -> validate by forward solve_pde

The render/extract pair uses the same (vmin, vmax) so the colormap inversion
recovers the field amplitude; 256-level colormap quantization plus H.264
compression act as realistic video degradation.

Usage:
    python experiments/exp_video.py --pde burgers
    python experiments/exp_video.py --pde kdv --seed 42 --max-epochs 500
    python experiments/exp_video.py --pde ks --video data/videos/ks_3d_seed42.mp4
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from simp_video.utils.pde_solvers import solve_pde
from simp_video.utils.video_render import render_field_to_video
from simp_video.utils.video_extract import extract_field_from_video
from simp_video.models.simp_v import train_simp_v
from simp_video.utils.metrics import coefficient_error


def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "experiment_configs.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def run(pde_name: str, seed: int, config: dict, video_path: str = None,
        max_epochs: int = None) -> dict:
    cfg = config["pde_configs"][pde_name]
    L = cfg["domain"]["L"]
    nx = cfg["domain"]["nx"]
    T = cfg["time"]["T"]
    dt = cfg["time"]["dt"]
    true_coeffs = cfg["true_coeffs"]
    coeff_order = cfg.get("coeff_order", list(true_coeffs.keys()))

    vc = config["video_configs"]
    fps = vc["fps"]
    colormap = vc["colormap"]
    # Render at native spatial resolution (width = nx) so colormap inversion
    # returns the field on the original grid with no spatial interpolation.
    render_height = 64
    resolution = (nx, render_height)

    tc = config["training_configs"]
    if max_epochs is None:
        max_epochs = tc["max_epochs"]

    # --- ground-truth field (also fixes the color scale for invertibility) ---
    x = np.linspace(-L / 2, L / 2, nx)
    t = np.arange(0, T + dt / 2, dt)
    u_true = solve_pde(pde_name, x, t, true_coeffs, seed=seed)
    vmin, vmax = float(np.nanmin(u_true)), float(np.nanmax(u_true))

    # --- render to MP4 with KNOWN (vmin, vmax) so extraction can invert it ---
    tmp = None
    if video_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        video_path = tmp.name
        render_field_to_video(
            u_true, x, t, video_path,
            resolution=resolution, fps=fps, colormap=colormap,
            vmin=vmin, vmax=vmax, subsample=False,
        )
        print(f"[render] wrote video -> {video_path}")

    # --- STAGE 1: extract u(x,t) back from the video (colormap inversion) ---
    x_ext, t_ext, u_video = extract_field_from_video(
        video_path,
        x_extent=(float(x[0]), float(x[-1])),
        t_extent=(float(t[0]), float(t[-1])),
        colormap=colormap,
        vmin=vmin, vmax=vmax,
    )
    print(f"[extract] field from video: u shape {u_video.shape} "
          f"(nt={u_video.shape[0]}, nx={u_video.shape[1]})")

    # --- stage-1 fidelity: how well does the video round-trip recover u? ---
    if u_video.shape == u_true.shape:
        denom = float(np.sqrt(np.nanmean(u_true ** 2))) or 1.0
        extract_rmse = float(np.sqrt(np.nanmean((u_video - u_true) ** 2)))
        corr = float(np.corrcoef(u_video.ravel(), u_true.ravel())[0, 1])
        print(f"[fidelity] extraction RMSE={extract_rmse:.4e} "
              f"(rel={extract_rmse/denom:.2%}), corr={corr:.4f}")
    else:
        print(f"[fidelity] shape mismatch u_video {u_video.shape} "
              f"vs u_true {u_true.shape}; skipping fidelity metric")

    def recover(u_field, x_grid, t_grid, tag):
        # Re-seed before each run so the only difference between the direct and
        # video baselines is the input field, not network initialization.
        torch.manual_seed(seed)
        np.random.seed(seed)
        r = train_simp_v(
            u_field, x_grid, t_grid, pde_name, true_coeffs, coeff_order,
            max_epochs=max_epochs,
            lr=tc["learning_rate"],
            patience=tc["patience"],
            log_every=tc.get("log_every", 0),
            log_prefix=f"{pde_name} ({tag})",
        )
        theta_est = np.asarray(r["theta_est"], dtype=float)
        recovered = {k: float(theta_est[i]) for i, k in enumerate(coeff_order)}
        u_check = solve_pde(pde_name, x, t, recovered, seed=seed)
        field_rmse = float(np.sqrt(np.nanmean((u_check - u_true) ** 2)))
        return recovered, float(r["coefficient_error_pct"]), field_rmse

    # --- baseline: recover from the clean solver field (paper Table 2 setting) ---
    direct_coeffs, direct_err, direct_rmse = recover(u_true, x, t, "solve_pde field")

    # --- recover from the EXTRACTED video field (the actual VIPER claim) ---
    video_coeffs, video_err, video_rmse = recover(u_video, x_ext, t_ext, "from video")

    if tmp is not None:
        Path(video_path).unlink(missing_ok=True)

    result = {
        "pde": pde_name,
        "seed": seed,
        "true_coeffs": {k: float(true_coeffs[k]) for k in coeff_order},
        "direct_coeffs": direct_coeffs,
        "direct_error_pct": direct_err,
        "direct_field_rmse": direct_rmse,
        "recovered_coeffs": video_coeffs,
        "coefficient_error_pct": video_err,
        "forward_field_rmse": video_rmse,
        "video_le_direct": video_err <= direct_err + 1e-9,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Video -> field -> PDE recovery")
    parser.add_argument("--pde", type=str, default="burgers",
                        help="PDE name (kdv, burgers, ks, schrodinger, nls)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", type=str, default=None,
                        help="Use an existing MP4 instead of rendering one")
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    res = run(args.pde, args.seed, config, video_path=args.video,
              max_epochs=args.max_epochs)

    print("\n=== Video pipeline result ===")
    print(f"PDE:                 {res['pde']} (seed {res['seed']})")
    print(f"True coeffs:          {res['true_coeffs']}")
    print(f"Direct (solve_pde):   {res['direct_coeffs']}  err={res['direct_error_pct']:.2f}%")
    print(f"Video (extracted):    {res['recovered_coeffs']}  err={res['coefficient_error_pct']:.2f}%")
    print(f"Forward field RMSE:   direct={res['direct_field_rmse']:.4e}  video={res['forward_field_rmse']:.4e}")
    ok = "OK" if res["video_le_direct"] else "VIOLATION"
    print(f"video_err <= direct_err: {res['video_le_direct']} [{ok}]")


if __name__ == "__main__":
    main()
