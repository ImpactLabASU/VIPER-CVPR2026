#!/usr/bin/env python3
"""
PRISM 3D Video Pipeline: Simulate → Render 3D → Extract → Recover coefficients

Generates u(x,t) for each PDE, renders as rotating 3D surface, evolving surface,
and wireframe videos. Uses existing simp_video PDE solvers.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml
from simp_video.utils.pde_solvers import solve_pde
from simp_video.utils.video_render_3d import (
    render_3d_surface_video,
    render_3d_evolving_video,
    render_3d_wireframe_video,
    render_3d_with_noise,
    render_3d_with_occlusion,
)


def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "experiment_configs.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_pde_data(pde_name: str, config: dict, nx: int = 128, nt: int = 200, seed: int = 42):
    """Generate u(x,t) for each PDE using existing solvers."""
    cfg = config["pde_configs"][pde_name]
    L = cfg["domain"]["L"]
    T = cfg["time"]["T"]
    dt = cfg["time"]["dt"]
    true_coeffs = cfg["true_coeffs"]

    x = np.linspace(-L / 2, L / 2, nx)
    t = np.arange(0, T + dt / 2, dt)[:nt]

    u_xt = solve_pde(pde_name, x, t, true_coeffs, seed=seed)
    return u_xt, x, t, true_coeffs


def main():
    parser = argparse.ArgumentParser(description="Generate 3D PDE videos for PRISM")
    parser.add_argument("--pde", type=str, default=None, choices=["kdv", "burgers", "ks", "schrodinger", "nls"])
    parser.add_argument("--style", type=str, default="all", choices=["all", "rotating", "evolving", "wireframe", "topdown"])
    parser.add_argument("--noise", type=float, default=None, help="Add Gaussian noise level (e.g. 0.05)")
    parser.add_argument("--occlusion", type=float, default=None, help="Spatial coverage (e.g. 0.6 for 60%%)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    base_dir = Path(__file__).parent.parent / "data" / "videos" / "3d"
    if args.out_dir:
        base_dir = Path(args.out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    sim_dir = Path(__file__).parent.parent / "data" / "simulations"
    sim_dir.mkdir(parents=True, exist_ok=True)

    pdes = [args.pde] if args.pde else list(config["pde_configs"].keys())

    for pde_name in pdes:
        print(f"\n=== Generating {pde_name.upper()} ===")

        u_xt, x, t, true_coeffs = generate_pde_data(pde_name, config, seed=args.seed)
        print(f"  Shape: {u_xt.shape}, True coeffs: {true_coeffs}")

        np.savez(
            sim_dir / f"{pde_name}_3d_seed{args.seed}.npz",
            u=u_xt, x=x, t=t, true_coeffs=true_coeffs, pde=pde_name
        )

        prefix_parts = [pde_name]
        if args.noise is not None:
            np.random.seed(args.seed)
            u_xt = u_xt + args.noise * np.std(u_xt) * np.random.randn(*u_xt.shape)
            prefix_parts.append(f"noise{args.noise}")
        if args.occlusion is not None:
            np.random.seed(args.seed)
            nx = len(x)
            n_keep = max(1, int(nx * args.occlusion))
            visible = np.sort(np.random.choice(nx, n_keep, replace=False))
            u_xt = u_xt[:, visible]
            x = x[visible]
            prefix_parts.append(f"occ{int(args.occlusion*100)}")
        prefix = base_dir / "_".join(prefix_parts)

        if args.style in ("all", "rotating"):
            p = str(prefix) + "_rotating.mp4"
            render_3d_surface_video(u_xt, x, t, p, fps=args.fps, duration=args.duration)
            print(f"  Saved: {p}")

        if args.style in ("all", "evolving"):
            p = str(prefix) + "_evolving.mp4"
            render_3d_evolving_video(u_xt, x, t, p, fps=args.fps)
            print(f"  Saved: {p}")

        if args.style in ("all", "topdown"):
            p = str(prefix) + "_topdown.mp4"
            render_3d_surface_video(
                u_xt, x, t, p,
                fps=args.fps,
                duration=args.duration,
                view_mode="topdown",
                proj_type="ortho",
                azim_start=-90,
                azim_end=-90,
                elev_base=90,
                elev_amp=0,
            )
            print(f"  Saved: {p}")

        if args.style in ("all", "wireframe"):
            p = str(prefix) + "_wireframe.mp4"
            render_3d_wireframe_video(u_xt, x, t, p, fps=args.fps, duration=args.duration)
            print(f"  Saved: {p}")

    print("\n✓ All 3D videos generated!")


if __name__ == "__main__":
    main()
