#!/usr/bin/env python3
"""
Generate sample video frames figure for VIPER supplementary.
Produces a 5x3 grid: rows = PDEs, cols = clean / noisy / 20% coverage.

Usage:
    python generate_sample_frames.py

Output:
    figures/sample_frames.pdf

Place this script in your experiments/ folder, or adjust
sys.path below to point to your project root.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---- Try importing your project solvers ----
# Adjust this path to your project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from simp_video.utils.pde_solvers import solve_pde
    HAS_SOLVER = True
    print("Using project PDE solvers")
except ImportError:
    HAS_SOLVER = False
    print("WARNING: Could not import solve_pde.")
    print("Using built-in synthetic fallback data.")


# ---- PDE configs (matching your experiment_configs.yaml) ----
PDE_CONFIGS = {
    "kdv": {
        "L": 20, "nx": 128, "T": 10, "dt": 0.01,
        "true_coeffs": {"alpha": 6.0, "beta": 1.0},
        "label": "KdV",
    },
    "burgers": {
        "L": 8, "nx": 128, "T": 2, "dt": 0.005,
        "true_coeffs": {"nu": 0.1},
        "label": "Burgers",
    },
    "ks": {
        "L": 64, "nx": 128, "T": 50, "dt": 0.05,
        "true_coeffs": {"nu": 1.0},
        "label": "KS",
    },
    "schrodinger": {
        "L": 20, "nx": 128, "T": 5, "dt": 0.01,
        "true_coeffs": {"alpha": 0.5},
        "label": r"Schr$\ddot{o}$dinger",
    },
    "nls": {
        "L": 20, "nx": 128, "T": 5, "dt": 0.01,
        "true_coeffs": {"alpha": 0.5, "beta": 1.0},
        "label": "NLS",
    },
}

PDE_ORDER = ["kdv", "burgers", "ks", "schrodinger", "nls"]
SEED = 42


def generate_field(pde_name):
    """Generate u(x,t) for a PDE."""
    cfg = PDE_CONFIGS[pde_name]
    L = cfg["L"]
    nx = cfg["nx"]
    T = cfg["T"]
    dt = cfg["dt"]

    x = np.linspace(-L / 2, L / 2, nx)
    t = np.arange(0, T + dt / 2, dt)

    if HAS_SOLVER:
        u = solve_pde(
            pde_name, x, t,
            cfg["true_coeffs"], seed=SEED
        )
    else:
        # Fallback: generate simple synthetic data
        np.random.seed(SEED)
        X, T_grid = np.meshgrid(x, t)
        if pde_name == "kdv":
            u = 2 * np.exp(-(X - 0.5 * T_grid)**2)
        elif pde_name == "burgers":
            u = -np.sin(np.pi * X / 4) * np.exp(-0.1 * T_grid)
        elif pde_name == "ks":
            u = np.cos(2 * np.pi * X / L) * np.sin(0.1 * T_grid + 0.3)
            u += 0.3 * np.random.randn(*u.shape)
        elif pde_name == "schrodinger":
            u = np.abs(np.exp(-X**2) * np.exp(1j * T_grid))
        elif pde_name == "nls":
            u = 1.0 / np.cosh(X) * np.cos(T_grid)
        else:
            u = np.sin(X) * np.cos(T_grid)

    return u, x, t


def add_noise(u, level=0.05):
    """Add Gaussian noise at given level."""
    np.random.seed(SEED + 1)
    sigma = level * np.std(u)
    return u + sigma * np.random.randn(*u.shape)


def apply_occlusion(u, coverage=0.2):
    """Mask spatial columns and interpolate."""
    from scipy import interpolate as sci_interp

    np.random.seed(SEED + 2)
    nx = u.shape[1]
    n_keep = max(1, int(nx * coverage))
    idx = np.random.permutation(nx)[:n_keep]
    mask = np.zeros(nx, dtype=bool)
    mask[idx] = True

    u_occ = u.copy()
    u_occ[:, ~mask] = np.nan

    # Interpolate missing
    u_filled = u_occ.copy()
    for ti in range(u.shape[0]):
        valid = ~np.isnan(u_occ[ti])
        if valid.sum() < 2:
            u_filled[ti] = np.nanmean(u)
            continue
        x_valid = np.where(valid)[0]
        f = sci_interp.interp1d(
            x_valid, u_occ[ti, valid],
            kind="linear", fill_value="extrapolate"
        )
        u_filled[ti] = f(np.arange(nx))

    return u_filled, mask


def pick_frame(u, frac=0.3):
    """Pick a representative frame at frac of total time."""
    idx = int(u.shape[0] * frac)
    idx = min(idx, u.shape[0] - 1)
    return u[idx]


def main():
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 12))
    gs = GridSpec(
        5, 3, figure=fig,
        wspace=0.08, hspace=0.25,
        left=0.08, right=0.95,
        top=0.95, bottom=0.03
    )

    col_titles = [
        "(a) Clean",
        "(b) 5% Gaussian noise",
        "(c) 20% spatial coverage"
    ]

    for row_idx, pde_name in enumerate(PDE_ORDER):
        cfg = PDE_CONFIGS[pde_name]
        label = cfg["label"]

        print(f"Generating {pde_name}...")
        u, x, t = generate_field(pde_name)

        # Pick a time frame at ~30% of simulation
        frame_clean = pick_frame(u, frac=0.3)
        frame_noisy = pick_frame(add_noise(u, 0.05), frac=0.3)

        u_occ, mask = apply_occlusion(u, 0.2)
        frame_partial = pick_frame(u_occ, frac=0.3)

        frames = [frame_clean, frame_noisy, frame_partial]

        # Use consistent color limits across the 3
        # conditions for each PDE
        vmin = min(f.min() for f in frames)
        vmax = max(f.max() for f in frames)

        for col_idx, frame in enumerate(frames):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            # Tile the 1D field vertically to make
            # a 2D heatmap (matching paper's rendering)
            img = np.tile(
                frame.reshape(1, -1), (32, 1)
            )
            ax.imshow(
                img, aspect="auto",
                cmap="viridis",
                vmin=vmin, vmax=vmax,
                interpolation="nearest"
            )
            ax.set_xticks([])
            ax.set_yticks([])

            # Column titles on top row
            if row_idx == 0:
                ax.set_title(
                    col_titles[col_idx],
                    fontsize=11, fontweight="bold"
                )

            # Row labels on left column
            if col_idx == 0:
                ax.set_ylabel(
                    label, fontsize=11,
                    fontweight="bold",
                    rotation=0, labelpad=45,
                    va="center"
                )

    out_path = out_dir / "sample_frames.pdf"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # Also save PNG for quick viewing
    out_png = out_dir / "sample_frames.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")

    plt.close(fig)


if __name__ == "__main__":
    main()