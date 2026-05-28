#!/usr/bin/env python3
"""
Render u(x,t) simulations to videos.
Reads from data/simulations/, writes to data/videos/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml
from simp_video.utils.video_render import render_field_to_video


def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "experiment_configs.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Render all simulations")
    parser.add_argument("--pde", type=str)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    sim_dir = Path(__file__).parent.parent / "data" / "simulations"
    out_dir = Path(__file__).parent.parent / "data" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    vc = config["video_configs"]
    res = tuple(vc["resolution"])
    fps = vc["fps"]
    colormap = vc["colormap"]
    fmt = vc["format"]

    files = list(sim_dir.glob("*.npz"))
    if args.pde:
        files = [f for f in files if args.pde in f.name]
    if args.seed is not None:
        files = [f for f in files if f"seed{args.seed}" in f.name]

    for f in files:
        data = np.load(f, allow_pickle=True)
        u, x, t = data["u"], data["x"], data["t"]
        out_path = out_dir / f"{f.stem}.mp4"
        render_field_to_video(u, x, t, str(out_path), resolution=res, fps=fps, colormap=colormap)
        print(f"Rendered {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
