"""
Extract u(x,t) from 3D surface videos (PRISM).

Supports two modes:
  - topdown: orthographic top-down renders (direct colormap inversion)
  - uvmap: multi-view rotating renders using a UV map pass
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy.interpolate import griddata
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .video_extract import _get_colormap_scale, _rgb_to_scalar


def _read_video_frames(video_path: str, stride: int = 1, max_frames: Optional[int] = None) -> np.ndarray:
    if not HAS_CV2:
        raise ImportError("opencv-python (cv2) required for video extraction")
    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % max(1, stride) == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            if max_frames is not None and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from {video_path}")
    return np.array(frames)


def _load_metadata(metadata_path: Optional[str], video_path: str) -> Optional[Dict]:
    if metadata_path is None:
        metadata_path = str(Path(video_path).with_suffix(".json"))
    p = Path(metadata_path)
    if not p.exists():
        return None
    with open(p, "r") as f:
        return json.load(f)


def _camera_path(total_frames: int, meta: Optional[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    if meta and "view_path" in meta:
        vp = meta["view_path"]
        az0 = float(vp.get("azim_start", 0.0))
        az1 = float(vp.get("azim_end", 360.0))
        elev_base = float(vp.get("elev_base", 20.0))
        elev_amp = float(vp.get("elev_amp", 15.0))
    else:
        az0, az1, elev_base, elev_amp = 0.0, 360.0, 20.0, 15.0
    azim = np.linspace(az0, az1, total_frames, endpoint=False)
    phase = np.linspace(0, 2 * np.pi, total_frames, endpoint=False)
    elev = elev_base + elev_amp * np.sin(phase)
    return elev, azim


def _render_uv_map(
    x: np.ndarray,
    t: np.ndarray,
    elev: float,
    azim: float,
    figsize: Tuple[float, float],
    dpi: int,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    zlim: Tuple[float, float],
    proj_type: str = "persp",
    bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for UV map rendering")
    X, T = np.meshgrid(x, t)
    Z = np.zeros_like(X)

    x_cell = 0.5 * (X[:-1, :-1] + X[1:, 1:])
    t_cell = 0.5 * (T[:-1, :-1] + T[1:, 1:])
    denom_x = max(1e-9, xlim[1] - xlim[0])
    denom_t = max(1e-9, ylim[1] - ylim[0])
    x_norm = (x_cell - xlim[0]) / denom_x
    t_norm = (t_cell - ylim[0]) / denom_t

    colors = np.zeros(x_cell.shape + (4,), dtype=float)
    colors[..., 0] = np.clip(x_norm, 0.0, 1.0)
    colors[..., 1] = np.clip(t_norm, 0.0, 1.0)
    colors[..., 2] = 0.0
    colors[..., 3] = 1.0

    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(bg_color)
    ax = fig.add_subplot(111, projection="3d")
    if hasattr(ax, "set_proj_type"):
        ax.set_proj_type(proj_type)
    ax.set_facecolor(bg_color)
    ax.set_axis_off()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.view_init(elev=elev, azim=azim)

    ax.plot_surface(
        X, T, Z,
        facecolors=colors,
        shade=False,
        antialiased=False,
        linewidth=0,
    )

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(height, width, 3)
    plt.close(fig)
    return img


def _resample_grid(u_img: np.ndarray, nt: int, nx: int) -> np.ndarray:
    t_old = np.linspace(0.0, 1.0, u_img.shape[0])
    x_old = np.linspace(0.0, 1.0, u_img.shape[1])
    t_new = np.linspace(0.0, 1.0, nt)
    x_new = np.linspace(0.0, 1.0, nx)

    u_x = np.empty((u_img.shape[0], nx), dtype=float)
    for i in range(u_img.shape[0]):
        u_x[i] = np.interp(x_new, x_old, u_img[i])

    u_out = np.empty((nt, nx), dtype=float)
    for j in range(nx):
        u_out[:, j] = np.interp(t_new, t_old, u_x[:, j])

    return u_out


def extract_field_from_3d_video(
    video_path: str,
    x_extent: Optional[Tuple[float, float]] = None,
    t_extent: Optional[Tuple[float, float]] = None,
    nx: int = 128,
    nt: int = 200,
    colormap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    mode: str = "auto",
    stride: int = 2,
    max_frames: Optional[int] = None,
    metadata_path: Optional[str] = None,
    fill_missing: bool = True,
    flip_y: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract u(x,t) from a 3D surface video.

    Args:
        video_path: Path to MP4 rendered by video_render_3d.py
        x_extent: (x_min, x_max) spatial range
        t_extent: (t_min, t_max) time range
        nx, nt: Output grid size
        colormap: Colormap used during render
        vmin, vmax: Scalar range used during render
        mode: "auto", "topdown", or "uvmap"
        stride: Frame stride for uvmap mode
        max_frames: Limit frames processed
        metadata_path: Optional sidecar JSON (defaults to video_path with .json)
        fill_missing: Fill NaNs (uvmap mode) via interpolation if available
        flip_y: Flip vertical axis for topdown mode
    """
    meta = _load_metadata(metadata_path, video_path)
    frames = _read_video_frames(video_path, stride=stride, max_frames=max_frames)

    if meta and vmin is None:
        vmin = meta.get("vmin", None)
    if meta and vmax is None:
        vmax = meta.get("vmax", None)
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = 1.0

    if x_extent is None:
        if meta and "x_range" in meta:
            x_extent = tuple(meta["x_range"])
        else:
            x_extent = (0.0, 1.0)
    if t_extent is None:
        if meta and "t_range" in meta:
            t_extent = tuple(meta["t_range"])
        else:
            t_extent = (0.0, 1.0)

    if mode == "auto":
        if meta and meta.get("proj_type") == "ortho":
            vp = meta.get("view_path", {})
            elev_amp = float(vp.get("elev_amp", 0.0))
            az0 = float(vp.get("azim_start", 0.0))
            az1 = float(vp.get("azim_end", 0.0))
            if meta.get("view_mode") == "topdown" or (elev_amp == 0.0 and az0 == az1):
                mode = "topdown"
            else:
                mode = "uvmap"
        else:
            mode = "uvmap"

    cmap_rgb, cmap_vals = _get_colormap_scale(colormap)
    cmap_vals_scaled = vmin + cmap_vals * (vmax - vmin)

    if mode == "topdown":
        frame = frames[0]
        u_img = _rgb_to_scalar(frame, cmap_rgb, cmap_vals_scaled)
        if flip_y:
            u_img = np.flipud(u_img)
        u = _resample_grid(u_img, nt=nt, nx=nx)
        x = np.linspace(x_extent[0], x_extent[1], nx)
        t = np.linspace(t_extent[0], t_extent[1], nt)
        return x, t, u

    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for uvmap extraction")

    figsize = (8.0, 6.0)
    dpi = 100
    proj_type = "persp"
    xlim = x_extent
    ylim = t_extent
    zlim = (vmin, vmax)
    if meta:
        if "figsize" in meta:
            figsize = tuple(meta["figsize"])
        if "dpi" in meta:
            dpi = int(meta["dpi"])
        if "proj_type" in meta:
            proj_type = meta["proj_type"]
        if "xlim" in meta:
            xlim = tuple(meta["xlim"])
        if "ylim" in meta:
            ylim = tuple(meta["ylim"])
        if "zlim" in meta:
            zlim = tuple(meta["zlim"])

    elev, azim = _camera_path(len(frames), meta)

    x_vals = []
    t_vals = []
    u_vals = []

    for i in range(len(frames)):
        uv_map = _render_uv_map(
            np.linspace(x_extent[0], x_extent[1], nx),
            np.linspace(t_extent[0], t_extent[1], nt),
            float(elev[i]),
            float(azim[i]),
            figsize=figsize,
            dpi=dpi,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            proj_type=proj_type,
        )
        uv_bg = uv_map[0, 0].astype(float)
        uv_diff = np.mean(np.abs(uv_map.astype(float) - uv_bg), axis=-1)
        mask = uv_diff > 2.0

        x_norm = uv_map[..., 0].astype(float) / 255.0
        t_norm = uv_map[..., 1].astype(float) / 255.0
        x_map = x_extent[0] + x_norm * (x_extent[1] - x_extent[0])
        t_map = t_extent[0] + t_norm * (t_extent[1] - t_extent[0])

        u_map = _rgb_to_scalar(frames[i], cmap_rgb, cmap_vals_scaled)

        x_vals.append(x_map[mask])
        t_vals.append(t_map[mask])
        u_vals.append(u_map[mask])

    xs = np.concatenate(x_vals)
    ts = np.concatenate(t_vals)
    us = np.concatenate(u_vals)

    x_grid = np.linspace(x_extent[0], x_extent[1], nx)
    t_grid = np.linspace(t_extent[0], t_extent[1], nt)
    ix = np.clip(np.searchsorted(x_grid, xs) - 1, 0, nx - 1)
    it = np.clip(np.searchsorted(t_grid, ts) - 1, 0, nt - 1)

    accum = np.zeros((nt, nx), dtype=float)
    counts = np.zeros((nt, nx), dtype=float)
    np.add.at(accum, (it, ix), us)
    np.add.at(counts, (it, ix), 1.0)
    u = np.where(counts > 0, accum / counts, np.nan)

    if fill_missing and np.isnan(u).any() and HAS_SCIPY:
        grid_x, grid_t = np.meshgrid(x_grid, t_grid)
        valid = ~np.isnan(u)
        pts = np.column_stack([grid_x[valid], grid_t[valid]])
        vals = u[valid]
        u = griddata(pts, vals, (grid_x, grid_t), method="nearest")

    return x_grid, t_grid, u
