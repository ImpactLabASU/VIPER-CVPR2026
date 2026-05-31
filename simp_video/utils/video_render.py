"""
Render u(x,t) field to video (MP4).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple

try:
    import matplotlib.animation as animation
    HAS_ANIMATION = True
except ImportError:
    HAS_ANIMATION = False


def render_field_to_video(
    u: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    out_path: str,
    resolution: Tuple[int, int] = (256, 128),
    fps: int = 30,
    colormap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fmt: str = "mp4",
    subsample: bool = True,
) -> str:
    """
    Render u(nt, nx) to video.

    Each video frame encodes one time slice u(t_i, x) as a colormap strip with
    the spatial coordinate x along the WIDTH (one column per grid point),
    replicated down the height. This matches extract_field_from_video, which
    reads x from the frame width and averages over height.

    Args:
        u: Field (nt, nx)
        x: Spatial grid
        t: Time grid
        out_path: Output file path
        resolution: (width, height) in pixels
        fps: Frames per second
        colormap: Matplotlib colormap name
        vmin, vmax: Color scale limits
        subsample: If True (default, for viewable videos) sample n=duration*fps
            frames evenly across [0, T]. If False, render one frame per time
            step (no temporal loss) so the round-trip preserves dt for recovery.

    Returns:
        Path to saved video
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if vmin is None:
        vmin = float(np.nanmin(u))
    if vmax is None:
        vmax = float(np.nanmax(u))

    nt_total = u.shape[0]
    if subsample:
        # Sample n_frames evenly across the FULL [0, T] range (never the first-N).
        duration_sec = t[-1] - t[0]
        n_frames = max(1, int(duration_sec * fps))
        frame_indices = np.linspace(0, nt_total - 1, n_frames).astype(int)
    else:
        # Faithful: one frame per time step, preserves the true time cadence.
        frame_indices = np.arange(nt_total)

    dpi = 100
    fig_w, fig_h = resolution[0] / dpi, resolution[1] / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Each frame: u(t_i, x) with x along width, replicated over height.
    frame0 = np.tile(u[0:1], (resolution[1], 1))  # (height, nx)
    im = ax.imshow(
        frame0,
        aspect="auto",
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        extent=[x[0], x[-1], t[0], t[-1]],
        origin="lower",
    )

    def update(frame_idx):
        idx = frame_indices[min(frame_idx, len(frame_indices) - 1)]
        frame = np.tile(u[idx : idx + 1], (resolution[1], 1))
        im.set_data(frame)
        return [im]
    
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000.0 / fps,
        blit=True,
    )
    
    writer = "ffmpeg" if fmt == "mp4" else "pillow"
    anim.save(out_path, writer=writer, fps=fps, dpi=dpi)
    plt.close(fig)
    
    return out_path
