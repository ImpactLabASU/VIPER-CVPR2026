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
) -> str:
    """
    Render u(nt, nx) to video.
    
    Args:
        u: Field (nt, nx)
        x: Spatial grid
        t: Time grid
        out_path: Output file path
        resolution: (width, height) in pixels
        fps: Frames per second
        colormap: Matplotlib colormap name
        vmin, vmax: Color scale limits
    
    Returns:
        Path to saved video
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    if vmin is None:
        vmin = float(np.nanmin(u))
    if vmax is None:
        vmax = float(np.nanmax(u))
    
    # Subsample time to match fps and duration
    nt_total = u.shape[0]
    duration_sec = t[-1] - t[0]
    n_frames = int(duration_sec * fps)
    if n_frames > nt_total:
        frame_indices = np.linspace(0, nt_total - 1, n_frames).astype(int)
    else:
        frame_indices = np.arange(min(n_frames, nt_total))
    
    dpi = 100
    fig_w, fig_h = resolution[0] / dpi, resolution[1] / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_axis_off()

    # Each frame: u(t_i, x) repeated vertically to fill resolution (width=nx, height=res_h)
    frame0 = np.tile(u[0:1].T, (resolution[1], 1))  # (height, width)
    im = ax.imshow(
        frame0,
        aspect="auto",
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        extent=[x[0], x[-1], t[0], t[-1]],
        origin="lower",
    )

    def update(frame_idx):
        idx = frame_indices[min(frame_idx, len(frame_indices) - 1)]
        frame = np.tile(u[idx : idx + 1].T, (resolution[1], 1))
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
