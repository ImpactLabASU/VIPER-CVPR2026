"""
3D surface video rendering for PRISM pipeline.
Render u(x,t) as rotating 3D surface, evolving surface, or wireframe.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Optional, Tuple

try:
    import matplotlib.animation as animation
    HAS_ANIMATION = True
except ImportError:
    HAS_ANIMATION = False


def render_3d_surface_video(
    u_xt: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    save_path: str,
    fps: int = 30,
    duration: float = 10,
    cmap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: int = 100,
    figsize: Tuple[float, float] = (8.0, 6.0),
    azim_start: float = 0.0,
    azim_end: float = 360.0,
    elev_base: float = 20.0,
    elev_amp: float = 15.0,
    shade: bool = True,
    show_axes: bool = True,
    show_colorbar: bool = True,
    proj_type: str = "persp",
    view_mode: str = "rotate",
    metadata_path: Optional[str] = None,
) -> str:
    """
    Render rotating 3D surface video of u(x,t).

    Args:
        u_xt: [nt, nx] solution array
        x: spatial coordinates
        t: time coordinates
        save_path: output .mp4 path
        fps: frames per second
        duration: video duration in seconds
        view_mode: "rotate" (default) or "fixed"/"topdown" to keep camera constant
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if vmin is None:
        vmin = float(np.nanmin(u_xt))
    if vmax is None:
        vmax = float(np.nanmax(u_xt))

    X, T = np.meshgrid(x, t)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        X, T, u_xt, cmap=cmap, edgecolor='none', alpha=0.9,
        vmin=vmin, vmax=vmax, shade=shade
    )
    if hasattr(ax, "set_proj_type"):
        ax.set_proj_type(proj_type)
    if show_axes:
        ax.set_xlabel('Space (x)', fontsize=10)
        ax.set_ylabel('Time (t)', fontsize=10)
        ax.set_zlabel('u(x,t)', fontsize=10)
        ax.set_title('PDE Solution Surface', fontsize=12)
    else:
        ax.set_axis_off()
    if show_colorbar:
        fig.colorbar(surf, shrink=0.5, aspect=10)

    total_frames = int(fps * duration)

    def animate(frame):
        if view_mode in ("fixed", "topdown"):
            azim = azim_start
            elev = elev_base
        else:
            azim = azim_start + frame * ((azim_end - azim_start) / total_frames)
            elev = elev_base + elev_amp * np.sin(2 * np.pi * frame / total_frames)
        ax.view_init(elev=elev, azim=azim)
        return []

    anim = animation.FuncAnimation(
        fig, animate, frames=total_frames, interval=1000 / fps
    )
    anim.save(save_path, writer='ffmpeg', fps=fps, dpi=dpi)
    if metadata_path is not None:
        meta = {
            "type": "surface",
            "view_mode": view_mode,
            "x_range": [float(np.min(x)), float(np.max(x))],
            "t_range": [float(np.min(t)), float(np.max(t))],
            "vmin": float(vmin),
            "vmax": float(vmax),
            "colormap": cmap,
            "fps": int(fps),
            "duration": float(duration),
            "total_frames": int(total_frames),
            "figsize": [float(figsize[0]), float(figsize[1])],
            "dpi": int(dpi),
            "proj_type": proj_type,
            "shade": bool(shade),
            "show_axes": bool(show_axes),
            "show_colorbar": bool(show_colorbar),
            "view_path": {
                "azim_start": float(azim_start),
                "azim_end": float(azim_end),
                "elev_base": float(elev_base),
                "elev_amp": float(elev_amp),
            },
            "xlim": [float(v) for v in ax.get_xlim()],
            "ylim": [float(v) for v in ax.get_ylim()],
            "zlim": [float(v) for v in ax.get_zlim()],
        }
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=2)
    plt.close()
    return save_path


def render_3d_evolving_video(
    u_xt: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    save_path: str,
    fps: int = 30,
    cmap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: int = 100,
) -> str:
    """
    Render 3D video showing surface growing in time.
    The surface "fills in" as time progresses.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    nt, nx = u_xt.shape
    if vmin is None:
        vmin = float(np.nanmin(u_xt))
    if vmax is None:
        vmax = float(np.nanmax(u_xt))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    def animate(frame):
        ax.clear()
        t_idx = min(frame + 1, nt)

        if t_idx > 1:
            X_partial, T_partial = np.meshgrid(x, t[:t_idx])
            ax.plot_surface(
                X_partial, T_partial, u_xt[:t_idx, :],
                cmap=cmap, edgecolor='none', alpha=0.9,
                vmin=vmin, vmax=vmax
            )

        ax.plot(
            x, np.full_like(x, t[t_idx - 1]), u_xt[t_idx - 1, :],
            'r-', lw=2
        )
        ax.set_xlabel('Space (x)')
        ax.set_ylabel('Time (t)')
        ax.set_zlabel('u(x,t)')
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(t[0], t[-1])
        ax.set_zlim(vmin, vmax)
        ax.view_init(elev=25, azim=45)
        return []

    anim = animation.FuncAnimation(fig, animate, frames=nt, interval=1000 / fps)
    anim.save(save_path, writer='ffmpeg', fps=fps, dpi=dpi)
    plt.close()
    return save_path


def render_3d_wireframe_video(
    u_xt: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    save_path: str,
    fps: int = 30,
    duration: float = 10,
    rstride: Optional[int] = None,
    cstride: Optional[int] = None,
    dpi: int = 100,
) -> str:
    """
    Rotating wireframe style - lighter, shows structure better.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    X, T = np.meshgrid(x, t)
    stride = max(1, len(x) // 30) if rstride is None else rstride
    cstride = stride if cstride is None else cstride

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(
        X, T, u_xt, rstride=stride, cstride=cstride,
        color='steelblue', alpha=0.7
    )
    ax.set_xlabel('Space (x)')
    ax.set_ylabel('Time (t)')
    ax.set_zlabel('u(x,t)')

    total_frames = int(fps * duration)

    def animate(frame):
        azim = frame * (360 / total_frames)
        ax.view_init(elev=25, azim=azim)
        return []

    anim = animation.FuncAnimation(
        fig, animate, frames=total_frames, interval=1000 / fps
    )
    anim.save(save_path, writer='ffmpeg', fps=fps, dpi=dpi)
    plt.close()
    return save_path


def render_3d_with_noise(
    u_xt: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    save_path: str,
    noise_level: float = 0.05,
    seed: Optional[int] = None,
    **kwargs
) -> str:
    """Add Gaussian noise to the solution before rendering 3D surface."""
    if seed is not None:
        np.random.seed(seed)
    u_noisy = u_xt + noise_level * np.std(u_xt) * np.random.randn(*u_xt.shape)
    return render_3d_surface_video(u_noisy, x, t, save_path, **kwargs)


def render_3d_with_occlusion(
    u_xt: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    save_path: str,
    coverage: float = 0.6,
    seed: Optional[int] = None,
    **kwargs
) -> str:
    """Render 3D surface with partial spatial domain (random columns)."""
    if seed is not None:
        np.random.seed(seed)
    nx = len(x)
    n_keep = max(1, int(nx * coverage))
    visible = np.sort(np.random.choice(nx, n_keep, replace=False))
    u_partial = u_xt[:, visible]
    x_partial = x[visible]
    return render_3d_surface_video(u_partial, x_partial, t, save_path, **kwargs)
