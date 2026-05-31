"""
Extract u(x,t) field from rendered video via colormap inversion.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _get_colormap_scale(colormap: str = "viridis") -> Tuple[np.ndarray, np.ndarray]:
    """Get RGB to scalar mapping for colormap inversion."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for colormap inversion")
    cmap = plt.get_cmap(colormap)
    # Sample colormap at 256 levels
    levels = np.linspace(0, 1, 256)
    rgb = cmap(levels)[:, :3]  # (256, 3)
    return rgb, levels


def _rgb_to_scalar(rgb: np.ndarray, cmap_rgb: np.ndarray, cmap_vals: np.ndarray) -> np.ndarray:
    """Map RGB pixel to scalar via nearest neighbor in colormap."""
    rgb_flat = rgb.reshape(-1, 3)
    diff = np.sum((cmap_rgb[np.newaxis, :, :] - rgb_flat[:, np.newaxis, :]) ** 2, axis=2)
    idx = np.argmin(diff, axis=1)
    return cmap_vals[idx].reshape(rgb.shape[:2])


def extract_field_from_video(
    video_path: str,
    x_extent: Optional[Tuple[float, float]] = None,
    t_extent: Optional[Tuple[float, float]] = None,
    colormap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract u(x,t) from video via colormap inversion.
    
    Args:
        video_path: Path to MP4
        x_extent: (x_min, x_max) for spatial axis
        t_extent: (t_min, t_max) for time axis
        colormap: Colormap used during render
        vmin, vmax: Scalar range used during render
    
    Returns:
        x: Spatial grid (nx,)
        t: Time grid (nt,)
        u: Reconstructed field (nt, nx)
    """
    if not HAS_CV2:
        raise ImportError("opencv-python (cv2) required for video extraction")
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    
    if not frames:
        raise ValueError(f"No frames read from {video_path}")
    
    frames = np.array(frames)
    # Video is (nt, height, width, 3); we treat width as x
    nt, height, width, _ = frames.shape
    
    cmap_rgb, cmap_vals = _get_colormap_scale(colormap)
    # Map [0,1] colormap values to [vmin, vmax]
    cmap_vals_scaled = vmin + cmap_vals * (vmax - vmin)
    
    # cv2 frames are uint8 in [0, 255]; the colormap table is in [0, 1].
    # Normalize pixels to [0, 1] so the nearest-color match is on the same scale.
    frames = frames.astype(np.float64) / 255.0

    u_list = []
    for i in range(nt):
        u_frame = _rgb_to_scalar(frames[i], cmap_rgb, cmap_vals_scaled)
        # Transpose so u is (nx,) per frame -> stack to (nt, nx)
        u_list.append(u_frame.mean(axis=0))  # average over height
    
    u = np.array(u_list)
    
    if x_extent is None:
        x_extent = (0, 1)
    if t_extent is None:
        t_extent = (0, nt)
    
    x = np.linspace(x_extent[0], x_extent[1], u.shape[1])
    t = np.linspace(t_extent[0], t_extent[1], u.shape[0])
    
    return x, t, u
