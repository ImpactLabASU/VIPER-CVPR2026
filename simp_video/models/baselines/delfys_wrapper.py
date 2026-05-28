"""
Delfys-style baseline (placeholder).
Video -> Delfys encoder -> latent trajectory -> PDE-FIND on latent.
Requires external Delfys model; falls back to identity + PDE-FIND on raw u.
"""

import numpy as np
from typing import Dict, List, Optional

from .pde_find import pde_find_video
from ...utils.metrics import coefficient_error


def delfys_wrapper_video(
    u: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    pde_type: str,
    true_coeffs: Dict[str, float],
    coeff_order: List[str],
    delfys_model_path: Optional[str] = None,
) -> Dict:
    """
    Delfys + PDE-FIND: Encode video to latent, run PDE-FIND on latent.
    If delfys_model_path is None, we use raw u (same as PDE-FIND baseline).
    """
    # Placeholder: no Delfys model available, use raw u
    u_latent = u
    return pde_find_video(u_latent, x, t, pde_type, true_coeffs, coeff_order)
