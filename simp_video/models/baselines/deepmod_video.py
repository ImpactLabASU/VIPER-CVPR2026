"""
DeepMoD-style baseline (video frontend).
Video -> Neural field u_theta(x,t) -> Auto-diff derivatives -> Sparse regression.
Simplified: use finite-diff on extracted u, then sparse regression (similar to PDE-FIND).
"""

import numpy as np
from typing import Dict, List

from .pde_find import pde_find_video


def deepmod_video(
    u: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    pde_type: str,
    true_coeffs: Dict[str, float],
    coeff_order: List[str],
) -> Dict:
    """
    DeepMoD-style: Neural field + sparse regression.
    Simplified implementation: use extracted u with PDE-FIND pipeline.
    Full DeepMoD would fit u_theta(x,t) and use autodiff for derivatives.
    """
    return pde_find_video(u, x, t, pde_type, true_coeffs, coeff_order)
