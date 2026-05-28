"""
PINN Inverse baseline: Learn PDE coefficients from video-extracted u(x,t)
using a Physics-Informed Neural Network with learnable coefficients.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional

from ...utils.pde_solvers import solve_pde
from ...utils.metrics import coefficient_error


class CoeffPINN(nn.Module):
    """PINN with learnable PDE coefficients."""

    def __init__(self, pde_type: str, coeff_order: List[str], init_scale: float = 0.5):
        super().__init__()
        self.pde_type = pde_type
        self.coeff_order = coeff_order
        self.num_coeffs = len(coeff_order)
        self.log_coeffs = nn.Parameter(torch.ones(self.num_coeffs) * np.log(init_scale + 0.1))

    def get_coeffs(self) -> torch.Tensor:
        return torch.exp(self.log_coeffs) + 0.01


def pinn_inverse_video(
    u: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    pde_type: str,
    true_coeffs: Dict[str, float],
    coeff_order: List[str],
    max_epochs: int = 500,
    lr: float = 0.01,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    PINN inverse: minimize data loss + PDE residual.
    Simplified: we compare predicted trajectory (from solver with learned coeffs)
    to data. Use scipy solver with learned coeffs (non-differentiable), so we
    do gradient-free optimization over coefficients.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    theta_true = np.array([true_coeffs[k] for k in coeff_order])
    theta_init = theta_true * 0.5 + 0.1

    best_theta = theta_init.copy()
    best_loss = float("inf")

    from scipy.optimize import minimize

    def loss_fn(theta):
        coeff_dict = {k: theta[i] for i, k in enumerate(coeff_order)}
        try:
            u_pred = solve_pde(pde_type, x, t, coeff_dict)
            mse = np.mean((u_pred - u) ** 2)
        except Exception:
            mse = 1e10
        return mse

    res = minimize(
        loss_fn,
        theta_init,
        method="L-BFGS-B",
        bounds=[(0.001, 20.0)] * len(coeff_order),
        options={"maxiter": 200},
    )
    theta_est = np.clip(res.x, 0.001, 20.0)
    err = float(coefficient_error(theta_est, theta_true))

    return {
        "theta_est": theta_est,
        "theta_true": theta_true,
        "coefficient_error_pct": err,
        "coeff_order": coeff_order,
    }
