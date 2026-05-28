"""
PDE-FIND baseline with video frontend.
Video -> Colormap inversion -> u(x,t) -> Finite differences -> STLS regression
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from ...utils.metrics import coefficient_error


def finite_differences(u: np.ndarray, dx: float, dt: float) -> Tuple[np.ndarray, ...]:
    """Compute u_t, u_x, u_xx, u_xxx, u_xxxx using central/forward differences."""
    nt, nx = u.shape
    u_t = np.zeros_like(u)
    u_t[1:-1] = (u[2:] - u[:-2]) / (2 * dt)
    u_t[0] = (u[1] - u[0]) / dt
    u_t[-1] = (u[-1] - u[-2]) / dt

    u_x = np.zeros_like(u)
    u_x[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    u_x[:, 0] = (u[:, 1] - u[:, 0]) / dx
    u_x[:, -1] = (u[:, -1] - u[:, -2]) / dx

    u_xx = np.zeros_like(u)
    u_xx[:, 1:-1] = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / (dx ** 2)
    u_xx[:, 0] = u_xx[:, 1]
    u_xx[:, -1] = u_xx[:, -2]

    u_xxx = np.zeros_like(u)
    u_xxx[:, 2:-2] = (u[:, 4:] - 2 * u[:, 3:-1] + 2 * u[:, 1:-3] - u[:, :-4]) / (2 * dx ** 3)
    u_xxx[:, :2] = u_xxx[:, 2:3]
    u_xxx[:, -2:] = u_xxx[:, -3:-2]

    u_xxxx = np.zeros_like(u)
    u_xxxx[:, 2:-2] = (
        u[:, 4:] - 4 * u[:, 3:-1] + 6 * u[:, 2:-2] - 4 * u[:, 1:-3] + u[:, :-4]
    ) / (dx ** 4)
    u_xxxx[:, :2] = u_xxxx[:, 2:3]
    u_xxxx[:, -2:] = u_xxxx[:, -3:-2]

    return u_t, u_x, u_xx, u_xxx, u_xxxx


def build_library(
    u: np.ndarray, u_t: np.ndarray, u_x: np.ndarray, u_xx: np.ndarray,
    u_xxx: np.ndarray, u_xxxx: np.ndarray, pde_type: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build PDE term library (flat) and Ut for given PDE type."""
    def flat(*arrs):
        return [a.ravel() for a in arrs]

    if pde_type == "kdv":
        Theta = np.column_stack(flat(u * u_x, u_xxx))
        names = ["alpha", "beta"]
    elif pde_type == "burgers":
        Theta = np.column_stack(flat(u * u_x, u_xx))
        names = ["nu_neg_ux", "nu"]
    elif pde_type == "ks":
        Theta = np.column_stack(flat(u * u_x, u_xx, u_xxxx))
        names = ["c1", "c2", "nu"]
    elif pde_type == "heat":
        Theta = np.column_stack(flat(u_xx))
        names = ["D"]
    elif pde_type == "advection_diffusion":
        Theta = np.column_stack(flat(u_x, u_xx))
        names = ["c", "D"]
    elif pde_type == "schrodinger":
        Theta = np.column_stack(flat(u_xx))
        names = ["alpha"]
    elif pde_type == "nls":
        Theta = np.column_stack(flat(u_xx, u**3))
        names = ["alpha", "beta"]
    else:
        Theta = np.column_stack(flat(u, u_x, u_xx, u * u_x, u_xxx))
        names = ["u", "ux", "uxx", "u_ux", "uxxx"]

    Ut = u_t.ravel()
    return Theta, Ut, names


def stls_regression(Theta: np.ndarray, Ut: np.ndarray, lam: float = 1e-5, max_iter: int = 10) -> np.ndarray:
    """Sequential Threshold Least Squares for sparse coefficient recovery."""
    xi = np.linalg.lstsq(Theta, Ut, rcond=None)[0]
    for _ in range(max_iter):
        small = np.abs(xi) < lam
        xi[small] = 0
        if np.all(small):
            break
        xi[~small] = np.linalg.lstsq(Theta[:, ~small], Ut, rcond=None)[0]
    return xi


def pde_find_video(
    u: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    pde_type: str,
    true_coeffs: Dict[str, float],
    coeff_order: List[str],
) -> Dict:
    """
    Run PDE-FIND on u(x,t) (extracted from video).
    Returns estimated coefficients and error.
    """
    dx = x[1] - x[0]
    dt_val = t[1] - t[0]
    u_t, u_x, u_xx, u_xxx, u_xxxx = finite_differences(u, dx, dt_val)
    Theta, Ut, names = build_library(u, u_t, u_x, u_xx, u_xxx, u_xxxx, pde_type)
    xi = stls_regression(Theta, Ut, lam=1e-3)

    # Map xi to coeff_order
    coeff_map = {
        "kdv": {"alpha": 0, "beta": 1},
        "burgers": {"nu": 1},
        "ks": {"nu": 2},
        "heat": {"D": 0},
        "advection_diffusion": {"c": 0, "D": 1},
        "schrodinger": {"alpha": 0},
        "nls": {"alpha": 0, "beta": 1},
    }
    theta_est = np.zeros(len(coeff_order))
    for i, name in enumerate(coeff_order):
        if pde_type == "burgers" and name == "nu":
            theta_est[i] = abs(xi[1])
        elif pde_type == "kdv":
            theta_est[i] = abs(xi[coeff_map[pde_type].get(name, i)])
        else:
            idx = coeff_map.get(pde_type, {}).get(name, i)
            if idx < len(xi):
                theta_est[i] = abs(xi[idx])

    theta_true = np.array([true_coeffs[k] for k in coeff_order])
    err = float(coefficient_error(theta_est, theta_true))

    return {
        "theta_est": theta_est,
        "theta_true": theta_true,
        "coefficient_error_pct": err,
        "coeff_order": coeff_order,
    }
