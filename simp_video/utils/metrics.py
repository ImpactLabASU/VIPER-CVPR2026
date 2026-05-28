"""
Metrics for SIMP-V experiments.
"""

import numpy as np
from typing import Union, Dict


def coefficient_error(theta_est: np.ndarray, theta_true: np.ndarray) -> float:
    """Relative coefficient error (%)."""
    norm_true = np.linalg.norm(theta_true)
    if norm_true < 1e-10:
        return np.linalg.norm(theta_est - theta_true) * 100.0
    return np.linalg.norm(theta_est - theta_true) / norm_true * 100.0


def trajectory_rmse(u_pred: np.ndarray, u_true: np.ndarray) -> float:
    """RMSE of forward simulation (field comparison)."""
    return float(np.sqrt(np.mean((u_pred - u_true) ** 2)))


def structure_recovery(
    theta_est: np.ndarray, 
    theta_true: np.ndarray, 
    tol: float = 0.1
) -> float:
    """Fraction of correctly identified active/inactive terms."""
    true_active = np.abs(theta_true) > tol
    est_active = np.abs(theta_est) > tol
    return float(np.mean(true_active == est_active))


def noise_amplification_factor(
    error_noisy: float, 
    error_clean: float,
    eps: float = 1e-10
) -> float:
    """How much does noise degrade performance?"""
    if error_clean < eps:
        return 1.0
    return error_noisy / error_clean


def coefficient_mae(theta_est: np.ndarray, theta_true: np.ndarray) -> np.ndarray:
    """Per-coefficient mean absolute error."""
    return np.abs(theta_est - theta_true)
