"""
PDE solvers for SIMP-V experiments.
Generates u(x,t) arrays for KdV, Burgers, KS, Schrodinger, and NLS.
Uses spectral methods (ETDRK4) where appropriate.
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.integrate import solve_ivp
from typing import Tuple, Dict, Optional, List


def get_initial_condition(
    x: np.ndarray, 
    ic_type: str, 
    pde_type: str,
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate initial condition u0(x)."""
    if seed is not None:
        np.random.seed(seed)
    
    L = x[-1] - x[0]
    
    if ic_type == "sech_soliton" or pde_type == "kdv":
        c = 0.5
        u0 = 2 * (c / 2) * (1.0 / np.cosh(np.clip(np.sqrt(c) / 2 * x, -30, 30))) ** 2
        
    elif ic_type == "sinusoidal" or pde_type == "burgers":
        u0 = -np.sin(np.pi * x / (L / 2))
        
    elif ic_type == "cos_perturbed" or pde_type == "ks":
        u0 = np.cos(2 * np.pi * x / L) * (1 + 0.1 * np.sin(2 * np.pi * x / L))
        
    elif ic_type == "gaussian" or pde_type == "heat":
        u0 = np.exp(-x**2)
        
    elif ic_type == "gaussian_shifted" or pde_type == "advection_diffusion":
        u0 = np.exp(-(x - 2) ** 2)
    elif ic_type == "sech" or pde_type == "nls":
        u0 = 1.0 / np.cosh(x)
    elif ic_type == "gaussian_phase" or pde_type == "schrodinger":
        u0 = np.exp(-x**2)
        
    else:
        u0 = np.exp(-x**2)
    
    return u0.astype(np.float64)


def solve_kdv(
    x: np.ndarray, 
    t: np.ndarray, 
    alpha: float = 6.0, 
    beta: float = 1.0,
    u0: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """Solve KdV: u_t + alpha*u*u_x + beta*u_xxx = 0 using ETDRK4."""
    N, dt = len(x), t[1] - t[0]
    dx = x[1] - x[0]
    k = 2 * np.pi * fftfreq(N, d=dx)
    
    if u0 is None:
        u0 = get_initial_condition(x, "sech_soliton", "kdv", seed)
    
    u_all = np.zeros((len(t), N), dtype=np.float64)
    u_all[0] = u0
    
    Lin = -1j * beta * k**3
    Lin_dt = np.clip(Lin * dt, -100, 100)
    Lin_dt2 = np.clip(Lin * dt / 2, -100, 100)
    E, E2 = np.exp(Lin_dt).astype(complex), np.exp(Lin_dt2).astype(complex)
    M = 32
    r = 15 * np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * Lin[:, np.newaxis] + r[np.newaxis, :]
    LR = np.clip(LR, -100, 100)
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / (LR + 1e-10), axis=1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3*LR + LR**2)) / (LR**3 + 1e-10), axis=1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / (LR**3 + 1e-10), axis=1))
    f3 = dt * np.real(np.mean((-4 - 3*LR - LR**2 + np.exp(LR) * (4 - LR)) / (LR**3 + 1e-10), axis=1))
    
    def nonlinear(u_hat):
        u = np.real(ifft(u_hat))
        u = np.clip(u, -100, 100)
        ux = np.real(ifft(1j * k * u_hat))
        ux = np.clip(ux, -100, 100)
        return -alpha * fft(u * ux)
    
    u_hat = fft(u0).astype(complex)
    for n in range(1, len(t)):
        Nu = nonlinear(u_hat)
        a = E2 * u_hat + Q * Nu
        Na = nonlinear(a)
        b = E2 * u_hat + Q * Na
        Nb = nonlinear(b)
        c = E2 * a + Q * (2 * Nb - Nu)
        Nc = nonlinear(c)
        u_hat = E * u_hat + Nu * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        u_real = np.real(ifft(u_hat))
        u_real = np.clip(u_real, -100, 100)
        u_all[n] = u_real
        u_hat = fft(u_real)
    
    return u_all


def solve_burgers(
    x: np.ndarray, 
    t: np.ndarray, 
    nu: float = 0.1,
    u0: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """Solve Burgers: u_t + u*u_x = nu*u_xx using spectral method."""
    N, dt = len(x), t[1] - t[0]
    dx = x[1] - x[0]
    k = 2 * np.pi * fftfreq(N, d=dx)
    
    if u0 is None:
        L = x[-1] - x[0]
        u0 = -np.sin(np.pi * x / (L / 2))
    
    u_all = np.zeros((len(t), N))
    u_all[0] = u0
    u_hat = fft(u0).astype(complex)
    
    for n in range(1, len(t)):
        u = np.real(ifft(u_hat))
        ux = np.real(ifft(1j * k * u_hat))
        rhs = fft(-u * ux) - nu * k**2 * u_hat
        u_hat = u_hat + dt * rhs
        u_all[n] = np.real(ifft(u_hat))
    
    return u_all


def solve_ks(
    x: np.ndarray, 
    t: np.ndarray, 
    nu: float = 1.0,
    u0: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """Solve KS: u_t + u*u_x + u_xx + nu*u_xxxx = 0 using ETDRK4."""
    N, dt = len(x), t[1] - t[0]
    dx = x[1] - x[0]
    k = 2 * np.pi * fftfreq(N, d=dx)
    Lin = -k**2 - nu * k**4
    
    if u0 is None:
        L = x[-1] - x[0]
        u0 = np.cos(2 * np.pi * x / L) * (1 + 0.1 * np.sin(2 * np.pi * x / L))
    
    u_all = np.zeros((len(t), N))
    u_all[0] = u0
    
    E = np.exp(Lin * dt)
    E2 = np.exp(Lin * dt / 2)
    M = 16
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * Lin[:, np.newaxis] + r[np.newaxis, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / (LR + 1e-10), axis=1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3*LR + LR**2)) / (LR**3 + 1e-10), axis=1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / (LR**3 + 1e-10), axis=1))
    f3 = dt * np.real(np.mean((-4 - 3*LR - LR**2 + np.exp(LR) * (4 - LR)) / (LR**3 + 1e-10), axis=1))
    
    def nonlinear(u_hat):
        u = np.real(ifft(u_hat))
        ux = np.real(ifft(1j * k * u_hat))
        return -fft(u * ux)
    
    u_hat = fft(u0).astype(complex)
    for n in range(1, len(t)):
        Nu = nonlinear(u_hat)
        a = E2 * u_hat + Q * Nu
        Na = nonlinear(a)
        b = E2 * u_hat + Q * Na
        Nb = nonlinear(b)
        c = E2 * a + Q * (2 * Nb - Nu)
        Nc = nonlinear(c)
        u_hat = E * u_hat + Nu * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        u_all[n] = np.real(ifft(u_hat))
    
    return u_all


def solve_heat(
    x: np.ndarray, 
    t: np.ndarray, 
    D: float = 0.5,
    u0: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """Solve Heat: u_t = D*u_xx using spectral method."""
    N, dt = len(x), t[1] - t[0]
    dx = x[1] - x[0]
    k = 2 * np.pi * fftfreq(N, d=dx)
    
    if u0 is None:
        u0 = np.exp(-x**2)
    
    u_hat = fft(u0)
    u_all = np.zeros((len(t), N))
    u_all[0] = u0
    
    for n in range(1, len(t)):
        u_hat = u_hat * np.exp(-D * k**2 * dt)
        u_all[n] = np.real(ifft(u_hat))
    
    return u_all


def solve_advection_diffusion(
    x: np.ndarray, 
    t: np.ndarray, 
    c: float = 1.0, 
    D: float = 0.1,
    u0: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """Solve Advection-Diffusion: u_t + c*u_x = D*u_xx using spectral method."""
    N, dt = len(x), t[1] - t[0]
    dx = x[1] - x[0]
    k = 2 * np.pi * fftfreq(N, d=dx)
    
    if u0 is None:
        u0 = np.exp(-(x - 2)**2)
    
    u_hat = fft(u0)
    u_all = np.zeros((len(t), N))
    u_all[0] = u0
    
    for n in range(1, len(t)):
        u_hat = u_hat * np.exp((-1j * c * k - D * k**2) * dt)
        u_all[n] = np.real(ifft(u_hat))
    
    return u_all


def solve_schrodinger(
    x: np.ndarray,
    t: np.ndarray,
    alpha: float = 0.5,
    u0: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Solve linear Schrodinger: i u_t + alpha*u_xx = 0 using spectral method.
    Returns |u| as a real-valued field for video pipelines.
    """
    N, dt = len(x), t[1] - t[0]
    dx = x[1] - x[0]
    k = 2 * np.pi * fftfreq(N, d=dx)

    if u0 is None:
        u0 = np.exp(-x**2)
    if not np.iscomplexobj(u0):
        # Gaussian wavepacket with phase
        u0 = u0 * np.exp(1j * 0.5 * x)
    else:
        u0 = u0.astype(complex)

    u_hat = fft(u0)
    u_all = np.zeros((len(t), N), dtype=np.float64)

    for n in range(len(t)):
        if n > 0:
            u_hat = u_hat * np.exp(-1j * alpha * k**2 * dt)
        u = ifft(u_hat)
        u_all[n] = np.abs(u)

    return u_all


def solve_nls(
    x: np.ndarray,
    t: np.ndarray,
    alpha: float = 0.5,
    beta: float = 1.0,
    u0: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Solve focusing NLS: i u_t + alpha*u_xx + beta*|u|^2 u = 0 using split-step Fourier.
    Returns |u| as a real-valued field for video pipelines.
    """
    N, dt = len(x), t[1] - t[0]
    dx = x[1] - x[0]
    k = 2 * np.pi * fftfreq(N, d=dx)

    if u0 is None:
        u0 = 1.0 / np.cosh(x)
    u = u0.astype(complex)

    u_all = np.zeros((len(t), N), dtype=np.float64)

    for n in range(len(t)):
        u_all[n] = np.abs(u)
        # Nonlinear half step
        u = u * np.exp(1j * beta * np.abs(u) ** 2 * dt / 2)
        # Linear full step
        u_hat = fft(u)
        u_hat = u_hat * np.exp(-1j * alpha * k**2 * dt)
        u = ifft(u_hat)
        # Nonlinear half step
        u = u * np.exp(1j * beta * np.abs(u) ** 2 * dt / 2)

    return u_all


def solve_pde(
    pde_type: str,
    x: np.ndarray,
    t: np.ndarray,
    true_coeffs: Dict[str, float],
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Solve PDE given type and coefficients.
    
    Returns:
        u: shape (nt, nx)
    """
    pde_type = pde_type.lower()
    if pde_type == "kdv":
        u0 = get_initial_condition(x, "sech_soliton", pde_type, seed)
        return solve_kdv(x, t, alpha=true_coeffs.get("alpha", 6.0), beta=true_coeffs.get("beta", 1.0), u0=u0, seed=seed)
    elif pde_type == "burgers":
        u0 = get_initial_condition(x, "sinusoidal", pde_type, seed)
        return solve_burgers(x, t, nu=true_coeffs.get("nu", 0.1), u0=u0, seed=seed)
    elif pde_type == "ks":
        u0 = get_initial_condition(x, "cos_perturbed", pde_type, seed)
        return solve_ks(x, t, nu=true_coeffs.get("nu", 1.0), u0=u0, seed=seed)
    elif pde_type == "heat":
        u0 = get_initial_condition(x, "gaussian", pde_type, seed)
        return solve_heat(x, t, D=true_coeffs.get("D", 0.5), u0=u0, seed=seed)
    elif pde_type == "advection_diffusion":
        u0 = get_initial_condition(x, "gaussian_shifted", pde_type, seed)
        return solve_advection_diffusion(x, t, c=true_coeffs.get("c", 1.0), D=true_coeffs.get("D", 0.1), u0=u0, seed=seed)
    elif pde_type == "schrodinger":
        u0 = get_initial_condition(x, "gaussian_phase", pde_type, seed)
        return solve_schrodinger(x, t, alpha=true_coeffs.get("alpha", 0.5), u0=u0, seed=seed)
    elif pde_type == "nls":
        u0 = get_initial_condition(x, "sech", pde_type, seed)
        return solve_nls(x, t, alpha=true_coeffs.get("alpha", 0.5), beta=true_coeffs.get("beta", 1.0), u0=u0, seed=seed)
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")
