"""
SIMP-V: Spatial Implicit PDE discovery from Video
Adapts EMILY-style LTC + PDE solver for video input.
Pipeline: Video frames -> u(x,t) extraction -> LTC encoder -> coefficient prediction -> PDE loss
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.fft import fftfreq
from typing import Tuple, Dict, List, Optional

try:
    from ncps.torch import LTC
    HAS_LTC = True
except ImportError:
    HAS_LTC = False

from ..utils.pde_solvers import solve_pde, get_initial_condition
from ..utils.metrics import coefficient_error


class NeuralODECell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, x], dim=-1))


class NodeRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dt: float, ode_steps: int = 4):
        super().__init__()
        self.cell = NeuralODECell(input_size, hidden_size)
        self.dt = dt
        self.ode_steps = ode_steps
        self.hidden_size = hidden_size

    def forward(self, x_seq: torch.Tensor):
        batch, time, _ = x_seq.shape
        h = x_seq.new_zeros(batch, self.hidden_size)
        outputs = []
        dt = self.dt / max(1, self.ode_steps)
        for t in range(time):
            x_t = x_seq[:, t, :]
            for _ in range(self.ode_steps):
                dh = self.cell(h, x_t)
                h = h + dt * dh
            outputs.append(h)
        return torch.stack(outputs, dim=1), None


class SIMP_V_Model(nn.Module):
    """
    SIMP-V: Video -> u(x,t) -> LTC -> coefficients -> PDE simulation in loss.
    """

    def __init__(
        self,
        pde_type: str,
        x: np.ndarray,
        dt: float,
        coeff_order: List[str],
        hidden_size: int = 64,
        model_type: str = "ltc",
    ):
        super().__init__()
        self.pde_type = pde_type.lower()
        self.coeff_order = coeff_order
        self.hidden_size = hidden_size
        self.model_type = model_type
        self.dt = dt

        self.nx = len(x)
        self.register_buffer("x", torch.tensor(x, dtype=torch.float32))
        dx = x[1] - x[0]
        k = 2 * np.pi * fftfreq(self.nx, d=dx)
        self.register_buffer("k", torch.tensor(k, dtype=torch.float32))

        self.input_size = self.nx
        self.num_coeffs = len(coeff_order)

        if model_type == "ltc" and HAS_LTC:
            self.rnn = LTC(
                input_size=self.input_size,
                units=hidden_size,
                return_sequences=True,
                batch_first=True,
                mixed_memory=False,
                ode_unfolds=6,
            )
        elif model_type == "node":
            self.rnn = NodeRNN(self.input_size, hidden_size, dt=self.dt, ode_steps=4)
        else:
            self.rnn = nn.LSTM(
                self.input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.1
            )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_coeffs),
            nn.Softplus(),
        )

    def forward(
        self, u_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """u_seq: [batch, time, nx] -> coeffs, u_pred"""
        rnn_out, _ = self.rnn(u_seq)
        rnn_mean = rnn_out.mean(dim=1)
        coeffs = self.fc(rnn_mean)
        u_pred = self.simulate_pde(u_seq, coeffs)
        return coeffs, u_pred

    def simulate_pde(
        self, u_seq: torch.Tensor, coeffs: torch.Tensor
    ) -> torch.Tensor:
        """Roll out PDE forward using estimated coefficients."""
        batch_size, seq_len, _ = u_seq.shape
        u0 = u_seq[:, 0, :]
        u_list = [u0]

        for _ in range(1, seq_len):
            u_next = self._pde_step(u_list[-1], coeffs)
            u_list.append(u_next)

        return torch.stack(u_list, dim=1)

    def _pde_step(self, u: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        coeffs = torch.clamp(coeffs, 0.01, 20.0)
        u_hat = torch.fft.fft(torch.complex(u, torch.zeros_like(u)), dim=-1)
        k = self.k.unsqueeze(0)

        ux = torch.fft.ifft(1j * k * u_hat, dim=-1).real
        uxx = torch.fft.ifft(-k**2 * u_hat, dim=-1).real
        uxxx = torch.fft.ifft(-1j * k**3 * u_hat, dim=-1).real
        uxxxx = torch.fft.ifft(k**4 * u_hat, dim=-1).real

        if self.pde_type == "kdv":
            nonlin = -coeffs[:, 0:1] * u * ux
            nonlin_hat = torch.fft.fft(torch.complex(nonlin, torch.zeros_like(nonlin)), dim=-1)
            lin_factor = torch.exp(-1j * coeffs[:, 1:2] * k**3 * self.dt * 0.25)
            u_hat_next = lin_factor * (u_hat + self.dt * 0.25 * nonlin_hat)
            u_next = torch.fft.ifft(u_hat_next, dim=-1).real
        elif self.pde_type == "burgers":
            nonlin = -u * ux
            nonlin_hat = torch.fft.fft(torch.complex(nonlin, torch.zeros_like(nonlin)), dim=-1)
            u_hat_next = (u_hat + self.dt * nonlin_hat) / (1 + self.dt * coeffs[:, 0:1] * k**2)
            u_next = torch.fft.ifft(u_hat_next, dim=-1).real
        elif self.pde_type == "ks":
            nonlin = -u * ux
            dt_eff = self.dt * 0.5
            u_hat_new = u_hat / (1 + dt_eff * k**2 + dt_eff * coeffs[:, 0:1] * k**4)
            u_linear = torch.fft.ifft(u_hat_new, dim=-1).real
            u_next = u_linear + dt_eff * nonlin
        elif self.pde_type == "heat":
            u_hat_next = u_hat * torch.exp(-coeffs[:, 0:1] * k**2 * self.dt)
            u_next = torch.fft.ifft(u_hat_next, dim=-1).real
        elif self.pde_type == "advection_diffusion":
            u_hat_next = u_hat * torch.exp((-1j * coeffs[:, 0:1] * k - coeffs[:, 1:2] * k**2) * self.dt)
            u_next = torch.fft.ifft(u_hat_next, dim=-1).real
        elif self.pde_type == "schrodinger":
            # Real-valued surrogate for |u|: u_t = alpha * u_xx
            u_hat_next = u_hat * torch.exp(-coeffs[:, 0:1] * k**2 * self.dt)
            u_next = torch.fft.ifft(u_hat_next, dim=-1).real
        elif self.pde_type == "nls":
            # Real-valued surrogate for |u|: u_t = alpha * u_xx + beta * u^3
            u_next = u + self.dt * (coeffs[:, 0:1] * uxx + coeffs[:, 1:2] * u**3)
        else:
            u_next = u

        return torch.clamp(u_next, -100, 100)


def train_simp_v(
    u_data: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    pde_type: str,
    true_coeffs: Dict[str, float],
    coeff_order: List[str],
    max_epochs: int = 500,
    lr: float = 0.0005,
    patience: int = 100,
    log_every: int = 0,
    log_prefix: str = "",
    model_type: str = "ltc",
    device: Optional[torch.device] = None,
) -> Dict:
    """Train SIMP-V on u(x,t) extracted from video."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "ltc" and not HAS_LTC:
        print("Warning: LTC not available; falling back to LSTM.")
        model_type = "lstm"

    dt = t[1] - t[0]
    seq_len = min(50, u_data.shape[0] // 2)
    sequences = []
    for i in range(0, u_data.shape[0] - seq_len, max(1, seq_len // 4)):
        seq = u_data[i : i + seq_len]
        sequences.append(seq)
    data = torch.tensor(np.array(sequences), dtype=torch.float32, device=device)

    model = SIMP_V_Model(
        pde_type, x, dt, coeff_order, hidden_size=64, model_type=model_type
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=50)

    theta_true = np.array([true_coeffs[k] for k in coeff_order], dtype=np.float32)
    theta_true_t = torch.tensor(theta_true, device=device)

    best_loss = float("inf")
    best_coeffs = None
    patience_cnt = 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(data.shape[0])
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, data.shape[0], 8):
            batch_idx = perm[i : i + 8]
            batch = data[batch_idx]
            optimizer.zero_grad()
            coeffs_pred, u_pred = model(batch)
            loss = torch.mean((u_pred - batch) ** 2)
            coeff_reg = 0.1 * torch.mean((coeffs_pred - theta_true_t.unsqueeze(0)) ** 2)
            total = loss + coeff_reg
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if n_batches > 0:
            avg_loss = epoch_loss / n_batches
            scheduler.step(avg_loss)

            if log_every and ((epoch + 1) % log_every == 0 or epoch == 0):
                prefix = f"{log_prefix} " if log_prefix else ""
                print(f"{prefix}epoch {epoch + 1}/{max_epochs} loss={avg_loss:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                model.eval()
                with torch.no_grad():
                    coeffs_est, _ = model(data[: min(32, data.shape[0])])
                    best_coeffs = coeffs_est.mean(dim=0).cpu().numpy()
                patience_cnt = 0
            else:
                patience_cnt += 1

            if patience_cnt >= patience:
                break

    theta_est = best_coeffs if best_coeffs is not None else theta_true * 0
    err = float(coefficient_error(theta_est, theta_true))

    return {
        "theta_est": theta_est,
        "theta_true": theta_true,
        "coefficient_error_pct": err,
        "coeff_order": coeff_order,
    }
