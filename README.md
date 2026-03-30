# VIPER: Video-Informed PDE Extraction and Recovery

**CVPR 2026 Workshop on Women in Computer Vision (WiCV)**

[Paper PDF](WiCV_main.pdf) | [Supplementary](WiCV_suppl.pdf)

> **VIPER** recovers sparse PDE coefficients directly from RGB video of spatiotemporal dynamics — no direct field measurements required.

**Authors:** Farhat Shaikh, Ayan Banerjee, Sandeep Gupta
IMPACT Lab, School of Computing & Augmented Intelligence (SCAI), Arizona State University

---

## Overview

Existing PDE discovery methods (PDE-FIND, SINDy) assume direct access to spatiotemporal field measurements u(x,t) and rely on numerical differentiation, which amplifies noise at rate O((sigma/dx)^k) for k-th order derivatives. In practice, dynamics are often captured as **video** — from thermal cameras to simulation visualizations.

**VIPER** bridges this gap through a four-stage pipeline:

1. **Video-to-Field Extraction** — Colormap inversion recovers u(x,t) from RGB frames
2. **Method-of-Lines Discretization** — Converts PDE to coupled ODEs on a spatial grid
3. **LTC Encoder** — A Liquid Time-Constant neural network estimates coefficients, with hidden states modeling unobserved spatial modes
4. **Differentiable PDE Solver** — Validates coefficients by forward integration (ETDRK4 / split-step Fourier), converting the noise-amplifying differentiation problem into a noise-attenuating integration problem

<p align="center">
  <img src="untitled folder/high-level.pdf" alt="VIPER pipeline" width="700"/>
</p>

## Key Results

### Clean Video (Table 2)

| PDE | VIPER (%) | PDE-FIND (%) | Improvement |
|-----|-----------|-------------|-------------|
| KdV | **5.27** | 534.70 | 101x |
| Burgers | **0.67** | 1.95 | 2.9x |
| KS | **2.30** | 2.31 | 1.0x |
| Schrodinger | **6.22** | 92.95 | 14.9x |
| NLS | **28.56** | 100.00 | 3.5x |

### Noisy Video — 5% Gaussian noise (Table 3)

| PDE | VIPER (%) | PDE-FIND (%) | Improvement |
|-----|-----------|-------------|-------------|
| KdV | 87.30 | 100.00 | 1.15x |
| Burgers | **10.97** | 100.00 | 9.1x |
| KS | **3.47** | 99.95 | 28.8x |
| Schrodinger | **12.51** | 99.02 | 7.9x |
| NLS | **23.36** | 99.82 | 4.3x |

### Implicit Dynamics — Partial Spatial Observability (Table 4)

VIPER recovers coefficients from as little as **20% spatial coverage**, a regime where derivative-based methods fundamentally cannot operate.

| PDE | 20% Cov. | 40% Cov. | 60% Cov. | 80% Cov. |
|-----|----------|----------|----------|----------|
| KS | 3.19% | 1.87% | **0.52%** | 0.89% |
| Schrodinger | **0.04%** | 0.12% | 0.20% | 0.35% |
| Burgers | 19.27% | 8.43% | 2.15% | **0.63%** |

*Coefficient error (%) — lower is better. PDE-FIND fails (>90% error) across all coverage levels.*

## PDEs Evaluated

| PDE | Equation | Dynamics |
|-----|----------|----------|
| KdV | u_t + 6u·u_x + u_xxx = 0 | Dispersive solitons |
| Burgers | u_t + u·u_x = 0.1·u_xx | Viscous shocks |
| KS | u_t + u·u_x + u_xx + u_xxxx = 0 | Spatiotemporal chaos |
| Schrodinger | i·u_t + 0.5·u_xx = 0 | Linear dispersion |
| NLS | i·u_t + 0.5·u_xx + \|u\|^2·u = 0 | Nonlinear optics |

## Installation

```bash
git clone https://github.com/<your-username>/VIPER.git
cd VIPER
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch >= 1.10
- NumPy, SciPy, Matplotlib
- [ncps](https://github.com/mlech26l/ncps) >= 0.0.4 (LTC networks)
- OpenCV (optional, for video I/O)

## Usage

### Full Pipeline

```bash
# 1. Generate PDE simulations
python experiments/generate_simulations.py --all

# 2. Render to video
python experiments/render_videos.py --all

# 3. Run experiments
python experiments/exp1_clean.py                          # Table 2: Clean video
python experiments/exp2_noisy.py --noise gaussian --level 0.05  # Table 3: Noisy video
python experiments/exp3_implicit.py --seeds 3             # Table 4: Partial observability

# 4. Ablation studies
python experiments/exp4_ablation.py --ablation hidden --pde burgers

# 5. Generate result tables
python experiments/generate_results.py
```

Or run everything at once:

```bash
bash run_experiments.sh
```

## Project Structure

```
VIPER/
├── simp_video/
│   ├── models/
│   │   ├── simp_v.py                # VIPER model (LTC + differentiable solver)
│   │   └── baselines/
│   │       ├── pde_find.py          # PDE-FIND with video frontend
│   │       ├── pinn_inverse.py      # Inverse PINN baseline
│   │       ├── deepmod_video.py     # DeepMoD baseline
│   │       └── delfys_wrapper.py    # Delfys baseline
│   └── utils/
│       ├── pde_solvers.py           # Spectral PDE solvers (ETDRK4, split-step)
│       ├── video_render.py          # u(x,t) → MP4 via colormap
│       ├── video_extract.py         # MP4 → u(x,t) via colormap inversion
│       └── metrics.py               # Coefficient error metric
├── experiments/
│   ├── exp1_clean.py                # Experiment 1: Clean video
│   ├── exp2_noisy.py                # Experiment 2: Noisy video
│   ├── exp3_implicit.py             # Experiment 3: Partial observability
│   ├── exp4_ablation.py             # Experiment 4: Ablations
│   ├── generate_simulations.py      # PDE data generation
│   ├── render_videos.py             # Video rendering
│   └── generate_results.py          # Tables and figures
├── configs/
│   └── experiment_configs.yaml      # All hyperparameters
├── results/                         # Output JSON files
├── requirements.txt
└── run_experiments.sh
```

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| LTC hidden units | 64 |
| Learning rate | 5 x 10^-4 |
| Optimizer | Adam (beta1=0.9, beta2=0.999) |
| Max epochs | 500 |
| Early stopping patience | 100 |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 50) |
| Sparsity lambda | 10^-3 |
| Spatial grid n_x | 128 |
| Video resolution | 256 x 256 |
| Random seeds | {42, 123, 456, 789, 1011} |

All experiments were conducted on CPU. Training time ranges from 5 minutes (KS) to 45 minutes (NLS).

## Acknowledgments

This project is partially funded by DARPA AMP-N6600120C4020, DARPA FIRE-P000050426, NSF FDT-Biotech grant (2436801), NIH R21 grant (1R21HL175632).

## Citation

```bibtex
@inproceedings{shaikh2026viper,
  title={VIPER: Video-Informed PDE Extraction and Recovery},
  author={Shaikh, Farhat and Banerjee, Ayan and Gupta, Sandeep},
  booktitle={CVPR 2026 Workshop on Women in Computer Vision (WiCV)},
  year={2026}
}
```

## License

This project is for academic research purposes. Please contact the authors for commercial use.
