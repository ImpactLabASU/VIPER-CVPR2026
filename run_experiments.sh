#!/bin/bash
# SIMP-V Experiment Execution Script

set -e
cd "$(dirname "$0")"

echo "=== Step 1: Generate simulations ==="
python experiments/generate_simulations.py --all

echo "=== Step 2: Render videos ==="
python experiments/render_videos.py --all

echo "=== Step 3: Run experiments ==="
python experiments/exp1_clean.py
python experiments/exp2_noisy.py --noise gaussian --level 0.05
python experiments/exp3_implicit.py

echo "=== Step 4: Ablation ==="
python experiments/exp4_ablation.py --ablation hidden --pde burgers

echo "=== Step 5: Generate tables ==="
python experiments/generate_results.py

echo "Done. Results in results/"
