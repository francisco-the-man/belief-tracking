#!/bin/bash
#SBATCH --job-name=lb_svd47
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/phase0_svd47_%j.out
#SBATCH --error=logs/phase0_svd47_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export HF_HOME="/scratch/users/${USER}/hf_cache"
mkdir -p logs additionals
python scripts/compute_svd.py --experiment answer_lookback-payload --layers "[47]" --out_prefix additionals
