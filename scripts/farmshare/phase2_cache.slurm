#!/bin/bash
#SBATCH --job-name=lb_cache
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/phase2_cache_%x_%j.out
#SBATCH --error=logs/phase2_cache_%x_%j.err

# Phase 2 step 1: cache residual stream activations across (layer, token)
# for a pool of CausalToM prompts, so we can train linear probes off-GPU.
#
# Usage:
#   sbatch scripts/farmshare/phase2_cache.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv/bin/activate not found" >&2
    exit 1
fi

mkdir -p logs additionals

MODEL_KEY="${MODEL_KEY:-Qwen/Qwen2.5-14B-Instruct}"
N_SAMPLES="${N_SAMPLES:-500}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUT_DIR="${OUT_DIR:-additionals/cached_acts/causalToM_novis}"

echo "----------------------------------------"
echo "Node:        $(hostname)"
echo "Date:        $(date)"
echo "Model:       ${MODEL_KEY}"
echo "n_samples:   ${N_SAMPLES}"
echo "out_dir:     ${OUT_DIR}"
echo "GPU info:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true)"
echo "----------------------------------------"

python scripts/cache_activations.py \
    --model_key "${MODEL_KEY}" \
    --n_samples "${N_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --out_dir "${OUT_DIR}"

echo ""
echo "Done at $(date)"
