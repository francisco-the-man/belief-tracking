#!/bin/bash
#SBATCH --job-name=lb_svd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/phase0_svd_%x_%j.out
#SBATCH --error=logs/phase0_svd_%x_%j.err

# Phase 0 step 1: compute SVD bases for Answer Lookback (pointer + payload).
#
# Usage (from repo root, after `uv sync`):
#   sbatch scripts/farmshare/phase0_svd.slurm
#
# Or to compute a single experiment:
#   sbatch --export=EXPERIMENT=answer_lookback-pointer scripts/farmshare/phase0_svd.slurm
#
# Both pointer and payload share the same vector_type ("last_token") and the
# same intervention position ([-1]), but the dataset constructors differ, so
# we compute the basis for each independently in case the activation
# distribution at the ':' token differs across counterfactual schemes.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# Activate the project venv. Adjust if your venv is elsewhere.
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv/bin/activate not found in $(pwd)" >&2
    echo "Run 'uv sync' first." >&2
    exit 1
fi

# Redirect HuggingFace cache to scratch — SLURM jobs don't source ~/.bashrc.
export HF_HOME="/scratch/users/${USER}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}" logs additionals

EXPERIMENT="${EXPERIMENT:-both}"
MODEL_KEY="${MODEL_KEY:-Qwen/Qwen2.5-14B-Instruct}"
N_SAMPLES="${N_SAMPLES:-80}"
BATCH_SIZE="${BATCH_SIZE:-4}"

echo "----------------------------------------"
echo "Node:        $(hostname)"
echo "Date:        $(date)"
echo "Experiment:  ${EXPERIMENT}"
echo "Model:       ${MODEL_KEY}"
echo "GPU info:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true)"
echo "----------------------------------------"

run_one() {
    local exp="$1"
    echo ""
    echo "=== Computing SVD for ${exp} ==="
    python scripts/compute_svd.py \
        --experiment "${exp}" \
        --model_key "${MODEL_KEY}" \
        --n_samples "${N_SAMPLES}" \
        --batch_size "${BATCH_SIZE}" \
        --out_prefix additionals
}

if [[ "${EXPERIMENT}" == "both" ]]; then
    run_one answer_lookback-pointer
    run_one answer_lookback-payload
else
    run_one "${EXPERIMENT}"
fi

echo ""
echo "Done at $(date)"
