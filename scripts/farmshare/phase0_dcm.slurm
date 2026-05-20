#!/bin/bash
#SBATCH --job-name=lb_dcm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase0_dcm_%x_%j.out
#SBATCH --error=logs/phase0_dcm_%x_%j.err

# Phase 0 step 2: train DCM masks and run validation across the requested
# layer range for one Answer Lookback experiment.
#
# Expects phase0_svd.slurm to have completed and populated
# additionals/svd/causalToM/last_token/singular_vecs/<layer>.pt
#
# Usage:
#   sbatch --export=EXPERIMENT=answer_lookback-pointer scripts/farmshare/phase0_dcm.slurm
#   sbatch --export=EXPERIMENT=answer_lookback-payload scripts/farmshare/phase0_dcm.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv/bin/activate not found" >&2
    exit 1
fi

# Redirect HuggingFace cache to scratch — SLURM jobs don't source ~/.bashrc.
export HF_HOME="/scratch/users/${USER}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}"   # legacy compat
mkdir -p "${HF_HOME}" logs

EXPERIMENT="${EXPERIMENT:?must set EXPERIMENT to answer_lookback-pointer or answer_lookback-payload}"
MODEL_KEY="${MODEL_KEY:-Qwen/Qwen2.5-14B-Instruct}"
TRAIN_SIZE="${TRAIN_SIZE:-80}"
VALID_SIZE="${VALID_SIZE:-80}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-0.1}"
LAMB="${LAMB:-0.1}"
N_EPOCHS="${N_EPOCHS:-1}"
# Save to our_repro to avoid overwriting the paper authors' reference Qwen-14B
# results that ship with the fork (under results/causalToM_novis/).
SAVE_PATH="${SAVE_PATH:-results/our_repro/causalToM_novis/}"

echo "----------------------------------------"
echo "Node:        $(hostname)"
echo "Date:        $(date)"
echo "Experiment:  ${EXPERIMENT}"
echo "Model:       ${MODEL_KEY}"
echo "Save path:   ${SAVE_PATH}"
echo "GPU info:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true)"
echo "----------------------------------------"

python scripts/patching_scripts/run_single_layer_patching_exps.py \
    --experiment "${EXPERIMENT}" \
    --model_key "${MODEL_KEY}" \
    --train_size "${TRAIN_SIZE}" \
    --validation_size "${VALID_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LR}" \
    --lamb "${LAMB}" \
    --n_epochs "${N_EPOCHS}" \
    --save_path "${SAVE_PATH}" \
    --skip_subspace_patching False \
    --verbose True

echo ""
echo "Done at $(date)"
