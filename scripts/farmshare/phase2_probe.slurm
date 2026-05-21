#!/bin/bash
#SBATCH --job-name=lb_probe
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/phase2_probe_%x_%j.out
#SBATCH --error=logs/phase2_probe_%x_%j.err

# Phase 2 step 2: train linear probes on cached activations.
# CPU-only job; doesn't need a GPU. Runs on a dedicated compute node so
# it doesn't contend with other users on the login node.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv/bin/activate not found" >&2
    exit 1
fi

mkdir -p logs results/probes

CACHE_DIR="${CACHE_DIR:-additionals/cached_acts/causalToM_novis}"
OUT_PATH="${OUT_PATH:-results/probes/causalToM_novis_probe_results.json}"

echo "----------------------------------------"
echo "Node:       $(hostname)"
echo "Date:       $(date)"
echo "Cache dir:  ${CACHE_DIR}"
echo "Out path:   ${OUT_PATH}"
echo "CPU info:   $(nproc) cores"
echo "----------------------------------------"

python scripts/train_probes.py \
    --cache_dir "${CACHE_DIR}" \
    --out_path  "${OUT_PATH}"

echo ""
echo "Done at $(date)"
