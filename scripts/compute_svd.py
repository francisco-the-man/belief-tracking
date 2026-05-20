"""
Phase 0 — Step 1: Compute SVD bases at the residual-stream sites used by
Answer Lookback experiments.

For each requested layer, this script:
  1. Builds the appropriate CausalToM dataset (clean + counterfactual prompts).
  2. Filters to samples where the LM is correct on both prompts.
  3. Caches the residual stream at the intervention token position for both
     prompts in a single batched forward pass per (layer, prompt) pair.
  4. Computes SVD over the resulting activation matrix and saves the right
     singular vectors so DCM training (in run_single_layer_patching_exps.py)
     can load them via `load_basis_directions`.

Output layout matches `load_basis_directions` expectations:
  <out_prefix>/svd/<path>/<vector_type>/singular_vecs/<layer>.pt

where <path> = "causalToM" or "bigToM", <vector_type> e.g. "last_token".

Usage:
  python scripts/compute_svd.py \\
      --experiment answer_lookback-payload \\
      --model_key Qwen/Qwen2.5-14B-Instruct \\
      --layers '[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46]' \\
      --n_samples 80 \\
      --out_prefix additionals
"""

import json
import os
import sys
from typing import Literal

import fire
import torch
from nnsight import LanguageModel
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make sibling modules importable.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(SCRIPT_DIR, "patching_scripts"))

from run_patching_exp_utils import (  # noqa: E402
    exp_to_intervention_positions,
    exp_to_vec_type,
    free_gpu_cache,
    prepare_dataset,
    set_seed,
)
from src import global_utils  # noqa: E402


def _resolve_token_positions(experiment: str, lm: LanguageModel) -> list[int]:
    """Return the list of token positions (potentially negative) at which we
    should cache residuals for SVD. Mirrors the patching scripts' logic.
    """
    positions = exp_to_intervention_positions[experiment]["cache"].copy()

    # Qwen2.5-7B (and presumably 14B) shifts non-answer-lookback intervention
    # positions by -1 because of tokenization differences. We mirror exactly
    # what run_single_layer_patching_exps.py does so the SVD matches the
    # actual patching site.
    if (
        lm.config.architectures[0] == "Qwen2ForCausalLM"
        and "answer_lookback" not in experiment
    ):
        positions = [p - 1 for p in positions]
    return positions


@torch.inference_mode()
def _collect_residuals(
    lm: LanguageModel,
    dataloader: DataLoader,
    layers: list[int],
    token_positions: list[int],
) -> dict[int, dict[int, torch.Tensor]]:
    """Cache residual streams at every (layer, token_position) for both
    clean and counterfactual prompts.

    Returns:
        out[layer][position] = tensor of shape (2 * n_samples, hidden_dim)
        with clean-then-counterfactual stacking.
    """
    cache: dict[int, dict[int, list[torch.Tensor]]] = {
        layer: {t: [] for t in token_positions} for layer in layers
    }

    for batch in tqdm(dataloader, desc="Caching residuals"):
        clean_prompts = batch["clean_prompt"]
        cf_prompts = batch["counterfactual_prompt"]

        # Single trace, two invokes — one forward pass per prompt batch.
        with lm.trace() as tracer:
            saved_clean: dict[int, dict[int, torch.Tensor]] = {
                layer: {} for layer in layers
            }
            saved_cf: dict[int, dict[int, torch.Tensor]] = {
                layer: {} for layer in layers
            }

            with tracer.invoke(clean_prompts):
                for layer in layers:
                    for t in token_positions:
                        saved_clean[layer][t] = (
                            lm.model.layers[layer].output[:, t].clone().save()
                        )

            with tracer.invoke(cf_prompts):
                for layer in layers:
                    for t in token_positions:
                        saved_cf[layer][t] = (
                            lm.model.layers[layer].output[:, t].clone().save()
                        )

        for layer in layers:
            for t in token_positions:
                cache[layer][t].append(saved_clean[layer][t].detach().cpu().float())
                cache[layer][t].append(saved_cf[layer][t].detach().cpu().float())

        free_gpu_cache()

    out: dict[int, dict[int, torch.Tensor]] = {layer: {} for layer in layers}
    for layer in layers:
        for t in token_positions:
            out[layer][t] = torch.cat(cache[layer][t], dim=0)
    return out


def _svd_basis(activations: torch.Tensor) -> torch.Tensor:
    """Right singular vectors of `activations` (rows = directions).

    activations: shape (n, d)
    returns: Vh, shape (min(n, d), d) — each row is a unit-norm right singular
    vector, ordered by descending singular value.
    """
    # Center: SVD on (X - mean) recovers principal directions of variance.
    centered = activations - activations.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return vh.contiguous()


def main(
    experiment: Literal[
        "answer_lookback-pointer",
        "answer_lookback-payload",
        "binding_lookback-pointer_object",
        "binding_lookback-pointer_character",
        "binding_lookback-address_and_payload",
        "binding_lookback-object_oi",
        "binding_lookback-character_oi",
        "visibility_lookback-source",
    ] = "answer_lookback-payload",
    model_key: str = "Qwen/Qwen2.5-14B-Instruct",
    layers: list[int] | None = None,
    n_samples: int = 80,
    batch_size: int = 4,
    out_prefix: str = "additionals",
    dataset_path_tag: str = "causalToM",
    seed: int = 10,
):
    """Compute SVD bases and save them at:
       <out_prefix>/svd/<dataset_path_tag>/<vector_type>/singular_vecs/<layer>.pt
    """
    set_seed(seed)

    vector_type = exp_to_vec_type[experiment]
    if vector_type is None:
        raise ValueError(
            f"Experiment {experiment} has no SVD vector_type — handled differently."
        )
    if isinstance(vector_type, list):
        raise NotImplementedError(
            f"Experiment {experiment} uses a dict basis (multiple vector_types). "
            "Run this script once per vector_type, or extend to multi-type."
        )

    print("#" * 60)
    print(f"Computing SVD for experiment={experiment}")
    print(f"  model_key={model_key}")
    print(f"  vector_type={vector_type}")
    print(f"  n_samples={n_samples} batch_size={batch_size}")
    print("#" * 60)

    # We want all `n_samples` examples in a single pool — pass them all as
    # train_size with valid_size=0 so prepare_dataset doesn't split them.
    # NOTE: The paper used fp32 for Qwen-14B on 80GB A100s. On L40S (48GB)
    # fp32 doesn't fit — we use fp16 universally for ≤70B models. SVD bases
    # are robust to fp16 numerics.
    is_405b = "405B" in model_key
    dtype = torch.bfloat16 if is_405b else torch.float16  # 405B handled separately

    lm = LanguageModel(
        model_key,
        device_map="auto",
        dtype=dtype,
        dispatch=True,
    )

    # Position -1 is the ':' of 'Answer:'; with batched inputs we need
    # left-padding so that index -1 is always the genuine last token.
    lm.tokenizer.padding_side = "left"
    if lm.tokenizer.pad_token is None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token

    if layers is None:
        n_layer = lm.config.num_hidden_layers
        # Default: every other layer, matching the paper's Qwen sweep.
        step = 2 if n_layer <= 64 else 4
        layers = list(range(0, n_layer, step))
    else:
        # `fire` passes JSON-like list literal as-is.
        layers = list(layers)

    print(f"  layers={layers}")

    train_loader, _ = prepare_dataset(
        experiment_name=experiment,
        train_size=n_samples,
        valid_size=0,
        batch_size=batch_size,
        lm=lm,
        remote=False,
    )

    token_positions = _resolve_token_positions(experiment, lm)
    print(f"  token_positions={token_positions}")

    # Cache residuals at every requested (layer, position).
    residuals = _collect_residuals(lm, train_loader, layers, token_positions)

    # SVD per layer. For experiments with a single intervention position (the
    # Answer Lookback case), residuals[layer] has one entry; we use it
    # directly. For multi-position experiments (binding lookback at state
    # tokens etc), we concatenate residuals across positions before SVD so
    # the basis spans variation at any of them — consistent with how the
    # paper applies the same per-layer basis to multiple positions.
    # load_basis_directions iterates every file in singular_vecs/ and parses
    # the filename as int(layer), so we keep that dir pure .pt and write
    # metadata one level up.
    vec_type_dir = os.path.join(out_prefix, "svd", dataset_path_tag, vector_type)
    out_dir = os.path.join(vec_type_dir, "singular_vecs")
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "experiment": experiment,
        "model_key": model_key,
        "vector_type": vector_type,
        "n_samples": n_samples,
        "token_positions": token_positions,
        "layers": layers,
        "seed": seed,
    }
    layer_meta: dict[int, dict] = {}

    for layer in layers:
        per_pos = [residuals[layer][t] for t in token_positions]
        stacked = torch.cat(per_pos, dim=0)  # (k * 2 * n_samples, hidden_dim)
        basis = _svd_basis(stacked)
        save_path = os.path.join(out_dir, f"{layer}.pt")
        torch.save(basis, save_path)
        layer_meta[layer] = {
            "n_directions": basis.shape[0],
            "hidden_dim": basis.shape[1],
            "input_rows": stacked.shape[0],
        }
        print(
            f"  layer {layer:>3}: saved {basis.shape} to {save_path} "
            f"(from {stacked.shape[0]} rows)"
        )

    meta["per_layer"] = layer_meta
    meta_path = os.path.join(vec_type_dir, "svd_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    fire.Fire(main)
