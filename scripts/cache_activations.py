"""
Phase 2 — Step 1: Cache residual stream activations at every (layer, token)
for a pool of CausalToM prompts, so we can train linear probes for the
lookback variables without further GPU calls.

For each prompt we record ground-truth labels for the lookback variables
(character index queried, object index queried, state index of the answer,
etc.) alongside the cached activations.

We left-pad so that negative token positions (e.g. -1 = the ':' of 'Answer:')
are stable across prompts of varying length.

Output layout:
  <out_dir>/labels.json
  <out_dir>/cache_meta.json
  <out_dir>/layer_<L>/pos_<T>.pt   # shape (n_samples, hidden_dim), fp16

Usage:
  python scripts/cache_activations.py \\
      --model_key Qwen/Qwen2.5-14B-Instruct \\
      --n_samples 500 \\
      --batch_size 8
"""

import json
import os
import random
import sys

import fire
import torch
from nnsight import LanguageModel
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(SCRIPT_DIR, "patching_scripts"))

from src.dataset import Dataset, Sample  # noqa: E402
from src import global_utils  # noqa: E402

# Default token positions to cache (relative to end of prompt).
# -1 is ':' of 'Answer:'; -8..-1 covers the question; -30..-9 reaches into
# the story; -1 step gives full resolution.
DEFAULT_NEG_POSITIONS = list(range(-30, 0))


def _build_samples(n_samples: int, template_idx: int, seed: int) -> list[dict]:
    """Generate n_samples CausalToM prompts with ground-truth labels.

    For template_idx=2 (no-visibility), each character only knows the state
    of the object they themselves filled:
      char 0 knows objects[0] -> states[0], believes objects[1] is unknown
      char 1 knows objects[1] -> states[1], believes objects[0] is unknown
    So the queried-character × queried-object combination fully determines
    the answer.
    """
    random.seed(seed)
    data_dir = os.path.join(PROJECT_ROOT, "data", "synthetic_entities")
    all_chars = json.load(open(os.path.join(data_dir, "characters.json")))
    all_objs = json.load(open(os.path.join(data_dir, "bottles.json")))
    all_states = json.load(open(os.path.join(data_dir, "drinks.json")))

    seen_stories: set[str] = set()
    samples: list[dict] = []

    attempts = 0
    while len(samples) < n_samples and attempts < n_samples * 10:
        attempts += 1
        characters = random.sample(all_chars, 2)
        objects = random.sample(all_objs, 2)
        states = random.sample(all_states, 2)

        s = Sample(
            template_idx=template_idx,
            characters=characters,
            objects=objects,
            states=states,
        )
        if s.story in seen_stories:
            continue
        seen_stories.add(s.story)

        ds = Dataset([s])
        char_idx = random.choice([0, 1])
        obj_idx = random.choice([0, 1])
        item = ds.__getitem__(0, set_character=char_idx, set_container=obj_idx)

        target = item["target"]  # state name or "unknown"
        if target == "unknown":
            state_idx = -1  # encoded as -1 in the label, filtered later
        else:
            state_idx = states.index(target)

        samples.append(
            {
                "prompt": item["prompt"],
                "labels": {
                    "char_idx": char_idx,
                    "obj_idx": obj_idx,
                    "target_state": target,
                    "state_idx": state_idx,
                    "is_unknown": int(target == "unknown"),
                    "characters": characters,
                    "objects": objects,
                    "states": states,
                    "template_idx": template_idx,
                },
            }
        )

    if len(samples) < n_samples:
        print(
            f"WARNING: only generated {len(samples)} unique samples "
            f"(asked for {n_samples})."
        )
    return samples


@torch.inference_mode()
def _cache_batch(
    lm: LanguageModel,
    prompts: list[str],
    layers: list[int],
    neg_positions: list[int],
) -> dict[int, dict[int, torch.Tensor]]:
    """Single forward pass, cache residuals at all requested (layer, token).

    Returns:
        out[layer][token_pos] = tensor (batch_size, hidden_dim)
    """
    saved: dict[int, dict[int, torch.Tensor]] = {layer: {} for layer in layers}

    with lm.trace() as tracer:
        with tracer.invoke(prompts):
            for layer in layers:
                for t in neg_positions:
                    saved[layer][t] = (
                        lm.model.layers[layer].output[:, t].clone().save()
                    )

    out: dict[int, dict[int, torch.Tensor]] = {layer: {} for layer in layers}
    for layer in layers:
        for t in neg_positions:
            # Move to CPU and downcast for storage. Cast back to fp32 at
            # probe-train time if needed.
            out[layer][t] = saved[layer][t].detach().cpu().to(torch.float16)
    return out


def main(
    model_key: str = "Qwen/Qwen2.5-14B-Instruct",
    n_samples: int = 500,
    batch_size: int = 8,
    out_dir: str = "additionals/cached_acts/causalToM_novis",
    layers: list | None = None,
    neg_positions: list | None = None,
    template_idx: int = 2,
    seed: int = 42,
):
    """Cache residual stream activations for a pool of CausalToM prompts."""
    neg_positions = (
        list(neg_positions) if neg_positions is not None else list(DEFAULT_NEG_POSITIONS)
    )

    is_70b = "Meta-Llama-3-70B-Instruct" in model_key
    dtype = torch.float16 if is_70b else torch.float32

    lm = LanguageModel(
        model_key,
        device_map="auto",
        dtype=dtype,
        dispatch=True,
    )
    lm.tokenizer.padding_side = "left"
    if lm.tokenizer.pad_token is None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token

    if layers is None:
        n_layer = lm.config.num_hidden_layers
        step = 2 if n_layer <= 64 else 4
        layers = list(range(0, n_layer, step))
    else:
        layers = list(layers)

    print("#" * 60)
    print(f"Caching activations for {model_key}")
    print(f"  n_samples={n_samples}, batch_size={batch_size}")
    print(f"  template_idx={template_idx} (2 = no-visibility)")
    print(f"  layers={layers}")
    print(f"  neg_positions={neg_positions}")
    print(f"  out_dir={out_dir}")
    print("#" * 60)

    samples = _build_samples(n_samples, template_idx, seed)
    print(f"Generated {len(samples)} unique prompts")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump([s["labels"] for s in samples], f, indent=2)

    # Per-layer-per-position accumulators of CPU tensors.
    acc: dict[int, dict[int, list[torch.Tensor]]] = {
        layer: {t: [] for t in neg_positions} for layer in layers
    }

    prompts = [s["prompt"] for s in samples]
    for i in tqdm(range(0, len(prompts), batch_size), desc="Caching"):
        batch = prompts[i : i + batch_size]
        cached = _cache_batch(lm, batch, layers, neg_positions)
        for layer in layers:
            for t in neg_positions:
                acc[layer][t].append(cached[layer][t])
        torch.cuda.empty_cache()

    # Concatenate and save per (layer, position).
    print("Saving per-(layer, position) tensors...")
    for layer in tqdm(layers):
        layer_dir = os.path.join(out_dir, f"layer_{layer}")
        os.makedirs(layer_dir, exist_ok=True)
        for t in neg_positions:
            tensor = torch.cat(acc[layer][t], dim=0)
            torch.save(tensor, os.path.join(layer_dir, f"pos_{t}.pt"))

    meta = {
        "model_key": model_key,
        "n_samples": len(samples),
        "layers": layers,
        "neg_positions": neg_positions,
        "template_idx": template_idx,
        "seed": seed,
        "hidden_dim": lm.config.hidden_size,
        "n_layers_total": lm.config.num_hidden_layers,
        "storage_dtype": "float16",
    }
    with open(os.path.join(out_dir, "cache_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    n_files = len(layers) * len(neg_positions)
    bytes_per_slot = len(samples) * lm.config.hidden_size * 2  # fp16
    print(
        f"Done. Wrote {n_files} tensor files "
        f"(~{n_files * bytes_per_slot / 1e9:.1f} GB total) to {out_dir}/"
    )


if __name__ == "__main__":
    fire.Fire(main)
