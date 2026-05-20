"""
Phase 2 — Step 3: Render the decodability heatmaps and (optionally) the
decodability x causality comparison.

Heatmap per variable:
  x = layer, y = token position, color = test acc (or test acc minus baseline)

If `--iia_results_dir` is given, also produces overlay plots comparing
probe accuracy and IIA at the same (layer, token) sites — the
Huang-Chang style alignment view.

Usage:
  python scripts/analyze_probes.py \\
      --probe_path results/probes/causalToM_novis_probe_results.json \\
      --out_dir figures/probes
"""

import json
import os

import fire
import matplotlib.pyplot as plt
import numpy as np


def _heatmap_one(
    var_name: str,
    var_results: dict,
    out_dir: str,
    show_baseline_subtracted: bool = True,
):
    # Build matrix Z[pos_idx, layer_idx] = test_acc
    positions = sorted(var_results.keys(), key=lambda s: int(s))
    layer_keys: set[str] = set()
    for p in positions:
        layer_keys.update(var_results[p].keys())
    layers = sorted({int(k) for k in layer_keys})

    Z = np.full((len(positions), len(layers)), np.nan)
    B = np.full((len(positions), len(layers)), np.nan)
    for i, p in enumerate(positions):
        for j, layer in enumerate(layers):
            entry = var_results[p].get(str(layer))
            if entry is None:
                continue
            acc = entry.get("test_acc")
            base = entry.get("baseline")
            if acc is not None and np.isfinite(acc):
                Z[i, j] = acc
                if base is not None:
                    B[i, j] = base

    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.45), max(3.5, len(positions) * 0.25)))

    if show_baseline_subtracted:
        display = Z - B
        title = f"Probe acc − majority baseline: {var_name}"
        vmax = float(np.nanmax(np.abs(display))) if np.isfinite(display).any() else 0.5
        vmin = -vmax
        cmap = "RdBu_r"
    else:
        display = Z
        title = f"Probe test accuracy: {var_name}"
        vmin, vmax = 0.0, 1.0
        cmap = "viridis"

    im = ax.imshow(display, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90, fontsize=7)
    ax.set_yticks(range(len(positions)))
    ax.set_yticklabels(positions, fontsize=8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Token position (relative to end)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="test acc − baseline" if show_baseline_subtracted else "test acc")
    plt.tight_layout()

    suffix = "_centered" if show_baseline_subtracted else ""
    out_path = os.path.join(out_dir, f"probe_heatmap_{var_name}{suffix}.png")
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  wrote {out_path}")


def _layerwise_alignment(
    var_name: str,
    var_results: dict,
    iia_curve: dict[int, float] | None,
    out_dir: str,
    token_pos: int = -1,
):
    """Overlay probe accuracy and IIA at a fixed token position vs layer."""
    p_dict = var_results.get(str(token_pos))
    if p_dict is None:
        print(f"  no probe data at token_pos={token_pos} for {var_name}; skipping overlay")
        return

    layers = sorted(int(k) for k in p_dict.keys())
    probe_accs = []
    baselines = []
    for layer in layers:
        entry = p_dict.get(str(layer))
        if entry is None or entry.get("test_acc") is None:
            probe_accs.append(np.nan)
            baselines.append(np.nan)
        else:
            probe_accs.append(entry["test_acc"])
            baselines.append(entry["baseline"])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(layers, probe_accs, "-o", label=f"probe acc ({var_name})", color="C0", linewidth=1.6)
    ax.plot(layers, baselines, "--", label="majority baseline", color="C0", alpha=0.4)

    if iia_curve is not None:
        iia_layers = sorted(iia_curve.keys())
        iia_vals = [iia_curve[L] for L in iia_layers]
        ax.plot(iia_layers, iia_vals, "-s", label="IIA (full residual)", color="C3", linewidth=1.6)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{var_name} at token {token_pos}: decodability vs causality")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"overlay_{var_name}_pos{token_pos}.png")
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  wrote {out_path}")


def _load_iia_curve(iia_dir: str) -> dict[int, float] | None:
    """Parse the per-layer JSON outputs produced by
    run_single_layer_patching_exps.py into {layer: full_rank_accuracy}.
    """
    if not os.path.isdir(iia_dir):
        return None
    curve: dict[int, float] = {}
    for fname in os.listdir(iia_dir):
        if not fname.endswith(".json"):
            continue
        try:
            layer = int(fname.split(".")[0])
        except ValueError:
            continue
        with open(os.path.join(iia_dir, fname)) as f:
            data = json.load(f)
        full = data.get("full_rank", {}).get("accuracy")
        if full is not None:
            curve[layer] = float(full)
    return curve if curve else None


def main(
    probe_path: str = "results/probes/causalToM_novis_probe_results.json",
    out_dir: str = "figures/probes",
    iia_results_dir: str | None = None,
):
    os.makedirs(out_dir, exist_ok=True)
    with open(probe_path) as f:
        results = json.load(f)

    iia_curve = _load_iia_curve(iia_results_dir) if iia_results_dir else None
    if iia_curve is None and iia_results_dir is not None:
        print(f"  no IIA curve loaded from {iia_results_dir}; rendering heatmaps only")

    for var_name, var_results in results.items():
        print(f"=== {var_name} ===")
        _heatmap_one(var_name, var_results, out_dir, show_baseline_subtracted=True)
        _heatmap_one(var_name, var_results, out_dir, show_baseline_subtracted=False)
        if iia_curve is not None:
            for pos in (-1, -4, -7):
                _layerwise_alignment(var_name, var_results, iia_curve, out_dir, token_pos=pos)


if __name__ == "__main__":
    fire.Fire(main)
