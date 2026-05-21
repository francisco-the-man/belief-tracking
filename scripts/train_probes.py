"""
Phase 2 — Step 2: Train linear probes on cached activations for each
lookback variable at every (layer, token) position.

Pure CPU. Runs on the Farmshare login node or your laptop after
cache_activations.py has populated <cache_dir>.

For each (variable, token_position, layer):
  - X = cached activations (n_samples, hidden_dim)
  - y = ground-truth labels from labels.json
  - Fit LogisticRegression with stratified train/test split
  - Record test accuracy + class-balance baseline

Output:
  <out_path>: nested dict { variable: { token_pos (str): { layer (str): {
      "test_acc": float, "baseline": float, "n": int, "n_classes": int
  } } } }
"""

import json
import os
from typing import Callable

import fire
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def _label_char_idx(lbl: dict):
    return lbl["char_idx"]


def _label_obj_idx(lbl: dict):
    return lbl["obj_idx"]


def _label_is_unknown(lbl: dict):
    return lbl["is_unknown"]


def _label_state_idx_when_known(lbl: dict):
    # Probing the Answer Pointer: which of state_0 / state_1 is the answer?
    # Only defined when the answer is not "unknown".
    return lbl["state_idx"] if lbl["state_idx"] in (0, 1) else None


def _label_query_pair(lbl: dict):
    # Joint (char_idx, obj_idx) — 4 classes. Tests whether the position
    # encodes the full query.
    return lbl["char_idx"] * 2 + lbl["obj_idx"]


VARIABLES: dict[str, Callable[[dict], int | None]] = {
    "char_idx": _label_char_idx,
    "obj_idx": _label_obj_idx,
    "is_unknown": _label_is_unknown,
    "state_idx_when_known": _label_state_idx_when_known,
    "query_pair": _label_query_pair,
}


def _make_y(labels: list[dict], var_name: str) -> tuple[np.ndarray, np.ndarray]:
    spec = VARIABLES[var_name]
    y: list[int] = []
    keep: list[int] = []
    for i, lbl in enumerate(labels):
        v = spec(lbl)
        if v is None:
            continue
        keep.append(i)
        y.append(v)
    return np.array(y), np.array(keep)


def _probe(X: np.ndarray, y: np.ndarray, seed: int = 0, pca_dim: int = 200) -> dict:
    """Train/test a logistic regression probe and return summary stats.

    Pipeline: StandardScaler -> PCA(pca_dim) -> LogReg(liblinear/saga).
    PCA reduction is ~25x faster than fitting LR directly on 5120 dims and
    preserves linearly decodable signal (for linearly separable classes the
    top PCs span the discriminative directions).
    """
    n_classes = int(len(np.unique(y)))
    counts = np.bincount(y)
    baseline = float(counts.max() / counts.sum())

    if n_classes < 2 or len(y) < 16:
        return {
            "test_acc": float("nan"),
            "baseline": baseline,
            "n": int(len(y)),
            "n_classes": n_classes,
            "note": "skipped (too few samples or only one class)",
        }

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=seed
        )

    # PCA dim capped by min(n_train, n_features)
    effective_pca = min(pca_dim, X_tr.shape[0] - 1, X_tr.shape[1])
    solver = "liblinear" if n_classes == 2 else "lbfgs"
    pipe = Pipeline([
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("pca",   PCA(n_components=effective_pca, random_state=seed)),
        ("lr",    LogisticRegression(max_iter=1000, C=1.0, solver=solver)),
    ])
    pipe.fit(X_tr, y_tr)
    return {
        "test_acc": float(pipe.score(X_te, y_te)),
        "baseline": baseline,
        "n": int(len(y)),
        "n_classes": n_classes,
        "pca_dim": effective_pca,
    }


def main(
    cache_dir: str = "additionals/cached_acts/causalToM_novis",
    out_path: str = "results/probes/causalToM_novis_probe_results.json",
    variables: list | None = None,
    seed: int = 0,
):
    with open(os.path.join(cache_dir, "cache_meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(cache_dir, "labels.json")) as f:
        labels = json.load(f)

    layers: list[int] = meta["layers"]
    neg_positions: list[int] = meta["neg_positions"]

    variables = list(variables) if variables else list(VARIABLES.keys())

    print(f"Variables to probe: {variables}")
    print(f"Layers: {layers}")
    print(f"Token positions: {neg_positions}")
    print(f"n_samples: {len(labels)}")

    results: dict[str, dict] = {}
    for var_name in variables:
        y, keep_idx = _make_y(labels, var_name)
        if len(y) < 16:
            print(f"  skipping {var_name}: only {len(y)} usable samples")
            continue
        print(
            f"\n=== {var_name} === n={len(y)} | class_counts={np.bincount(y).tolist()}"
        )
        var_results: dict[str, dict] = {}
        for t in tqdm(neg_positions, desc=f"  {var_name}", leave=False):
            t_key = str(t)
            var_results[t_key] = {}
            for layer in layers:
                tp = os.path.join(cache_dir, f"layer_{layer}", f"pos_{t}.pt")
                if not os.path.exists(tp):
                    var_results[t_key][str(layer)] = None
                    continue
                X = torch.load(tp).to(torch.float32).numpy()[keep_idx]
                var_results[t_key][str(layer)] = _probe(X, y, seed=seed)
        results[var_name] = var_results
        # save partial after each variable so a walltime kill doesn't lose everything
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved probe results to {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
