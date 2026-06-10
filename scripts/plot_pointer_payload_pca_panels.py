#!/usr/bin/env python3
"""Render pointer and payload DCM PCA plots side by side."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
PCA_ROOT = (
    REPO_ROOT
    / "results/dcm_runs/full_dcm_svd_n80_lr0p1/dcm_pca_clusters/"
    / "full_dcm_svd_n80_lr0p1_n400_active_layers"
)
DEFAULT_OUT_DIR = REPO_ROOT / "figures/answer_lookback_pca"


def load_points(feature_set: str) -> list[dict]:
    with (PCA_ROOT / feature_set / "cluster_assignments.csv").open() as f:
        return list(csv.DictReader(f))


def load_variance(feature_set: str) -> tuple[float, float]:
    summary = json.loads((PCA_ROOT / "cluster_summary.json").read_text())
    evr = summary["analyses"][feature_set]["explained_variance_ratio"]
    return float(evr[0]) * 100, float(evr[1]) * 100


def plot_panel(ax, feature_set: str, title: str, colors: dict[str, str]) -> None:
    points = load_points(feature_set)
    pc1_var, pc2_var = load_variance(feature_set)
    by_cluster: dict[str, list[dict]] = {}
    for point in points:
        by_cluster.setdefault(point["cluster"], []).append(point)

    for cluster, rows in sorted(by_cluster.items(), key=lambda item: int(item[0])):
        ax.scatter(
            [float(row["pc1"]) for row in rows],
            [float(row["pc2"]) for row in rows],
            s=30,
            alpha=0.82,
            color=colors.get(cluster, "#777777"),
            edgecolor="white",
            linewidth=0.3,
            label=f"Cluster {cluster} (n={len(rows)})",
        )

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel(f"PC1 ({pc1_var:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({pc2_var:.2f}% variance)")
    ax.grid(color="#dddddd", linewidth=0.7, alpha=0.75)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    colors = {"0": "#2f7fb8", "1": "#28b8c7"}
    plot_panel(axes[0], "pointer", "Pointer DCM PCA", colors)
    plot_panel(axes[1], "payload", "Payload DCM PCA", colors)
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    axes[1].legend(frameon=False, loc="upper right", fontsize=9)
    fig.suptitle("PCA of Pointer and Payload DCM Coordinates", fontsize=17, y=1.02)
    fig.tight_layout()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.out_dir / f"pointer_payload_dcm_pca_panels.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(args.out_dir / "pointer_payload_dcm_pca_panels.png")
    print(args.out_dir / "pointer_payload_dcm_pca_panels.pdf")


if __name__ == "__main__":
    main()
