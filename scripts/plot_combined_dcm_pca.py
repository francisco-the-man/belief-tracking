#!/usr/bin/env python3
"""Render a clean combined pointer/payload DCM PCA plot."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLUSTER_CSV = (
    REPO_ROOT
    / "results/dcm_runs/full_dcm_svd_n80_lr0p1/dcm_pca_clusters/"
    / "full_dcm_svd_n80_lr0p1_n400_active_layers/both/cluster_assignments.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "figures/answer_lookback_pca"


def load_points(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def cluster_label(rows: list[dict]) -> str:
    unknown = Counter(row["is_unknown"] for row in rows)
    n = len(rows)
    unknown_frac = unknown.get("1", 0) / max(n, 1)
    if unknown_frac > 0.8:
        return f"Cluster {rows[0]['cluster']}: unknown-answer prompts"
    if unknown_frac < 0.2:
        return f"Cluster {rows[0]['cluster']}: answer-known prompts"
    return f"Cluster {rows[0]['cluster']}: mixed prompts"


def plot(points: list[dict], out_dir: Path) -> None:
    by_cluster: dict[str, list[dict]] = {}
    for point in points:
        by_cluster.setdefault(point["cluster"], []).append(point)

    colors = {"0": "#2f7fb8", "1": "#28b8c7"}
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(8.2, 6.1))
    for cluster, rows in sorted(by_cluster.items(), key=lambda item: int(item[0])):
        xs = [float(row["pc1"]) for row in rows]
        ys = [float(row["pc2"]) for row in rows]
        ax.scatter(
            xs,
            ys,
            s=36,
            alpha=0.82,
            color=colors.get(cluster, "#777777"),
            edgecolor="white",
            linewidth=0.35,
            label=f"{cluster_label(rows)} (n={len(rows)})",
        )

    ax.set_title("PCA of Payload and Pointer DCM Coordinates", fontsize=16, pad=12)
    ax.set_xlabel("PC1 (21.73% variance)")
    ax.set_ylabel("PC2 (4.68% variance)")
    ax.grid(color="#dddddd", linewidth=0.7, alpha=0.75)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"payload_pointer_dcm_pca.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-csv", type=Path, default=DEFAULT_CLUSTER_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    points = load_points(args.cluster_csv)
    plot(points, args.out_dir)
    print(args.out_dir / "payload_pointer_dcm_pca.png")
    print(args.out_dir / "payload_pointer_dcm_pca.pdf")


if __name__ == "__main__":
    main()
