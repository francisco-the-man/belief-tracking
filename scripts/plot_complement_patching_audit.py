#!/usr/bin/env python3
"""Plot DCM subspace vs orthogonal-complement patching audits."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = REPO_ROOT / "results/dcm_runs/full_dcm_svd_n80_lr0p1"
DEFAULT_OUT_DIR = REPO_ROOT / "figures/answer_lookback_complement_audit"


@dataclass(frozen=True)
class AuditSpec:
    key: str
    title: str
    path: Path
    color: str


SPECS = (
    AuditSpec(
        key="pointer_lamb0p2",
        title="Pointer lambda 0.2 (rank-2 peak)",
        path=RUN_ROOT
        / "causal_complement_audit/full_dcm_svd_n80_lr0p1/answer_lookback-pointer/causal_complement_audit.json",
        color="#c76a97",
    ),
    AuditSpec(
        key="pointer_lamb0p02",
        title="Pointer lambda 0.02 (corrected)",
        path=RUN_ROOT
        / "causal_complement_audit_pointer_lamb0p02/full_dcm_svd_n80_lr0p1_pointer_lamb0p02/answer_lookback-pointer/causal_complement_audit.json",
        color="#7a5195",
    ),
    AuditSpec(
        key="payload_lamb0p1",
        title="Payload lambda 0.1",
        path=RUN_ROOT
        / "causal_complement_audit/full_dcm_svd_n80_lr0p1/answer_lookback-payload/causal_complement_audit.json",
        color="#4d4d4d",
    ),
)


def load_rows(spec: AuditSpec) -> list[dict]:
    data = json.loads(spec.path.read_text())
    rows = []
    for row in data["rows"]:
        rows.append(
            {
                "audit": spec.key,
                "title": spec.title,
                "layer": int(row["anchor_layer"]),
                "rank": int(float(row["rank"])),
                "full_iia": float(row["dcm_full_iia"]),
                "subspace_iia": float(row["dcm_sub_iia"]),
                "complement_iia": float(row["dcm_complement_iia"]),
            }
        )
    return rows


def write_table(rows: list[dict], out_dir: Path) -> None:
    fieldnames = [
        "audit",
        "title",
        "layer",
        "rank",
        "full_iia",
        "subspace_iia",
        "complement_iia",
    ]
    with (out_dir / "complement_patching_curves.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "complement_patching_curves.json").write_text(json.dumps(rows, indent=2))


def plot(rows: list[dict], out_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    by_key = {spec.key: [r for r in rows if r["audit"] == spec.key] for spec in SPECS}

    for ax, spec in zip(axes, SPECS, strict=True):
        curve = sorted(by_key[spec.key], key=lambda r: r["layer"])
        layers = np.array([r["layer"] for r in curve])
        full = np.array([r["full_iia"] for r in curve])
        subspace = np.array([r["subspace_iia"] for r in curve])
        complement = np.array([r["complement_iia"] for r in curve])
        ranks = np.array([r["rank"] for r in curve])

        ax.plot(layers, full, "-", color=spec.color, linewidth=2.2, label="Full residual")
        ax.plot(layers, subspace, "--", color=spec.color, linewidth=2.2, label="DCM subspace")
        ax.plot(layers, complement, ":", color="#1f7a5c", linewidth=2.6, label="Orthogonal complement")
        ax.scatter(layers[ranks > 0], subspace[ranks > 0], s=18 + ranks[ranks > 0] * 1.2, color=spec.color, alpha=0.35)

        if spec.key == "pointer_lamb0p2":
            for layer in (32, 34):
                row = next(r for r in curve if r["layer"] == layer)
                ax.annotate(
                    f"rank {row['rank']}",
                    xy=(layer, row["subspace_iia"]),
                    xytext=(layer - 2.8, row["subspace_iia"] - 0.22),
                    arrowprops=dict(arrowstyle="->", color="#555555", linewidth=0.8),
                    fontsize=9,
                    color="#333333",
                )

        ax.set_title(spec.title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_xlim(0, 47)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([0, 10, 20, 30, 40, 47])
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)

    axes[0].set_ylabel("Intervention accuracy / IIA")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Orthogonal-Complement Patching Audit, Qwen2.5-14B-Instruct Answer Lookback", fontsize=13)
    fig.text(
        0.5,
        -0.01,
        "Complement patching means: patch x - P_v(x), leaving out the DCM-selected subspace v. "
        "If complement IIA is low, the patched information outside v is not sufficient to causally transfer the target answer.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"complement_patching_audit.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    missing = [str(spec.path) for spec in SPECS if not spec.path.exists()]
    if missing:
        raise FileNotFoundError("Missing audit artifacts:\n" + "\n".join(missing))
    rows = []
    for spec in SPECS:
        rows.extend(load_rows(spec))
    write_table(rows, args.out_dir)
    plot(rows, args.out_dir)
    print(args.out_dir / "complement_patching_audit.png")
    print(args.out_dir / "complement_patching_curves.csv")


if __name__ == "__main__":
    main()
