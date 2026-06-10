import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DCM_ROOT = REPO_ROOT / "results" / "dcm_runs" / "full_dcm_svd_n80_lr0p1"
DEFAULT_OUT_DIR = REPO_ROOT / "figures" / "answer_lookback"


ORIGINAL_TEXT = (
    "Bob and Carla are working in a busy restaurant. To complete\n"
    "an order, Bob grabs an opaque bottle and fills it with beer.\n"
    "Then Carla grabs another opaque cup and fills it with coffee.\n"
    "Question: What does Carla believe the cup contains?\n"
    "Answer: coffee"
)

COUNTERFACTUAL_TEXT = (
    "Carla and Bob are working in a busy restaurant. To complete\n"
    "an order, Carla grabs an opaque cup and fills it with tea.\n"
    "Then Bob grabs another opaque bottle and fills it with water.\n"
    "Question: What does Carla believe the cup contains?\n"
    "Answer: tea"
)


@dataclass(frozen=True)
class CurveSpec:
    key: str
    title: str
    dcm_rel: Path
    svd_rel: Path
    color: str
    expected_output: str


def build_specs(pointer_lambda: str) -> tuple[CurveSpec, CurveSpec]:
    pointer_tags = {
        "0p02": (
            "full_dcm_svd_n80_lr0p1_pointer_lamb0p02",
            "lamb_0p02",
            "Answer pointer, lambda 0.02",
        ),
        "0p2": (
            "full_dcm_svd_n80_lr0p1_pointer_lamb0p2",
            "lamb_0p2",
            "Answer pointer, lambda 0.2",
        ),
    }
    pointer_run, pointer_lamb_dir, pointer_title = pointer_tags[pointer_lambda]
    return (
        CurveSpec(
        key="pointer",
        title=pointer_title,
        dcm_rel=Path(
            f"{pointer_run}/{pointer_lamb_dir}/causalToM_novis/"
            "Qwen2.5-14B-Instruct/answer_lookback/pointer"
        ),
        svd_rel=Path(
            "svd_snapshots/svd_n80/answer_lookback-pointer"
            "/causalToM/last_token/singular_vecs"
        ),
        color="#c76a97",
        expected_output="beer",
    ),
        CurveSpec(
        key="payload",
        title="Answer payload",
        dcm_rel=Path(
            "full_dcm_svd_n80_lr0p1_payload_lamb0p1"
            "/lamb_0p1/causalToM_novis/Qwen2.5-14B-Instruct"
            "/answer_lookback/payload"
        ),
        svd_rel=Path(
            "svd_snapshots/svd_n80/answer_lookback-payload"
            "/causalToM/last_token/singular_vecs"
        ),
        color="#4d4d4d",
        expected_output="tea",
    ),
    )


def load_curve(root: Path, spec: CurveSpec, verify_svd: bool) -> list[dict]:
    dcm_dir = root / spec.dcm_rel
    svd_dir = root / spec.svd_rel
    if not dcm_dir.is_dir():
        raise FileNotFoundError(f"Missing DCM directory: {dcm_dir}")
    if verify_svd and not svd_dir.is_dir():
        raise FileNotFoundError(f"Missing SVD directory: {svd_dir}")

    rows = []
    for dcm_path in sorted(dcm_dir.glob("*.json"), key=lambda path: int(path.stem)):
        layer = int(dcm_path.stem)
        dcm = json.loads(dcm_path.read_text())
        mask = torch.tensor(dcm["singular_vector"]["metadata"]["mask"], dtype=torch.bool)
        rank = int(dcm["singular_vector"]["rank"])
        mask_rank = int(mask.sum().item())
        if rank != mask_rank:
            raise ValueError(
                f"{spec.key} layer {layer}: rank={rank}, mask true count={mask_rank}"
            )

        basis_shape = None
        ortho_error = None
        if verify_svd:
            basis_path = svd_dir / f"{layer}.pt"
            if not basis_path.exists():
                raise FileNotFoundError(f"Missing SVD basis: {basis_path}")
            basis = torch.load(basis_path, map_location="cpu")
            if basis.ndim != 2:
                raise ValueError(f"{basis_path} should be a rank-2 tensor")
            if mask.numel() != basis.shape[0]:
                raise ValueError(
                    f"{spec.key} layer {layer}: mask length={mask.numel()}, "
                    f"basis rows={basis.shape[0]}"
                )
            selected = basis[mask]
            basis_shape = tuple(int(x) for x in basis.shape)
            if selected.numel() > 0:
                gram = selected @ selected.T
                identity = torch.eye(selected.shape[0], dtype=gram.dtype)
                ortho_error = float((gram - identity).abs().max().item())
            else:
                ortho_error = 0.0

        rows.append(
            {
                "intervention": spec.key,
                "layer": layer,
                "rank": rank,
                "full_residual_accuracy": float(dcm["full_rank"]["accuracy"]),
                "subspace_accuracy": float(dcm["singular_vector"]["accuracy"]),
                "basis_shape": basis_shape,
                "selected_basis_orthonormality_max_error": ortho_error,
                "expected_output": spec.expected_output,
                "dcm_path": str(dcm_path),
                "basis_path": str((svd_dir / f"{layer}.pt")) if verify_svd else "",
            }
        )

    return rows


def reconstruct_projection(
    root: Path,
    spec: CurveSpec,
    layer: int,
    complement: bool = False,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Materialize P_v, or I - P_v, for a DCM-selected SVD subspace.

    This is intentionally not called by the plotting path because each Qwen2.5-14B
    projection is 5120 x 5120. It is here so follow-up patching code can import
    the same artifact resolution and reconstruction logic used by this figure.
    """
    dcm_path = root / spec.dcm_rel / f"{layer}.json"
    basis_path = root / spec.svd_rel / f"{layer}.pt"
    dcm = json.loads(dcm_path.read_text())
    basis = torch.load(basis_path, map_location="cpu").to(dtype=dtype)
    mask = torch.tensor(dcm["singular_vector"]["metadata"]["mask"], dtype=torch.bool)
    selected = basis[mask]
    projection = selected.T @ selected
    if complement:
        projection = torch.eye(projection.shape[0], dtype=projection.dtype) - projection
    return projection


def write_table(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "figure4_qwen_dcm_curves.csv"
    json_path = out_dir / "figure4_qwen_dcm_curves.json"
    fieldnames = [
        "intervention",
        "layer",
        "rank",
        "full_residual_accuracy",
        "subspace_accuracy",
        "expected_output",
        "basis_shape",
        "selected_basis_orthonormality_max_error",
        "dcm_path",
        "basis_path",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    json_path.write_text(json.dumps(rows, indent=2))


def plot_figure(rows: list[dict], out_dir: Path, specs: tuple[CurveSpec, ...]) -> None:
    by_key = {spec.key: [row for row in rows if row["intervention"] == spec.key] for spec in specs}

    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(13.8, 4.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.25], wspace=0.3)
    ax_text = fig.add_subplot(grid[0, 0])
    ax_plot = fig.add_subplot(grid[0, 1])

    ax_text.axis("off")
    ax_text.text(0.0, 1.02, "(a) Intervention input example", fontsize=10, fontweight="bold")
    box = dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="#b8b8b8", linewidth=0.8)
    ax_text.text(
        0.0,
        0.86,
        "Counterfactual\n\n" + COUNTERFACTUAL_TEXT,
        ha="left",
        va="top",
        fontsize=8.8,
        family="monospace",
        bbox=box,
        linespacing=1.1,
    )
    ax_text.text(
        0.0,
        0.43,
        "Original\n\n" + ORIGINAL_TEXT,
        ha="left",
        va="top",
        fontsize=8.8,
        family="monospace",
        bbox=box,
        linespacing=1.1,
    )
    ax_text.text(
        0.0,
        0.06,
        "Intervention 1: Answer pointer, causal-model output: beer\n"
        "Intervention 2: Answer payload, causal-model output: tea",
        fontsize=9.2,
        fontstyle="italic",
        color="#555555",
    )

    ax_plot.text(
        0.0,
        1.04,
        "(b) Layer-wise intervention results",
        transform=ax_plot.transAxes,
        fontsize=10,
        fontweight="bold",
    )
    for spec in specs:
        curve = sorted(by_key[spec.key], key=lambda row: row["layer"])
        layers = np.array([row["layer"] for row in curve])
        full = np.array([row["full_residual_accuracy"] for row in curve])
        sub = np.array([row["subspace_accuracy"] for row in curve])
        ax_plot.plot(
            layers,
            full,
            "-",
            color=spec.color,
            linewidth=2.1,
            label=f"{spec.title} full residual",
        )
        ax_plot.plot(
            layers,
            sub,
            "--",
            color=spec.color,
            linewidth=2.1,
            label=f"{spec.title} DCM subspace",
        )

    ax_plot.set_xlabel("Layer")
    ax_plot.set_ylabel("Intervention accuracy")
    ax_plot.set_xlim(0, 47)
    ax_plot.set_ylim(-0.03, 1.03)
    ax_plot.set_xticks([0, 10, 20, 30, 40, 47])
    ax_plot.set_yticks(np.linspace(0, 1, 6))
    ax_plot.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax_plot.legend(frameon=False, fontsize=8.2, loc="upper left")

    fig.suptitle("Answer Lookback Pointer and Payload, Qwen2.5-14B-Instruct", y=0.99, fontsize=12)
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"figure4_qwen_dcm.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a Figure 4-style Answer Lookback DCM plot."
    )
    parser.add_argument("--dcm-root", type=Path, default=DEFAULT_DCM_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--no-verify-svd",
        action="store_true",
        help="Skip loading SVD bases and checking that DCM masks match them.",
    )
    parser.add_argument(
        "--pointer-lambda",
        choices=("0p02", "0p2"),
        default="0p02",
        help="Which pointer DCM run to plot. 0p2 is the original rank-2 peak run.",
    )
    args = parser.parse_args()

    specs = build_specs(args.pointer_lambda)
    all_rows = []
    for spec in specs:
        all_rows.extend(load_curve(args.dcm_root, spec, verify_svd=not args.no_verify_svd))

    write_table(all_rows, args.out_dir)
    plot_figure(all_rows, args.out_dir, specs)
    print(f"Wrote {args.out_dir / 'figure4_qwen_dcm.png'}")
    print(f"Wrote {args.out_dir / 'figure4_qwen_dcm.pdf'}")
    print(f"Wrote {args.out_dir / 'figure4_qwen_dcm_curves.csv'}")
    print(f"Wrote {args.out_dir / 'figure4_qwen_dcm_curves.json'}")


if __name__ == "__main__":
    main()
