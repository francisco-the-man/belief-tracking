import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "lookbacks-dcm"
VOLUME_NAME = "lookbacks-dcm-state"
VOL_MOUNT = Path("/vol")
REPO_DIR = Path("/root/repo")
MODEL_KEY = "Qwen/Qwen2.5-14B-Instruct"
QWEN_14B_LAYERS = list(range(0, 48, 2)) + [47]
ACTIVE_SWEEP_LAYERS = {
    "answer_lookback-pointer": [30, 32, 34, 36, 38, 40],
    "answer_lookback-payload": [40, 42, 44, 46, 47],
}

env_secret = modal.Secret.from_dict(
    {
        "HF_WRITE": os.environ.get("HF_WRITE") or os.environ.get("HF_TOKEN", ""),
        "NDIF_KEY": os.environ.get("NDIF_KEY", ""),
    }
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "accelerate>=1.0.0",
        "dataclasses-json>=0.6.7",
        "datasets>=4.1.1",
        "fire>=0.7.1",
        "huggingface-hub>=0.35.0",
        "matplotlib>=3.9.0",
        "nnsight==0.4.6",
        "numpy>=2.3.3",
        "pandas>=2.2.0",
        "pyyaml>=6.0.2",
        "scikit-learn>=1.5.0",
        "torch==2.8.0",
        "tqdm>=4.67.1",
        "transformers==4.48.3",
    )
    .add_local_dir("scripts", remote_path=str(REPO_DIR / "scripts"), copy=True)
    .add_local_dir("src", remote_path=str(REPO_DIR / "src"), copy=True)
    .add_local_dir("notebooks", remote_path=str(REPO_DIR / "notebooks"), copy=True)
    .add_local_dir("data", remote_path=str(REPO_DIR / "data"), copy=True)
    .workdir(str(REPO_DIR))
    .env(
        {
            "HF_HOME": str(VOL_MOUNT / "hf"),
            "HUGGINGFACE_HUB_CACHE": str(VOL_MOUNT / "hf" / "hub"),
            "TRANSFORMERS_CACHE": str(VOL_MOUNT / "hf"),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

app = modal.App(APP_NAME, image=image)


def _write_env_yml() -> None:
    (REPO_DIR / "env.yml").write_text(
        "\n".join(
            [
                "DATA_DIR: data",
                f"HF_WRITE: {json.dumps(os.environ.get('HF_WRITE', ''))}",
                f"NDIF_KEY: {json.dumps(os.environ.get('NDIF_KEY', ''))}",
                "",
            ]
        )
    )


def _set_repo_link(name: str, target: Path) -> None:
    link = REPO_DIR / name
    target.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.resolve() == target:
            return
        if link.is_dir() and not link.is_symlink():
            return
        link.unlink()
    link.symlink_to(target, target_is_directory=True)


def _ensure_runtime_layout(svd_target: Path | None = None) -> None:
    for path in [
        VOL_MOUNT / "hf",
        VOL_MOUNT / "svd",
        VOL_MOUNT / "svd_runs",
        VOL_MOUNT / "svd_snapshots",
        VOL_MOUNT / "results",
        VOL_MOUNT / "logs",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    _set_repo_link("svd", svd_target or VOL_MOUNT / "svd")
    _set_repo_link("results", VOL_MOUNT / "results")

    _write_env_yml()


def _run(cmd: list[str], label: str) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_DIR, check=True)


def _experiment_svd_root(save_tag: str, experiment: str) -> Path:
    return VOL_MOUNT / "svd_runs" / save_tag / experiment


def _snapshot_svd(
    experiment: str,
    save_tag: str,
    layers: list[int],
    svd_root: Path | None = None,
) -> str:
    src = (svd_root or VOL_MOUNT / "svd") / "causalToM" / "last_token"
    dst = (
        VOL_MOUNT
        / "svd_snapshots"
        / save_tag
        / experiment
        / "causalToM"
        / "last_token"
    )
    if not src.exists():
        raise FileNotFoundError(f"Expected SVD directory does not exist: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "singular_vecs").mkdir(parents=True, exist_ok=True)
    meta_path = src / "svd_meta.json"
    if meta_path.exists():
        shutil.copy2(meta_path, dst / "svd_meta.json")
    for layer in layers:
        layer_path = src / "singular_vecs" / f"{int(layer)}.pt"
        if not layer_path.exists():
            raise FileNotFoundError(f"Expected SVD tensor does not exist: {layer_path}")
        shutil.copy2(layer_path, dst / "singular_vecs" / layer_path.name)
    print(f"Snapshotted SVD basis for {experiment} to {dst}")
    return str(dst)


def _layers_arg(layers: list[int]) -> str:
    return json.dumps([int(layer) for layer in layers])


def _lambda_slug(lamb: float) -> str:
    return f"lamb_{float(lamb):g}".replace("-", "m").replace(".", "p")


def _snapshot_root(save_tag: str, experiment: str) -> Path:
    return VOL_MOUNT / "svd_snapshots" / save_tag / experiment


def _result_path(save_tag: str, experiment: str, layer: int) -> Path:
    return (
        VOL_MOUNT
        / "results"
        / save_tag
        / "causalToM_novis"
        / "Qwen2.5-14B-Instruct"
        / "answer_lookback"
        / experiment.split("-")[1]
        / f"{int(layer)}.json"
    )


def _summarize_results(save_tag: str, experiment: str, layers: list[int]) -> list[dict]:
    rows = []
    for layer in layers:
        path = _result_path(save_tag, experiment, layer)
        result = json.loads(path.read_text())
        rows.append(
            {
                "layer": int(layer),
                "full_rank_acc": result.get("full_rank", {}).get("accuracy"),
                "subspace_acc": result.get("singular_vector", {}).get("accuracy"),
                "rank": result.get("singular_vector", {}).get("rank"),
                "path": str(path),
            }
        )
    return rows


def _load_dcm_json_path(result_tag: str, experiment: str, layer: int) -> Path:
    kind = experiment.split("-")[1]
    return (
        VOL_MOUNT
        / "results"
        / result_tag
        / "causalToM_novis"
        / "Qwen2.5-14B-Instruct"
        / "answer_lookback"
        / kind
        / f"{int(layer)}.json"
    )


def _load_svd_basis_path(source_svd_tag: str, experiment: str, layer: int) -> Path:
    return (
        VOL_MOUNT
        / "svd_snapshots"
        / source_svd_tag
        / experiment
        / "causalToM"
        / "last_token"
        / "singular_vecs"
        / f"{int(layer)}.pt"
    )


def _compute_svd_cmd(
    experiment: str,
    layers: list[int],
    n_samples: int,
    batch_size: int,
) -> list[str]:
    return [
        "python",
        "scripts/compute_svd.py",
        "--experiment",
        experiment,
        "--model_key",
        MODEL_KEY,
        "--layers",
        _layers_arg(layers),
        "--n_samples",
        str(n_samples),
        "--batch_size",
        str(batch_size),
        "--out_prefix",
        str(REPO_DIR),
    ]


def _train_dcm_cmd(
    experiment: str,
    layers: list[int],
    train_size: int,
    validation_size: int,
    batch_size: int,
    learning_rate: float,
    lamb: float,
    n_epochs: int,
    save_path: str,
    verbose: bool,
) -> list[str]:
    return [
        "python",
        "scripts/patching_scripts/run_single_layer_patching_exps.py",
        "--experiment",
        experiment,
        "--model_key",
        MODEL_KEY,
        "--layers",
        _layers_arg(layers),
        "--train_size",
        str(train_size),
        "--validation_size",
        str(validation_size),
        "--batch_size",
        str(batch_size),
        "--learning_rate",
        str(learning_rate),
        "--lamb",
        str(lamb),
        "--n_epochs",
        str(n_epochs),
        "--save_path",
        save_path,
        "--skip_subspace_patching",
        "False",
        "--verbose",
        str(bool(verbose)),
    ]


@app.function(
    volumes={str(VOL_MOUNT): volume},
    secrets=[env_secret],
    timeout=20 * 60,
)
def remote_sanity() -> dict:
    _ensure_runtime_layout()
    import torch
    import transformers

    info = {
        "cwd": str(Path.cwd()),
        "repo_files": sorted(p.name for p in REPO_DIR.iterdir()),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "env_yml_exists": (REPO_DIR / "env.yml").exists(),
        "svd_link": str((REPO_DIR / "svd").resolve()),
        "results_link": str((REPO_DIR / "results").resolve()),
    }
    print(json.dumps(info, indent=2))
    return info


@app.function(
    gpu="H100",
    cpu=8,
    memory=65536,
    ephemeral_disk=524288,
    volumes={str(VOL_MOUNT): volume},
    secrets=[env_secret],
    timeout=24 * 60 * 60,
)
def remote_compute_svd(
    experiment: str,
    layers: list[int],
    n_samples: int = 500,
    batch_size: int = 4,
) -> str:
    _ensure_runtime_layout()
    _run(
        _compute_svd_cmd(experiment, layers, n_samples, batch_size),
        f"compute SVD: {experiment}",
    )
    volume.commit()
    return f"wrote SVD for {experiment} layers={layers}"


@app.function(
    gpu="H100",
    cpu=8,
    memory=65536,
    ephemeral_disk=524288,
    volumes={str(VOL_MOUNT): volume},
    secrets=[env_secret],
    timeout=24 * 60 * 60,
)
def remote_compute_svd_snapshot(
    experiment: str,
    layers: list[int],
    n_samples: int = 80,
    batch_size: int = 4,
    save_tag: str = "svd_n80",
) -> str:
    svd_root = _experiment_svd_root(save_tag, experiment)
    _ensure_runtime_layout(svd_target=svd_root)
    _run(
        _compute_svd_cmd(experiment, layers, n_samples, batch_size),
        f"compute SVD snapshot: {experiment}",
    )
    snapshot_path = _snapshot_svd(experiment, save_tag, layers, svd_root=svd_root)
    volume.commit()
    return (
        f"wrote SVD snapshot for {experiment} layers={layers}; "
        f"svd_root={svd_root}; snapshot={snapshot_path}"
    )


@app.function(
    gpu="H100",
    cpu=8,
    memory=65536,
    ephemeral_disk=524288,
    volumes={str(VOL_MOUNT): volume},
    secrets=[env_secret],
    timeout=24 * 60 * 60,
)
def remote_train_dcm(
    experiment: str,
    layers: list[int],
    train_size: int = 80,
    validation_size: int = 80,
    batch_size: int = 4,
    learning_rate: float = 0.01,
    lamb: float = 0.1,
    n_epochs: int = 1,
    save_tag: str = "paper_hparams",
    verbose: bool = False,
) -> str:
    _ensure_runtime_layout()
    save_path = str(VOL_MOUNT / "results" / save_tag / "causalToM_novis")
    _run(
        _train_dcm_cmd(
            experiment=experiment,
            layers=layers,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lamb=lamb,
            n_epochs=n_epochs,
            save_path=save_path,
            verbose=verbose,
        ),
        f"train DCM: {experiment}",
    )
    snapshot_path = _snapshot_svd(experiment, save_tag, layers)
    volume.commit()
    return (
        f"wrote DCM results for {experiment} layers={layers} under {save_path}; "
        f"SVD snapshot={snapshot_path}"
    )


@app.function(
    gpu="H100",
    cpu=8,
    memory=65536,
    ephemeral_disk=524288,
    volumes={str(VOL_MOUNT): volume},
    secrets=[env_secret],
    timeout=24 * 60 * 60,
)
def remote_full_experiment(
    experiment: str,
    layers: list[int],
    n_samples: int = 500,
    train_size: int = 80,
    validation_size: int = 80,
    batch_size: int = 4,
    learning_rate: float = 0.01,
    lamb: float = 0.1,
    n_epochs: int = 1,
    save_tag: str = "paper_hparams",
    verbose: bool = False,
) -> str:
    svd_root = _experiment_svd_root(save_tag, experiment)
    _ensure_runtime_layout(svd_target=svd_root)
    _run(
        _compute_svd_cmd(experiment, layers, n_samples, batch_size),
        f"compute SVD: {experiment}",
    )
    save_path = str(VOL_MOUNT / "results" / save_tag / "causalToM_novis")
    _run(
        _train_dcm_cmd(
            experiment=experiment,
            layers=layers,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lamb=lamb,
            n_epochs=n_epochs,
            save_path=save_path,
            verbose=verbose,
        ),
        f"train DCM: {experiment}",
    )
    snapshot_path = _snapshot_svd(experiment, save_tag, layers, svd_root=svd_root)
    volume.commit()
    return (
        f"finished {experiment} layers={layers}; results={save_path}; "
        f"svd_root={svd_root}; snapshot={snapshot_path}"
    )


@app.function(
    gpu="H100",
    cpu=8,
    memory=65536,
    ephemeral_disk=524288,
    volumes={str(VOL_MOUNT): volume},
    secrets=[env_secret],
    timeout=24 * 60 * 60,
)
def remote_smoke(
    experiment: str = "answer_lookback-pointer",
    layer: int = 0,
    n_samples: int = 20,
    train_size: int = 4,
    validation_size: int = 4,
    batch_size: int = 4,
) -> dict:
    _ensure_runtime_layout()
    layers = [int(layer)]
    _run(
        _compute_svd_cmd(experiment, layers, n_samples, batch_size),
        f"smoke SVD: {experiment}",
    )
    save_tag = "smoke"
    _run(
        _train_dcm_cmd(
            experiment=experiment,
            layers=layers,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            learning_rate=0.01,
            lamb=0.1,
            n_epochs=1,
            save_path=str(VOL_MOUNT / "results" / save_tag / "causalToM_novis"),
            verbose=False,
        ),
        f"smoke DCM: {experiment}",
    )

    kind = experiment.split("-")[1]
    result_path = (
        VOL_MOUNT
        / "results"
        / save_tag
        / "causalToM_novis"
        / "Qwen2.5-14B-Instruct"
        / "answer_lookback"
        / kind
        / f"{layer}.json"
    )
    result = json.loads(result_path.read_text())
    snapshot_path = _snapshot_svd(experiment, save_tag, layers)
    summary = {
        "result_path": str(result_path),
        "svd_snapshot": snapshot_path,
        "keys": list(result.keys()),
        "full_rank_acc": result.get("full_rank", {}).get("accuracy"),
        "subspace_acc": result.get("singular_vector", {}).get("accuracy"),
        "rank": result.get("singular_vector", {}).get("rank"),
        "has_mask": bool(
            result.get("singular_vector", {})
            .get("metadata", {})
            .get("mask")
        ),
    }
    print(json.dumps(summary, indent=2))
    volume.commit()
    return summary


@app.function(
    gpu="H100",
    cpu=8,
    memory=65536,
    ephemeral_disk=524288,
    volumes={str(VOL_MOUNT): volume},
    secrets=[env_secret],
    timeout=24 * 60 * 60,
)
def remote_sweep_dcm(
    experiment: str,
    layers: list[int],
    lambdas: list[float],
    source_svd_tag: str = "paper_hparams",
    sweep_tag: str = "lambda_sweep",
    train_size: int = 80,
    validation_size: int = 80,
    batch_size: int = 4,
    learning_rate: float = 0.1,
    n_epochs: int = 1,
    verbose: bool = False,
) -> dict:
    svd_root = _snapshot_root(source_svd_tag, experiment)
    if not (svd_root / "causalToM" / "last_token" / "singular_vecs").exists():
        raise FileNotFoundError(
            f"Missing SVD snapshot for {experiment}: {svd_root}. "
            "Run full() or snapshot SVD first."
        )

    _ensure_runtime_layout(svd_target=svd_root)
    summary = {
        "experiment": experiment,
        "layers": [int(layer) for layer in layers],
        "source_svd_tag": source_svd_tag,
        "sweep_tag": sweep_tag,
        "learning_rate": learning_rate,
        "train_size": train_size,
        "validation_size": validation_size,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "lambdas": {},
    }

    for lamb in lambdas:
        lamb = float(lamb)
        save_tag = f"{sweep_tag}/{_lambda_slug(lamb)}"
        save_path = str(VOL_MOUNT / "results" / save_tag / "causalToM_novis")
        _run(
            _train_dcm_cmd(
                experiment=experiment,
                layers=layers,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                learning_rate=learning_rate,
                lamb=lamb,
                n_epochs=n_epochs,
                save_path=save_path,
                verbose=verbose,
            ),
            f"sweep DCM: {experiment} lambda={lamb:g}",
        )
        rows = _summarize_results(save_tag, experiment, layers)
        summary["lambdas"][f"{lamb:g}"] = rows
        print(json.dumps({"lambda": lamb, "rows": rows}, indent=2), flush=True)
        volume.commit()

    return summary


@app.function(
    gpu="H100",
    cpu=16,
    memory=131072,
    ephemeral_disk=524288,
    volumes={str(VOL_MOUNT): volume},
    secrets=[env_secret],
    timeout=24 * 60 * 60,
)
def remote_probe_projection_audit(
    layers: list[int],
    source_svd_tag: str = "svd_n80",
    pointer_result_tag: str = "full_dcm_svd_n80_lr0p1_pointer_lamb0p2/lamb_0p2",
    payload_result_tag: str = "full_dcm_svd_n80_lr0p1_payload_lamb0p1/lamb_0p1",
    out_tag: str = "probe_projection_audit/full_dcm_svd_n80_lr0p1",
    n_samples: int = 500,
    batch_size: int = 8,
    seed: int = 42,
    probe_seed: int = 0,
    pca_dim: int = 200,
) -> dict:
    _ensure_runtime_layout()
    volume.reload()

    import numpy as np
    import torch
    from nnsight import LanguageModel
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from tqdm import tqdm

    from scripts.cache_activations import _build_samples
    from scripts.train_probes import _make_y

    def probe(X: np.ndarray, y: np.ndarray) -> dict:
        n_classes = int(len(np.unique(y)))
        counts = np.bincount(y)
        baseline = float(counts.max() / counts.sum())
        if n_classes < 2 or len(y) < 16 or X.shape[1] < 1:
            return {
                "test_acc": baseline if X.shape[1] < 1 else float("nan"),
                "baseline": baseline,
                "n": int(len(y)),
                "n_classes": n_classes,
                "pca_dim": 0,
                "note": "rank_zero_projection" if X.shape[1] < 1 else "skipped",
            }

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.25, random_state=probe_seed, stratify=y
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.25, random_state=probe_seed
            )

        effective_pca = min(pca_dim, X_tr.shape[0] - 1, X_tr.shape[1])
        solver = "liblinear" if n_classes == 2 else "lbfgs"
        steps = [("scale", StandardScaler(with_mean=True, with_std=True))]
        if effective_pca > 0:
            steps.append(("pca", PCA(n_components=effective_pca, random_state=probe_seed)))
        steps.append(("lr", LogisticRegression(max_iter=1000, C=1.0, solver=solver)))
        pipe = Pipeline(steps)
        pipe.fit(X_tr, y_tr)
        return {
            "test_acc": float(pipe.score(X_te, y_te)),
            "baseline": baseline,
            "n": int(len(y)),
            "n_classes": n_classes,
            "pca_dim": int(effective_pca),
        }

    layers = [int(layer) for layer in layers]
    samples = _build_samples(n_samples=n_samples, template_idx=2, seed=seed)
    prompts = [sample["prompt"] for sample in samples]
    labels = [sample["labels"] for sample in samples]

    lm = LanguageModel(
        MODEL_KEY,
        device_map="auto",
        torch_dtype=torch.float16,
        dispatch=True,
    )
    lm.tokenizer.padding_side = "left"
    if lm.tokenizer.pad_token is None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token

    acts: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    with torch.inference_mode():
        for i in tqdm(range(0, len(prompts), batch_size), desc="cache pos -1"):
            batch = prompts[i : i + batch_size]
            saved = {}
            with lm.trace() as tracer:
                with tracer.invoke(batch):
                    for layer in layers:
                        saved[layer] = (
                            lm.model.layers[layer].output[0][:, -1].clone().save()
                        )
            for layer in layers:
                acts[layer].append(saved[layer].detach().cpu().to(torch.float32))
            torch.cuda.empty_cache()
    acts = {layer: torch.cat(parts, dim=0) for layer, parts in acts.items()}

    specs = [
        {
            "subspace": "pointer_lamb0p2",
            "experiment": "answer_lookback-pointer",
            "result_tag": pointer_result_tag,
            "probe_variable": "state_idx_when_known",
        },
        {
            "subspace": "payload_lamb0p1",
            "experiment": "answer_lookback-payload",
            "result_tag": payload_result_tag,
            "probe_variable": "query_pair",
        },
    ]

    full_cache: dict[tuple[str, int], dict] = {}
    rows = []
    for spec in specs:
        y, keep_idx = _make_y(labels, spec["probe_variable"])
        keep = torch.tensor(keep_idx, dtype=torch.long)
        for layer in tqdm(layers, desc=f"audit {spec['subspace']}"):
            dcm_path = _load_dcm_json_path(spec["result_tag"], spec["experiment"], layer)
            basis_path = _load_svd_basis_path(source_svd_tag, spec["experiment"], layer)
            dcm = json.loads(dcm_path.read_text())
            mask = torch.tensor(
                dcm["singular_vector"]["metadata"]["mask"], dtype=torch.bool
            )
            basis = torch.load(basis_path, map_location="cpu").to(torch.float32)
            selected = basis[mask]
            X = acts[layer][keep]
            full_key = (spec["probe_variable"], layer)
            if full_key not in full_cache:
                full_cache[full_key] = probe(X.numpy(), y)

            if selected.numel() == 0:
                X_proj_features = np.zeros((X.shape[0], 0), dtype=np.float32)
                X_orth = X
            else:
                coords = X @ selected.T
                X_proj_features = coords.numpy()
                X_orth = X - coords @ selected

            row = {
                "subspace": spec["subspace"],
                "experiment": spec["experiment"],
                "anchor_layer": int(layer),
                "probe_variable": spec["probe_variable"],
                "rank": float(dcm["singular_vector"]["rank"]),
                "dcm_full_rank_acc": dcm["full_rank"]["accuracy"],
                "dcm_subspace_acc": dcm["singular_vector"]["accuracy"],
                "probe_full": full_cache[full_key],
                "probe_proj": probe(X_proj_features, y),
                "probe_orth": probe(X_orth.numpy(), y),
                "dcm_path": str(dcm_path),
                "basis_path": str(basis_path),
            }
            rows.append(row)

    out_path = VOL_MOUNT / "results" / out_tag / "probe_projection_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_key": MODEL_KEY,
        "source_svd_tag": source_svd_tag,
        "pointer_result_tag": pointer_result_tag,
        "payload_result_tag": payload_result_tag,
        "n_samples": n_samples,
        "batch_size": batch_size,
        "seed": seed,
        "probe_seed": probe_seed,
        "pca_dim": pca_dim,
        "position": -1,
        "label_mapping": {
            "pointer_lamb0p2": "state_idx_when_known",
            "payload_lamb0p1": "query_pair",
        },
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"out_path": str(out_path), "n_rows": len(rows)}, indent=2))
    volume.commit()
    return payload


@app.function(
    gpu="H100",
    cpu=16,
    memory=131072,
    ephemeral_disk=524288,
    volumes={str(VOL_MOUNT): volume},
    secrets=[env_secret],
    timeout=24 * 60 * 60,
)
def remote_dcm_pca_clusters(
    pointer_layers: list[int],
    payload_layers: list[int],
    source_svd_tag: str = "svd_n80",
    pointer_result_tag: str = "full_dcm_svd_n80_lr0p1_pointer_lamb0p02/lamb_0p02",
    payload_result_tag: str = "full_dcm_svd_n80_lr0p1_payload_lamb0p1/lamb_0p1",
    out_tag: str = "dcm_pca_clusters/full_dcm_svd_n80_lr0p1",
    n_samples: int = 400,
    batch_size: int = 4,
    seed: int = 42,
    pca_components: int = 10,
    cluster_ks: list[int] | None = None,
) -> dict:
    _ensure_runtime_layout()
    volume.reload()

    import numpy as np
    import pandas as pd
    import torch
    from matplotlib import pyplot as plt
    from nnsight import LanguageModel
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
    from sklearn.preprocessing import StandardScaler
    from tqdm import tqdm

    from scripts.cache_activations import _build_samples

    pointer_layers = [int(layer) for layer in pointer_layers]
    payload_layers = [int(layer) for layer in payload_layers]
    layers = sorted(set(pointer_layers + payload_layers))
    cluster_ks = [int(k) for k in (cluster_ks or [2, 3, 4, 5, 6, 7, 8])]

    samples = _build_samples(n_samples=n_samples, template_idx=2, seed=seed)
    prompts = [sample["prompt"] for sample in samples]
    labels = [sample["labels"] for sample in samples]

    lm = LanguageModel(
        MODEL_KEY,
        device_map="auto",
        torch_dtype=torch.float16,
        dispatch=True,
    )
    lm.tokenizer.padding_side = "left"
    if lm.tokenizer.pad_token is None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token

    acts: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    with torch.inference_mode():
        for i in tqdm(range(0, len(prompts), batch_size), desc="cache DCM PCA acts"):
            batch = prompts[i : i + batch_size]
            saved: dict[int, torch.Tensor] = {}
            with lm.trace() as tracer:
                with tracer.invoke(batch):
                    for layer in layers:
                        saved[layer] = (
                            lm.model.layers[layer].output[0][:, -1].clone().save()
                        )
            for layer in layers:
                acts[layer].append(saved[layer].detach().cpu().to(torch.float32))
            torch.cuda.empty_cache()
    acts_cat = {layer: torch.cat(parts, dim=0) for layer, parts in acts.items()}

    feature_sets: dict[str, list[np.ndarray]] = {
        "pointer": [],
        "payload": [],
        "both": [],
    }
    feature_meta = []

    def append_dcm_coords(
        feature_set: str,
        experiment: str,
        result_tag: str,
        layer: int,
        prefix: str,
    ) -> None:
        dcm_path = _load_dcm_json_path(result_tag, experiment, layer)
        basis_path = _load_svd_basis_path(source_svd_tag, experiment, layer)
        dcm = json.loads(dcm_path.read_text())
        mask = torch.tensor(
            dcm["singular_vector"]["metadata"]["mask"], dtype=torch.bool
        )
        basis = torch.load(basis_path, map_location="cpu").to(torch.float32)
        selected = basis[mask]
        coords = acts_cat[layer] @ selected.T
        arr = coords.numpy()
        feature_sets[feature_set].append(arr)
        feature_sets["both"].append(arr)
        feature_meta.append(
            {
                "feature_set": feature_set,
                "experiment": experiment,
                "layer": int(layer),
                "rank": int(selected.shape[0]),
                "dcm_subspace_acc": float(dcm["singular_vector"]["accuracy"]),
                "dcm_full_rank_acc": float(dcm["full_rank"]["accuracy"]),
                "feature_prefix": prefix,
                "dcm_path": str(dcm_path),
                "basis_path": str(basis_path),
            }
        )

    for layer in tqdm(pointer_layers, desc="project pointer DCMs"):
        append_dcm_coords(
            "pointer",
            "answer_lookback-pointer",
            pointer_result_tag,
            layer,
            f"pointer_L{layer}",
        )
    for layer in tqdm(payload_layers, desc="project payload DCMs"):
        append_dcm_coords(
            "payload",
            "answer_lookback-payload",
            payload_result_tag,
            layer,
            f"payload_L{layer}",
        )

    label_rows = []
    for i, label in enumerate(labels):
        char_idx = int(label["char_idx"])
        obj_idx = int(label["obj_idx"])
        query_pair = char_idx * 2 + obj_idx
        prompt = prompts[i]
        question = prompt.split("Question:")[-1].split("Answer:")[0].strip()
        label_rows.append(
            {
                "sample_idx": i,
                "char_idx": char_idx,
                "obj_idx": obj_idx,
                "query_pair": query_pair,
                "is_unknown": int(label["is_unknown"]),
                "state_idx": int(label["state_idx"]),
                "target_state": label["target_state"],
                "queried_character": label["characters"][char_idx],
                "queried_object": label["objects"][obj_idx],
                "question": question,
            }
        )
    label_df = pd.DataFrame(label_rows)

    out_dir = VOL_MOUNT / "results" / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    label_names = ["char_idx", "obj_idx", "query_pair", "is_unknown", "state_idx"]
    analyses = {}
    for set_name, blocks in feature_sets.items():
        if not blocks:
            continue
        set_dir = out_dir / set_name
        set_dir.mkdir(parents=True, exist_ok=True)
        X = np.concatenate(blocks, axis=1).astype(np.float32)
        np.save(set_dir / "dcm_features.npy", X)

        scaler = StandardScaler(with_mean=True, with_std=True)
        X_scaled = scaler.fit_transform(X)
        n_pca = max(1, min(int(pca_components), X_scaled.shape[0] - 1, X_scaled.shape[1]))
        pca = PCA(n_components=n_pca, random_state=seed)
        X_pca = pca.fit_transform(X_scaled)

        pca_df = label_df.copy()
        for pc_i in range(n_pca):
            pca_df[f"pc{pc_i + 1}"] = X_pca[:, pc_i]

        k_rows = []
        best = None
        for k in cluster_ks:
            if k < 2 or k >= X_pca.shape[0]:
                continue
            km = KMeans(n_clusters=k, random_state=seed, n_init=20)
            cluster = km.fit_predict(X_pca)
            sil = float(silhouette_score(X_pca, cluster))
            ami = {
                name: float(adjusted_mutual_info_score(label_df[name], cluster))
                for name in label_names
            }
            row = {"k": int(k), "silhouette": sil, "ami": ami}
            k_rows.append(row)
            if best is None or sil > best["silhouette"]:
                best = {"k": int(k), "silhouette": sil, "cluster": cluster, "ami": ami}

        if best is None:
            best = {
                "k": 1,
                "silhouette": float("nan"),
                "cluster": np.zeros(X_pca.shape[0], dtype=int),
                "ami": {name: float("nan") for name in label_names},
            }

        pca_df["cluster"] = best["cluster"]
        pca_df.to_csv(set_dir / "pca_points.csv", index=False)
        pca_df.to_csv(set_dir / "cluster_assignments.csv", index=False)

        cluster_rows = []
        for cluster_id, group in pca_df.groupby("cluster"):
            distributions = {}
            for name in label_names + ["target_state"]:
                distributions[name] = {
                    str(k): int(v)
                    for k, v in group[name].value_counts().head(10).items()
                }
            examples = (
                group[
                    [
                        "sample_idx",
                        "question",
                        "queried_character",
                        "queried_object",
                        "target_state",
                    ]
                ]
                .head(5)
                .to_dict(orient="records")
            )
            cluster_rows.append(
                {
                    "cluster": int(cluster_id),
                    "n": int(len(group)),
                    "distributions": distributions,
                    "examples": examples,
                }
            )

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1] if n_pca > 1 else np.zeros(X_pca.shape[0]),
            c=best["cluster"],
            cmap="tab10",
            s=28,
            alpha=0.85,
        )
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
        pc2_var = pca.explained_variance_ratio_[1] if n_pca > 1 else 0.0
        plt.ylabel(f"PC2 ({pc2_var:.2%})")
        plt.title(f"{set_name} DCM coordinates | KMeans k={best['k']}")
        plt.colorbar(scatter, label="cluster")
        plt.tight_layout()
        plt.savefig(set_dir / "pca_clusters.png", dpi=180)
        plt.close()

        analyses[set_name] = {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "pca_components": int(n_pca),
            "explained_variance_ratio": [
                float(v) for v in pca.explained_variance_ratio_.tolist()
            ],
            "k_sweep": k_rows,
            "best_k": int(best["k"]),
            "best_silhouette": float(best["silhouette"]),
            "best_cluster_ami": best["ami"],
            "clusters": cluster_rows,
            "paths": {
                "features": str(set_dir / "dcm_features.npy"),
                "pca_points": str(set_dir / "pca_points.csv"),
                "cluster_assignments": str(set_dir / "cluster_assignments.csv"),
                "plot": str(set_dir / "pca_clusters.png"),
            },
        }

    payload = {
        "model_key": MODEL_KEY,
        "source_svd_tag": source_svd_tag,
        "pointer_result_tag": pointer_result_tag,
        "payload_result_tag": payload_result_tag,
        "pointer_layers": pointer_layers,
        "payload_layers": payload_layers,
        "n_samples": len(samples),
        "batch_size": batch_size,
        "seed": seed,
        "position": -1,
        "feature_meta": feature_meta,
        "analyses": analyses,
    }
    (out_dir / "cluster_summary.json").write_text(json.dumps(payload, indent=2))
    label_df.to_csv(out_dir / "sample_labels.csv", index=False)
    print(json.dumps({"out_dir": str(out_dir), "analyses": analyses}, indent=2))
    volume.commit()
    return payload


@app.function(
    gpu="H100",
    cpu=8,
    memory=65536,
    ephemeral_disk=524288,
    volumes={str(VOL_MOUNT): volume},
    secrets=[env_secret],
    timeout=24 * 60 * 60,
)
def remote_causal_complement_audit(
    experiment: str,
    layers: list[int],
    source_svd_tag: str = "svd_n80",
    result_tag: str = "full_dcm_svd_n80_lr0p1_pointer_lamb0p2/lamb_0p2",
    out_tag: str = "causal_complement_audit/full_dcm_svd_n80_lr0p1",
    train_size: int = 80,
    validation_size: int = 80,
    batch_size: int = 4,
    seed: int = 0,
) -> dict:
    _ensure_runtime_layout()
    volume.reload()

    import sys

    import torch
    from nnsight import LanguageModel
    from tqdm import tqdm

    patching_dir = REPO_DIR / "scripts" / "patching_scripts"
    if str(patching_dir) not in sys.path:
        sys.path.insert(0, str(patching_dir))
    from run_patching_exp_utils import free_gpu_cache, prepare_dataset, set_seed
    from run_single_layer_patching_exps import validate

    set_seed(seed)
    layers = [int(layer) for layer in layers]
    lm = LanguageModel(
        MODEL_KEY,
        device_map="auto",
        torch_dtype=torch.float16,
        dispatch=True,
    )
    lm.tokenizer.padding_side = "left"
    if lm.tokenizer.pad_token is None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token

    _, valid_loader = prepare_dataset(
        lm=lm,
        experiment_name=experiment,
        train_size=train_size,
        valid_size=validation_size,
        batch_size=batch_size,
        remote=False,
    )

    rows = []
    for layer in tqdm(layers, desc=f"causal complement {experiment}"):
        dcm_path = _load_dcm_json_path(result_tag, experiment, layer)
        basis_path = _load_svd_basis_path(source_svd_tag, experiment, layer)
        dcm = json.loads(dcm_path.read_text())
        mask = torch.tensor(
            dcm["singular_vector"]["metadata"]["mask"], dtype=torch.bool
        )
        basis = torch.load(basis_path, map_location="cpu").to(
            device="cuda", dtype=torch.float16
        )
        selected = basis[mask]
        hidden_dim = int(basis.shape[1])
        complement = torch.eye(hidden_dim, device="cuda", dtype=torch.float16)
        if selected.numel() > 0:
            projection = selected.T @ selected
            complement.sub_(projection)
            del projection

        complement_acc = validate(
            exp_name=experiment,
            lm=lm,
            layer_idx=layer,
            validation_loader=valid_loader,
            projection=complement,
            verbose=False,
            save_outputs=False,
            projection_type="complement",
            remote=False,
            bigtom=False,
        )

        row = {
            "experiment": experiment,
            "anchor_layer": int(layer),
            "rank": float(dcm["singular_vector"]["rank"]),
            "dcm_full_iia": dcm["full_rank"]["accuracy"],
            "dcm_sub_iia": dcm["singular_vector"]["accuracy"],
            "dcm_complement_iia": complement_acc,
            "dcm_path": str(dcm_path),
            "basis_path": str(basis_path),
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

        del basis, selected, complement
        free_gpu_cache()

    out_path = (
        VOL_MOUNT
        / "results"
        / out_tag
        / experiment
        / "causal_complement_audit.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_key": MODEL_KEY,
        "experiment": experiment,
        "source_svd_tag": source_svd_tag,
        "result_tag": result_tag,
        "train_size": train_size,
        "validation_size": validation_size,
        "batch_size": batch_size,
        "seed": seed,
        "position": -1,
        "intervention_modes": {
            "v_only": "dcm_sub_iia",
            "v_orth_only": "dcm_complement_iia",
            "full_residual": "dcm_full_iia",
        },
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"out_path": str(out_path), "n_rows": len(rows)}, indent=2))
    volume.commit()
    return payload


@app.function(
    gpu="H100",
    cpu=16,
    memory=131072,
    ephemeral_disk=524288,
    volumes={str(VOL_MOUNT): volume},
    secrets=[env_secret],
    timeout=24 * 60 * 60,
)
def remote_payload_target_probe_audit(
    layers: list[int],
    source_svd_tag: str = "svd_n80",
    payload_result_tag: str = "full_dcm_svd_n80_lr0p1_payload_lamb0p1/lamb_0p1",
    out_tag: str = "payload_target_probe_audit/full_dcm_svd_n80_lr0p1",
    n_samples: int = 500,
    batch_size: int = 8,
    seed: int = 42,
    probe_seed: int = 0,
    pca_dim: int = 200,
    target_top_k: int = 12,
    target_min_count: int = 4,
) -> dict:
    _ensure_runtime_layout()
    volume.reload()

    from collections import Counter

    import numpy as np
    import torch
    from nnsight import LanguageModel
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from tqdm import tqdm

    from scripts.cache_activations import _build_samples
    from scripts.train_probes import _make_y

    def probe(X: np.ndarray, y: np.ndarray) -> dict:
        n_classes = int(len(np.unique(y)))
        counts = np.bincount(y)
        baseline = float(counts.max() / counts.sum())
        if n_classes < 2 or len(y) < 16 or X.shape[1] < 1:
            return {
                "test_acc": baseline if X.shape[1] < 1 else float("nan"),
                "baseline": baseline,
                "n": int(len(y)),
                "n_classes": n_classes,
                "pca_dim": 0,
                "note": "rank_zero_projection" if X.shape[1] < 1 else "skipped",
            }

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.25, random_state=probe_seed, stratify=y
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.25, random_state=probe_seed
            )

        effective_pca = min(pca_dim, X_tr.shape[0] - 1, X_tr.shape[1])
        solver = "liblinear" if n_classes == 2 else "lbfgs"
        steps = [("scale", StandardScaler(with_mean=True, with_std=True))]
        if effective_pca > 0:
            steps.append(("pca", PCA(n_components=effective_pca, random_state=probe_seed)))
        steps.append(("lr", LogisticRegression(max_iter=1000, C=1.0, solver=solver)))
        pipe = Pipeline(steps)
        pipe.fit(X_tr, y_tr)
        return {
            "test_acc": float(pipe.score(X_te, y_te)),
            "baseline": baseline,
            "n": int(len(y)),
            "n_classes": n_classes,
            "pca_dim": int(effective_pca),
            "class_counts": counts.tolist(),
        }

    def make_target_state_y(labels: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
        known = [
            str(label["target_state"])
            for label in labels
            if str(label["target_state"]) != "unknown"
        ]
        counts = Counter(known)
        eligible = [
            target
            for target, count in counts.most_common()
            if count >= int(target_min_count)
        ][: int(target_top_k)]
        mapping = {target: idx for idx, target in enumerate(eligible)}
        y: list[int] = []
        keep: list[int] = []
        for i, label in enumerate(labels):
            target = str(label["target_state"])
            if target not in mapping:
                continue
            keep.append(i)
            y.append(mapping[target])
        return (
            np.array(y, dtype=np.int64),
            np.array(keep, dtype=np.int64),
            {
                "target_to_class": mapping,
                "class_to_target": {str(v): k for k, v in mapping.items()},
                "all_known_target_counts": dict(counts.most_common()),
                "target_top_k": int(target_top_k),
                "target_min_count": int(target_min_count),
            },
        )

    layers = [int(layer) for layer in layers]
    samples = _build_samples(n_samples=n_samples, template_idx=2, seed=seed)
    prompts = [sample["prompt"] for sample in samples]
    labels = [sample["labels"] for sample in samples]

    lm = LanguageModel(
        MODEL_KEY,
        device_map="auto",
        torch_dtype=torch.float16,
        dispatch=True,
    )
    lm.tokenizer.padding_side = "left"
    if lm.tokenizer.pad_token is None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token

    acts: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    with torch.inference_mode():
        for i in tqdm(range(0, len(prompts), batch_size), desc="cache payload probe acts"):
            batch = prompts[i : i + batch_size]
            saved = {}
            with lm.trace() as tracer:
                with tracer.invoke(batch):
                    for layer in layers:
                        saved[layer] = (
                            lm.model.layers[layer].output[0][:, -1].clone().save()
                        )
            for layer in layers:
                acts[layer].append(saved[layer].detach().cpu().to(torch.float32))
            torch.cuda.empty_cache()
    acts = {layer: torch.cat(parts, dim=0) for layer, parts in acts.items()}

    y_state_idx, keep_state_idx = _make_y(labels, "state_idx_when_known")
    y_target, keep_target, target_meta = make_target_state_y(labels)
    variables = [
        {
            "probe_variable": "state_idx_when_known",
            "y": y_state_idx,
            "keep": keep_state_idx,
            "metadata": {},
        },
        {
            "probe_variable": "target_state_known_topk",
            "y": y_target,
            "keep": keep_target,
            "metadata": target_meta,
        },
    ]

    selected_by_layer: dict[int, torch.Tensor] = {}
    dcm_by_layer: dict[int, dict] = {}
    basis_paths: dict[int, str] = {}
    dcm_paths: dict[int, str] = {}
    for layer in layers:
        dcm_path = _load_dcm_json_path(
            payload_result_tag, "answer_lookback-payload", layer
        )
        basis_path = _load_svd_basis_path(
            source_svd_tag, "answer_lookback-payload", layer
        )
        dcm = json.loads(dcm_path.read_text())
        mask = torch.tensor(
            dcm["singular_vector"]["metadata"]["mask"], dtype=torch.bool
        )
        basis = torch.load(basis_path, map_location="cpu").to(torch.float32)
        selected_by_layer[layer] = basis[mask]
        dcm_by_layer[layer] = dcm
        dcm_paths[layer] = str(dcm_path)
        basis_paths[layer] = str(basis_path)

    rows = []
    for var in variables:
        y = var["y"]
        keep = torch.tensor(var["keep"], dtype=torch.long)
        concat_full: list[np.ndarray] = []
        concat_proj: list[np.ndarray] = []
        concat_orth: list[np.ndarray] = []

        for layer in tqdm(layers, desc=f"payload target probe {var['probe_variable']}"):
            selected = selected_by_layer[layer]
            dcm = dcm_by_layer[layer]
            X = acts[layer][keep]
            if selected.numel() == 0:
                X_proj_features = np.zeros((X.shape[0], 0), dtype=np.float32)
                X_orth = X
            else:
                coords = X @ selected.T
                X_proj_features = coords.numpy()
                X_orth = X - coords @ selected
            full_np = X.numpy()
            orth_np = X_orth.numpy()
            concat_full.append(full_np)
            concat_proj.append(X_proj_features)
            concat_orth.append(orth_np)

            row = {
                "subspace": "payload_lamb0p1",
                "experiment": "answer_lookback-payload",
                "anchor_layer": int(layer),
                "probe_variable": var["probe_variable"],
                "rank": float(dcm["singular_vector"]["rank"]),
                "dcm_full_rank_acc": dcm["full_rank"]["accuracy"],
                "dcm_subspace_acc": dcm["singular_vector"]["accuracy"],
                "probe_full": probe(full_np, y),
                "probe_proj": probe(X_proj_features, y),
                "probe_orth": probe(orth_np, y),
                "dcm_path": dcm_paths[layer],
                "basis_path": basis_paths[layer],
            }
            rows.append(row)

        rows.append(
            {
                "subspace": "payload_lamb0p1",
                "experiment": "answer_lookback-payload",
                "anchor_layer": "concat",
                "probe_variable": var["probe_variable"],
                "rank": int(sum(selected_by_layer[layer].shape[0] for layer in layers)),
                "layers": layers,
                "probe_full": probe(np.concatenate(concat_full, axis=1), y),
                "probe_proj": probe(np.concatenate(concat_proj, axis=1), y),
                "probe_orth": probe(np.concatenate(concat_orth, axis=1), y),
            }
        )

    out_path = VOL_MOUNT / "results" / out_tag / "payload_target_probe_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_key": MODEL_KEY,
        "source_svd_tag": source_svd_tag,
        "payload_result_tag": payload_result_tag,
        "n_samples": n_samples,
        "batch_size": batch_size,
        "seed": seed,
        "probe_seed": probe_seed,
        "pca_dim": pca_dim,
        "position": -1,
        "target_state_metadata": target_meta,
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"out_path": str(out_path), "n_rows": len(rows)}, indent=2))
    volume.commit()
    return payload


@app.function(
    gpu="H100",
    cpu=8,
    memory=65536,
    ephemeral_disk=524288,
    volumes={str(VOL_MOUNT): volume},
    secrets=[env_secret],
    timeout=24 * 60 * 60,
)
def remote_payload_grouped_patching_audit(
    layers: list[int],
    source_svd_tag: str = "svd_n80",
    payload_result_tag: str = "full_dcm_svd_n80_lr0p1_payload_lamb0p1/lamb_0p1",
    out_tag: str = "payload_grouped_patching_audit/full_dcm_svd_n80_lr0p1",
    train_size: int = 80,
    validation_size: int = 160,
    batch_size: int = 4,
    seed: int = 0,
) -> dict:
    _ensure_runtime_layout()
    volume.reload()

    import sys
    from collections import defaultdict

    import torch
    from nnsight import LanguageModel
    from tqdm import tqdm

    patching_dir = REPO_DIR / "scripts" / "patching_scripts"
    if str(patching_dir) not in sys.path:
        sys.path.insert(0, str(patching_dir))
    from run_patching_exp_utils import (
        exp_to_intervention_positions,
        free_gpu_cache,
        prepare_dataset,
        set_seed,
    )

    set_seed(seed)
    layers = [int(layer) for layer in layers]
    experiment = "answer_lookback-payload"
    lm = LanguageModel(
        MODEL_KEY,
        device_map="auto",
        torch_dtype=torch.float16,
        dispatch=True,
    )
    lm.tokenizer.padding_side = "left"
    if lm.tokenizer.pad_token is None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token

    _, valid_loader = prepare_dataset(
        lm=lm,
        experiment_name=experiment,
        train_size=train_size,
        valid_size=validation_size,
        batch_size=batch_size,
        remote=False,
    )

    intervention_positions = exp_to_intervention_positions[experiment].copy()
    patch_to_cache_map = {
        k: v
        for k, v in zip(
            intervention_positions["patch"], intervention_positions["cache"]
        )
    }

    def summarize(samples: list[dict]) -> dict:
        by_mode: dict[str, dict] = {}
        for mode in ["full_residual", "payload_only", "payload_complement"]:
            mode_samples = [sample for sample in samples if sample["mode"] == mode]
            correct = sum(int(sample["correct"]) for sample in mode_samples)
            by_target: dict[str, dict] = {}
            grouped: defaultdict[str, list[dict]] = defaultdict(list)
            for sample in mode_samples:
                grouped[sample["target"]].append(sample)
            for target, target_samples in sorted(grouped.items()):
                target_correct = sum(int(sample["correct"]) for sample in target_samples)
                by_target[target] = {
                    "n": len(target_samples),
                    "accuracy": target_correct / len(target_samples),
                    "pred_counts": dict(
                        sorted(
                            {
                                pred: sum(
                                    int(sample["prediction"] == pred)
                                    for sample in target_samples
                                )
                                for pred in {sample["prediction"] for sample in target_samples}
                            }.items()
                        )
                    ),
                }
            by_mode[mode] = {
                "n": len(mode_samples),
                "accuracy": correct / len(mode_samples) if mode_samples else float("nan"),
                "by_target": by_target,
            }
        return by_mode

    rows = []
    for layer in tqdm(layers, desc="payload grouped patching"):
        dcm_path = _load_dcm_json_path(payload_result_tag, experiment, layer)
        basis_path = _load_svd_basis_path(source_svd_tag, experiment, layer)
        dcm = json.loads(dcm_path.read_text())
        mask = torch.tensor(
            dcm["singular_vector"]["metadata"]["mask"], dtype=torch.bool
        )
        basis = torch.load(basis_path, map_location="cpu").to(
            device="cuda", dtype=torch.float16
        )
        selected = basis[mask]
        hidden_dim = int(basis.shape[1])
        eye = torch.eye(hidden_dim, device="cuda", dtype=torch.float16)
        projection = selected.T @ selected if selected.numel() else torch.zeros_like(eye)
        complement = eye - projection
        projections = {
            "full_residual": None,
            "payload_only": projection,
            "payload_complement": complement,
        }

        layer_samples: list[dict] = []
        for mode, proj in projections.items():
            for batch_idx, batch in tqdm(
                enumerate(valid_loader),
                total=len(valid_loader),
                desc=f"L{layer} {mode}",
                leave=False,
            ):
                alt_prompts = list(batch["counterfactual_prompt"])
                org_prompts = list(batch["clean_prompt"])
                targets = (
                    list(batch["target"])
                    if "target" in batch
                    else list(batch["counterfactual_target"])
                )
                alt_acts = {}
                with lm.trace() as tracer:
                    with tracer.invoke(alt_prompts):
                        for t in intervention_positions["cache"]:
                            alt_acts[t] = (
                                lm.model.layers[layer].output[0][:, t].clone().save()
                            )
                    with tracer.invoke(org_prompts):
                        for t in intervention_positions["patch"]:
                            curr_output = lm.model.layers[layer].output[0][:, t].clone()
                            if proj is None:
                                patch = alt_acts[patch_to_cache_map[t]]
                            else:
                                alt_proj = torch.matmul(
                                    alt_acts[patch_to_cache_map[t]], proj
                                )
                                org_proj = torch.matmul(curr_output, proj)
                                patch = curr_output - org_proj + alt_proj
                            lm.model.layers[layer].output[0][:, t] = patch
                        pred = torch.argmax(lm.lm_head.output[:, -1], dim=-1).save()

                pred = pred.detach().cpu()
                for i, target in enumerate(targets):
                    prediction = lm.tokenizer.decode(pred[i]).lower().strip()
                    target_norm = str(target).lower().strip()
                    layer_samples.append(
                        {
                            "layer": int(layer),
                            "batch_idx": int(batch_idx),
                            "sample_in_batch": int(i),
                            "mode": mode,
                            "target": target_norm,
                            "prediction": prediction,
                            "correct": bool(prediction == target_norm),
                        }
                    )
                del alt_acts, pred
                free_gpu_cache()

        row = {
            "experiment": experiment,
            "anchor_layer": int(layer),
            "rank": float(dcm["singular_vector"]["rank"]),
            "dcm_full_iia": dcm["full_rank"]["accuracy"],
            "dcm_sub_iia": dcm["singular_vector"]["accuracy"],
            "dcm_path": str(dcm_path),
            "basis_path": str(basis_path),
            "summary": summarize(layer_samples),
            "samples": layer_samples,
        }
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "samples"}, indent=2))
        del basis, selected, eye, projection, complement, projections
        free_gpu_cache()

    out_path = VOL_MOUNT / "results" / out_tag / "payload_grouped_patching_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_key": MODEL_KEY,
        "experiment": experiment,
        "source_svd_tag": source_svd_tag,
        "payload_result_tag": payload_result_tag,
        "train_size": train_size,
        "validation_size": validation_size,
        "batch_size": batch_size,
        "seed": seed,
        "position": -1,
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"out_path": str(out_path), "n_rows": len(rows)}, indent=2))
    volume.commit()
    return payload


@app.function(
    gpu="H100",
    cpu=8,
    memory=65536,
    ephemeral_disk=524288,
    volumes={str(VOL_MOUNT): volume},
    secrets=[env_secret],
    timeout=24 * 60 * 60,
)
def remote_payload_token_geometry(
    layers: list[int],
    source_svd_tag: str = "svd_n80",
    payload_result_tag: str = "full_dcm_svd_n80_lr0p1_payload_lamb0p1/lamb_0p1",
    grouped_patching_tag: str = (
        "payload_grouped_patching_audit/"
        "full_dcm_svd_n80_lr0p1_v240_payload_layers"
    ),
    out_tag: str = "payload_token_geometry/full_dcm_svd_n80_lr0p1",
) -> dict:
    _ensure_runtime_layout()
    volume.reload()

    import math

    import torch
    from nnsight import LanguageModel

    layers = [int(layer) for layer in layers]
    grouped_path = (
        VOL_MOUNT
        / "results"
        / grouped_patching_tag
        / "payload_grouped_patching_audit.json"
    )
    grouped = json.loads(grouped_path.read_text())
    targets = sorted(
        {
            target
            for row in grouped["rows"]
            for target in row["summary"]["payload_only"]["by_target"]
        }
    )

    lm = LanguageModel(
        MODEL_KEY,
        device_map="auto",
        torch_dtype=torch.float16,
        dispatch=True,
    )
    lm.tokenizer.padding_side = "left"
    if lm.tokenizer.pad_token is None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token

    unembed = lm.lm_head.weight.detach().to(device="cuda", dtype=torch.float32)
    unembed_norm = unembed.norm(dim=1).clamp_min(1e-8)

    selected_by_layer: dict[int, torch.Tensor] = {}
    for layer in layers:
        dcm_path = _load_dcm_json_path(
            payload_result_tag, "answer_lookback-payload", layer
        )
        basis_path = _load_svd_basis_path(
            source_svd_tag, "answer_lookback-payload", layer
        )
        dcm = json.loads(dcm_path.read_text())
        mask = torch.tensor(
            dcm["singular_vector"]["metadata"]["mask"], dtype=torch.bool
        )
        basis = torch.load(basis_path, map_location="cpu").to(
            device="cuda", dtype=torch.float32
        )
        selected_by_layer[layer] = basis[mask].contiguous()

    patch_acc = {
        int(row["anchor_layer"]): {
            target: stats["accuracy"]
            for target, stats in row["summary"]["payload_only"]["by_target"].items()
        }
        for row in grouped["rows"]
    }

    rows = []
    for target in targets:
        bare_ids = lm.tokenizer(
            target, add_special_tokens=False
        ).input_ids
        space_ids = lm.tokenizer(
            " " + target, add_special_tokens=False
        ).input_ids
        first_id = int(bare_ids[0])
        first_space_id = int(space_ids[0])
        token_text = lm.tokenizer.decode([first_id])
        space_token_text = lm.tokenizer.decode([first_space_id])

        for layer in layers:
            selected = selected_by_layer[layer]
            vec = unembed[first_id]
            vec_unit = vec / unembed_norm[first_id]
            if selected.numel() == 0:
                projection_norm = 0.0
                projection_fraction = 0.0
                max_abs_cos = 0.0
                mean_abs_cos = 0.0
            else:
                coords = selected @ vec
                projection_norm = float(coords.norm().detach().cpu())
                projection_fraction = float(
                    (coords.norm() / unembed_norm[first_id]).detach().cpu()
                )
                cosines = selected @ vec_unit
                max_abs_cos = float(cosines.abs().max().detach().cpu())
                mean_abs_cos = float(cosines.abs().mean().detach().cpu())

            rows.append(
                {
                    "target": target,
                    "layer": int(layer),
                    "payload_only_accuracy": float(patch_acc[layer][target]),
                    "bare_token_ids": [int(x) for x in bare_ids],
                    "space_token_ids": [int(x) for x in space_ids],
                    "bare_token_len": int(len(bare_ids)),
                    "space_token_len": int(len(space_ids)),
                    "first_token_id": first_id,
                    "first_token": token_text,
                    "first_space_token_id": first_space_id,
                    "first_space_token": space_token_text,
                    "unembed_norm": float(unembed_norm[first_id].detach().cpu()),
                    "payload_readout_projection_norm": projection_norm,
                    "payload_readout_projection_fraction": projection_fraction,
                    "payload_readout_max_abs_cos": max_abs_cos,
                    "payload_readout_mean_abs_cos": mean_abs_cos,
                    "log_unembed_norm": math.log(
                        float(unembed_norm[first_id].detach().cpu())
                    ),
                }
            )

    def pearson(xs: list[float], ys: list[float]) -> float | None:
        if len(xs) < 3:
            return None
        x = torch.tensor(xs, dtype=torch.float64)
        y = torch.tensor(ys, dtype=torch.float64)
        x = x - x.mean()
        y = y - y.mean()
        denom = x.norm() * y.norm()
        if float(denom) == 0.0:
            return None
        return float((x @ y / denom).item())

    correlations = {}
    for layer in layers:
        layer_rows = [row for row in rows if row["layer"] == layer]
        acc = [row["payload_only_accuracy"] for row in layer_rows]
        correlations[str(layer)] = {
            "n_targets": len(layer_rows),
            "projection_fraction": pearson(
                [row["payload_readout_projection_fraction"] for row in layer_rows],
                acc,
            ),
            "max_abs_cos": pearson(
                [row["payload_readout_max_abs_cos"] for row in layer_rows],
                acc,
            ),
            "unembed_norm": pearson(
                [row["unembed_norm"] for row in layer_rows],
                acc,
            ),
            "bare_token_len": pearson(
                [float(row["bare_token_len"]) for row in layer_rows],
                acc,
            ),
        }

    out_path = VOL_MOUNT / "results" / out_tag / "payload_token_geometry.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_key": MODEL_KEY,
        "source_svd_tag": source_svd_tag,
        "payload_result_tag": payload_result_tag,
        "grouped_patching_tag": grouped_patching_tag,
        "layers": layers,
        "rows": rows,
        "correlations": correlations,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"out_path": str(out_path), "n_rows": len(rows)}, indent=2))
    volume.commit()
    return payload


@app.function(
    gpu="H100",
    cpu=8,
    memory=65536,
    ephemeral_disk=524288,
    volumes={str(VOL_MOUNT): volume},
    secrets=[env_secret],
    timeout=24 * 60 * 60,
)
def remote_pointer_payload_cross_patch(
    pointer_layer: int = 32,
    payload_layer: int = 46,
    source_svd_tag: str = "svd_n80",
    pointer_result_tag: str = "full_dcm_svd_n80_lr0p1_pointer_lamb0p02/lamb_0p02",
    payload_result_tag: str = "full_dcm_svd_n80_lr0p1_payload_lamb0p1/lamb_0p1",
    out_tag: str = "pointer_payload_cross_patch/full_dcm_svd_n80_lr0p1_l32_l46",
    n_examples: int = 80,
    pool_size: int = 600,
    batch_size: int = 4,
    seed: int = 123,
) -> dict:
    _ensure_runtime_layout()
    volume.reload()

    import torch
    from nnsight import LanguageModel
    from tqdm import tqdm

    from scripts.cache_activations import _build_samples

    pointer_layer = int(pointer_layer)
    payload_layer = int(payload_layer)
    samples = _build_samples(n_samples=pool_size, template_idx=2, seed=seed)
    known = [sample for sample in samples if sample["labels"]["is_unknown"] == 0]
    unknown = [sample for sample in samples if sample["labels"]["is_unknown"] == 1]
    n = min(int(n_examples), len(known) // 2, len(unknown))
    known_a = known[:n]
    known_b = known[n : 2 * n]
    unknown_u = unknown[:n]

    ptr_dcm_path = _load_dcm_json_path(
        pointer_result_tag, "answer_lookback-pointer", pointer_layer
    )
    ptr_basis_path = _load_svd_basis_path(
        source_svd_tag, "answer_lookback-pointer", pointer_layer
    )
    ptr_dcm = json.loads(ptr_dcm_path.read_text())
    ptr_mask = torch.tensor(
        ptr_dcm["singular_vector"]["metadata"]["mask"], dtype=torch.bool
    )
    ptr_basis = torch.load(ptr_basis_path, map_location="cpu").to(
        device="cuda", dtype=torch.float16
    )
    ptr_selected = ptr_basis[ptr_mask].contiguous()
    ptr_proj = ptr_selected.T @ ptr_selected

    pay_dcm_path = _load_dcm_json_path(
        payload_result_tag, "answer_lookback-payload", payload_layer
    )
    pay_basis_path = _load_svd_basis_path(
        source_svd_tag, "answer_lookback-payload", payload_layer
    )
    pay_dcm = json.loads(pay_dcm_path.read_text())
    pay_mask = torch.tensor(
        pay_dcm["singular_vector"]["metadata"]["mask"], dtype=torch.bool
    )
    pay_basis = torch.load(pay_basis_path, map_location="cpu").to(
        device="cuda", dtype=torch.float16
    )
    pay_selected = pay_basis[pay_mask].contiguous()
    pay_proj = pay_selected.T @ pay_selected

    lm = LanguageModel(
        MODEL_KEY,
        device_map="auto",
        torch_dtype=torch.float16,
        dispatch=True,
    )
    lm.tokenizer.padding_side = "left"
    if lm.tokenizer.pad_token is None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token

    def norm_answer(text: str) -> str:
        return str(text).lower().strip()

    scenarios = [
        {
            "name": "known_base_payload_swap",
            "base": known_a,
            "pointer": known_a,
            "payload": known_b,
            "expected": "payload",
        },
        {
            "name": "known_base_unknown_pointer_payload_swap",
            "base": known_a,
            "pointer": unknown_u,
            "payload": known_b,
            "expected": "unknown",
        },
        {
            "name": "unknown_base_known_pointer_payload_swap",
            "base": unknown_u,
            "pointer": known_a,
            "payload": known_b,
            "expected": "payload",
        },
    ]
    modes = ["no_patch", "pointer_only", "payload_only", "crossed"]
    rows = []

    with torch.inference_mode():
        for scenario in scenarios:
            for start in tqdm(
                range(0, n, batch_size), desc=f"cross patch {scenario['name']}"
            ):
                end = min(n, start + batch_size)
                base_batch = scenario["base"][start:end]
                pointer_batch = scenario["pointer"][start:end]
                payload_batch = scenario["payload"][start:end]
                base_prompts = [sample["prompt"] for sample in base_batch]
                pointer_prompts = [sample["prompt"] for sample in pointer_batch]
                payload_prompts = [sample["prompt"] for sample in payload_batch]

                saved = {}
                with lm.trace() as tracer:
                    with tracer.invoke(pointer_prompts):
                        saved["ptr"] = (
                            lm.model.layers[pointer_layer]
                            .output[0][:, -1]
                            .clone()
                            .save()
                        )
                    with tracer.invoke(payload_prompts):
                        saved["pay"] = (
                            lm.model.layers[payload_layer]
                            .output[0][:, -1]
                            .clone()
                            .save()
                        )

                    for mode in modes:
                        with tracer.invoke(base_prompts):
                            if mode in ("pointer_only", "crossed"):
                                current = (
                                    lm.model.layers[pointer_layer]
                                    .output[0][:, -1]
                                    .clone()
                                )
                                ptr_alt = torch.matmul(saved["ptr"], ptr_proj)
                                ptr_cur = torch.matmul(current, ptr_proj)
                                lm.model.layers[pointer_layer].output[0][:, -1] = (
                                    current - ptr_cur + ptr_alt
                                )
                            if mode in ("payload_only", "crossed"):
                                current = (
                                    lm.model.layers[payload_layer]
                                    .output[0][:, -1]
                                    .clone()
                                )
                                pay_alt = torch.matmul(saved["pay"], pay_proj)
                                pay_cur = torch.matmul(current, pay_proj)
                                lm.model.layers[payload_layer].output[0][:, -1] = (
                                    current - pay_cur + pay_alt
                                )
                            saved[f"pred_{mode}"] = torch.argmax(
                                lm.lm_head.output[:, -1], dim=-1
                            ).save()

                for mode in modes:
                    preds = saved[f"pred_{mode}"].detach().cpu()
                    for i, pred_id in enumerate(preds):
                        base_sample = base_batch[i]
                        pointer_sample = pointer_batch[i]
                        payload_sample = payload_batch[i]
                        prediction = norm_answer(lm.tokenizer.decode(pred_id))
                        base_target = norm_answer(base_sample["labels"]["target_state"])
                        pointer_target = norm_answer(
                            pointer_sample["labels"]["target_state"]
                        )
                        payload_target = norm_answer(
                            payload_sample["labels"]["target_state"]
                        )
                        if scenario["expected"] == "payload":
                            expected_target = payload_target
                        elif scenario["expected"] == "unknown":
                            expected_target = "unknown"
                        else:
                            expected_target = base_target
                        rows.append(
                            {
                                "scenario": scenario["name"],
                                "mode": mode,
                                "sample_idx": int(start + i),
                                "prediction": prediction,
                                "base_target": base_target,
                                "pointer_target": pointer_target,
                                "payload_target": payload_target,
                                "expected_target": expected_target,
                                "matches_base": prediction == base_target,
                                "matches_pointer": prediction == pointer_target,
                                "matches_payload": prediction == payload_target,
                                "matches_expected": prediction == expected_target,
                                "base_is_unknown": int(
                                    base_sample["labels"]["is_unknown"]
                                ),
                                "pointer_is_unknown": int(
                                    pointer_sample["labels"]["is_unknown"]
                                ),
                                "payload_is_unknown": int(
                                    payload_sample["labels"]["is_unknown"]
                                ),
                            }
                        )
                torch.cuda.empty_cache()

    summary = {}
    for scenario in scenarios:
        summary[scenario["name"]] = {}
        for mode in modes:
            subset = [
                row
                for row in rows
                if row["scenario"] == scenario["name"] and row["mode"] == mode
            ]
            summary[scenario["name"]][mode] = {
                "n": len(subset),
                "matches_base": sum(row["matches_base"] for row in subset)
                / len(subset),
                "matches_pointer": sum(row["matches_pointer"] for row in subset)
                / len(subset),
                "matches_payload": sum(row["matches_payload"] for row in subset)
                / len(subset),
                "matches_expected": sum(row["matches_expected"] for row in subset)
                / len(subset),
                "unknown_rate": sum(row["prediction"] == "unknown" for row in subset)
                / len(subset),
            }

    out_path = VOL_MOUNT / "results" / out_tag / "pointer_payload_cross_patch.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_key": MODEL_KEY,
        "pointer_layer": pointer_layer,
        "payload_layer": payload_layer,
        "source_svd_tag": source_svd_tag,
        "pointer_result_tag": pointer_result_tag,
        "payload_result_tag": payload_result_tag,
        "n_examples": n,
        "batch_size": batch_size,
        "seed": seed,
        "pointer_rank": int(ptr_selected.shape[0]),
        "payload_rank": int(pay_selected.shape[0]),
        "summary": summary,
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"out_path": str(out_path), "n_rows": len(rows)}, indent=2))
    volume.commit()
    return payload


@app.function(
    gpu="H100",
    cpu=16,
    memory=131072,
    ephemeral_disk=524288,
    volumes={str(VOL_MOUNT): volume},
    secrets=[env_secret],
    timeout=24 * 60 * 60,
)
def remote_makelov_illusion_audit(
    layer: int = 32,
    q_start_layer: int = 33,
    q_end_layer: int = 40,
    kernel_dim: int = 512,
    rowspace_dim: int = 2048,
    dormant_mode: str = "q_rowspace_min_diff",
    source_svd_tag: str = "svd_n80",
    pointer_result_tag: str = "full_dcm_svd_n80_lr0p1_pointer_lamb0p02/lamb_0p02",
    search_candidates: int = 96,
    search_examples: int = 32,
    candidate_batch_size: int = 16,
    search_diff_penalty: float = 0.0,
    construction_size: int = 80,
    validation_size: int = 80,
    probe_samples: int = 500,
    batch_size: int = 4,
    probe_batch_size: int = 8,
    seed: int = 0,
    probe_seed: int = 0,
    pca_dim: int = 200,
    out_tag: str = "makelov_illusion/l32_q33_40",
) -> dict:
    _ensure_runtime_layout()
    volume.reload()

    import sys

    import numpy as np
    import torch
    from nnsight import LanguageModel
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from tqdm import tqdm

    from scripts.cache_activations import _build_samples
    from scripts.train_probes import _make_y

    patching_dir = REPO_DIR / "scripts" / "patching_scripts"
    if str(patching_dir) not in sys.path:
        sys.path.insert(0, str(patching_dir))
    from run_patching_exp_utils import free_gpu_cache, prepare_dataset, set_seed
    from run_single_layer_patching_exps import validate

    set_seed(seed)
    layer = int(layer)
    q_layers = list(range(int(q_start_layer), int(q_end_layer) + 1))

    lm = LanguageModel(
        MODEL_KEY,
        device_map="auto",
        torch_dtype=torch.float16,
        dispatch=True,
    )
    lm.tokenizer.padding_side = "left"
    if lm.tokenizer.pad_token is None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token

    hidden_dim = int(lm.config.hidden_size)
    gram = torch.zeros((hidden_dim, hidden_dim), device="cuda", dtype=torch.float32)
    for q_layer in q_layers:
        weight = (
            lm.model.layers[q_layer]
            .self_attn.q_proj.weight.detach()
            .to(device="cuda", dtype=torch.float32)
        )
        gram.add_(weight.T @ weight)
        del weight
        torch.cuda.empty_cache()

    eigvals, eigvecs = torch.linalg.eigh(gram)
    kernel_dim = min(int(kernel_dim), hidden_dim - 1)
    rowspace_dim = min(int(rowspace_dim), hidden_dim - kernel_dim)
    kernel_basis = eigvecs[:, :kernel_dim].T.contiguous()
    rowspace_basis = eigvecs[:, -rowspace_dim:].T.contiguous()

    train_loader, valid_loader = prepare_dataset(
        lm=lm,
        experiment_name="answer_lookback-pointer",
        train_size=construction_size,
        valid_size=validation_size,
        batch_size=batch_size,
        remote=False,
    )

    diffs: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in tqdm(train_loader, desc=f"cache paired L{layer} diffs"):
            clean_prompts = batch["clean_prompt"]
            cf_prompts = batch["counterfactual_prompt"]
            with lm.trace() as tracer:
                with tracer.invoke(clean_prompts):
                    clean = lm.model.layers[layer].output[0][:, -1].clone().save()
                with tracer.invoke(cf_prompts):
                    cf = lm.model.layers[layer].output[0][:, -1].clone().save()
            diffs.append(cf.detach().cpu().float() - clean.detach().cpu().float())
            del clean, cf
            free_gpu_cache()
    diff = torch.cat(diffs, dim=0).to(device="cuda", dtype=torch.float32)

    kernel_coords = diff @ kernel_basis.T
    _, _, vh_kernel = torch.linalg.svd(kernel_coords, full_matrices=False)
    v_disc = vh_kernel[0] @ kernel_basis
    v_disc = v_disc / v_disc.norm().clamp_min(1e-8)

    search_stats: dict = {}

    if dormant_mode == "q_rowspace_min_diff":
        row_coords = diff @ rowspace_basis.T
        _, _, vh_row = torch.linalg.svd(row_coords, full_matrices=True)
        v_dorm = vh_row[-1] @ rowspace_basis
    elif dormant_mode == "dcm_min_diff":
        dcm_path = _load_dcm_json_path(
            pointer_result_tag, "answer_lookback-pointer", layer
        )
        basis_path = _load_svd_basis_path(
            source_svd_tag, "answer_lookback-pointer", layer
        )
        dcm = json.loads(dcm_path.read_text())
        mask = torch.tensor(
            dcm["singular_vector"]["metadata"]["mask"],
            dtype=torch.bool,
            device="cuda",
        )
        basis = torch.load(basis_path, map_location="cpu").to(
            device="cuda", dtype=torch.float32
        )
        dcm_basis = basis[mask]
        dcm_coords = diff @ dcm_basis.T
        _, _, vh_dcm = torch.linalg.svd(dcm_coords, full_matrices=True)
        v_dorm = vh_dcm[-1] @ dcm_basis
        del basis, dcm_basis, dcm_coords
    elif dormant_mode == "causal_dcm_search":
        dcm_path = _load_dcm_json_path(
            pointer_result_tag, "answer_lookback-pointer", layer
        )
        basis_path = _load_svd_basis_path(
            source_svd_tag, "answer_lookback-pointer", layer
        )
        dcm = json.loads(dcm_path.read_text())
        mask = torch.tensor(
            dcm["singular_vector"]["metadata"]["mask"],
            dtype=torch.bool,
            device="cuda",
        )
        basis = torch.load(basis_path, map_location="cpu").to(
            device="cuda", dtype=torch.float32
        )
        dcm_basis = basis[mask]
        rank = int(dcm_basis.shape[0])
        n_random = max(0, int(search_candidates) - rank)
        candidate_blocks = [dcm_basis]
        if n_random > 0:
            generator = torch.Generator(device="cuda")
            generator.manual_seed(int(seed) + 12345)
            coeff = torch.randn(
                (n_random, rank), generator=generator, device="cuda", dtype=torch.float32
            )
            candidate_blocks.append(coeff @ dcm_basis)
        candidates = torch.cat(candidate_blocks, dim=0)
        candidates = candidates - (candidates @ v_disc)[:, None] * v_disc[None, :]
        candidates = candidates / candidates.norm(dim=1, keepdim=True).clamp_min(1e-8)
        candidate_diff = torch.sqrt(((diff @ candidates.T) ** 2).mean(dim=0))

        plus_vecs = v_disc[None, :] + candidates
        plus_vecs = plus_vecs / plus_vecs.norm(dim=1, keepdim=True).clamp_min(1e-8)
        minus_vecs = v_disc[None, :] - candidates
        minus_vecs = minus_vecs / minus_vecs.norm(dim=1, keepdim=True).clamp_min(1e-8)
        search_vecs = torch.cat([plus_vecs, minus_vecs], dim=0)
        search_candidate_idx = torch.arange(candidates.shape[0], device="cuda").repeat(2)
        search_variant = ["plus"] * candidates.shape[0] + ["minus"] * candidates.shape[0]

        def score_search_chunk(
            vecs: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            sub_correct = torch.zeros(vecs.shape[0], device="cuda", dtype=torch.float32)
            comp_correct = torch.zeros(vecs.shape[0], device="cuda", dtype=torch.float32)
            sub_margin_sum = torch.zeros(vecs.shape[0], device="cuda", dtype=torch.float32)
            comp_margin_sum = torch.zeros(vecs.shape[0], device="cuda", dtype=torch.float32)
            total = 0
            for search_batch in train_loader:
                if total >= int(search_examples):
                    break
                org_prompts = list(search_batch["clean_prompt"])
                alt_prompts = list(search_batch["counterfactual_prompt"])
                targets = (
                    list(search_batch["target"])
                    if "target" in search_batch
                    else list(search_batch["counterfactual_target"])
                )
                remaining = int(search_examples) - total
                if len(org_prompts) > remaining:
                    org_prompts = org_prompts[:remaining]
                    alt_prompts = alt_prompts[:remaining]
                    targets = targets[:remaining]
                target_tokens = lm.tokenizer(
                    targets,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                ).input_ids
                target_ids = target_tokens[:, 0].to(device="cuda")
                chunk_size = int(vecs.shape[0])
                batch_n = len(org_prompts)
                expanded_org = [
                    prompt for _ in range(chunk_size) for prompt in org_prompts
                ]
                with lm.trace() as tracer:
                    with tracer.invoke(alt_prompts):
                        alt = (
                            lm.model.layers[layer]
                            .output[0][:, -1]
                            .clone()
                            .save()
                        )
                    with tracer.invoke(expanded_org):
                        current = lm.model.layers[layer].output[0][:, -1].clone()
                        current = current.reshape(chunk_size, batch_n, hidden_dim)
                        delta = alt[None, :, :].to(current.dtype) - current
                        vecs_cast = vecs.to(current.dtype)
                        coeff = (delta * vecs_cast[:, None, :]).sum(
                            dim=-1, keepdim=True
                        )
                        patch = current + coeff * vecs_cast[:, None, :]
                        lm.model.layers[layer].output[0][:, -1] = patch.reshape(
                            chunk_size * batch_n, hidden_dim
                        )
                        logits_sub = lm.lm_head.output[:, -1].save()
                    with tracer.invoke(expanded_org):
                        current = lm.model.layers[layer].output[0][:, -1].clone()
                        current = current.reshape(chunk_size, batch_n, hidden_dim)
                        delta = current - alt[None, :, :].to(current.dtype)
                        vecs_cast = vecs.to(current.dtype)
                        coeff = (delta * vecs_cast[:, None, :]).sum(
                            dim=-1, keepdim=True
                        )
                        patch = alt[None, :, :].to(current.dtype) + coeff * vecs_cast[
                            :, None, :
                        ]
                        lm.model.layers[layer].output[0][:, -1] = patch.reshape(
                            chunk_size * batch_n, hidden_dim
                        )
                        logits_comp = lm.lm_head.output[:, -1].save()

                gather_ids = target_ids[None, :, None].expand(chunk_size, batch_n, 1)
                logits_sub = logits_sub.detach().float().reshape(
                    chunk_size, batch_n, -1
                )
                pred_sub = torch.argmax(logits_sub, dim=-1)
                sub_correct += (pred_sub == target_ids[None, :]).float().sum(dim=1)
                target_logits_sub = logits_sub.gather(2, gather_ids).squeeze(-1)
                logits_sub.scatter_(2, gather_ids, float("-inf"))
                sub_margin_sum += (
                    target_logits_sub - logits_sub.max(dim=-1).values
                ).sum(dim=1)

                logits_comp = logits_comp.detach().float().reshape(
                    chunk_size, batch_n, -1
                )
                pred_comp = torch.argmax(logits_comp, dim=-1)
                comp_correct += (pred_comp == target_ids[None, :]).float().sum(dim=1)
                target_logits_comp = logits_comp.gather(2, gather_ids).squeeze(-1)
                logits_comp.scatter_(2, gather_ids, float("-inf"))
                comp_margin_sum += (
                    target_logits_comp - logits_comp.max(dim=-1).values
                ).sum(dim=1)

                total += batch_n
                del (
                    logits_sub,
                    logits_comp,
                    pred_sub,
                    pred_comp,
                    target_tokens,
                    target_ids,
                    alt,
                    current,
                    delta,
                    patch,
                )
                free_gpu_cache()
            if total == 0:
                raise ValueError("No search examples were available.")
            return (
                sub_correct / total,
                sub_margin_sum / total,
                comp_correct / total,
                comp_margin_sum / total,
            )

        all_acc: list[torch.Tensor] = []
        all_margin: list[torch.Tensor] = []
        all_comp_acc: list[torch.Tensor] = []
        all_comp_margin: list[torch.Tensor] = []
        chunk = max(1, int(candidate_batch_size))
        for start in tqdm(
            range(0, search_vecs.shape[0], chunk), desc="causal dormant search"
        ):
            vec_chunk = search_vecs[start : start + chunk]
            acc, margin, comp_acc, comp_margin = score_search_chunk(vec_chunk)
            all_acc.append(acc.detach())
            all_margin.append(margin.detach())
            all_comp_acc.append(comp_acc.detach())
            all_comp_margin.append(comp_margin.detach())
        search_acc = torch.cat(all_acc)
        search_margin = torch.cat(all_margin)
        search_comp_acc = torch.cat(all_comp_acc)
        search_comp_margin = torch.cat(all_comp_margin)
        penalties = candidate_diff[search_candidate_idx] * float(search_diff_penalty)
        search_score = (
            torch.minimum(search_acc, search_comp_acc)
            + 0.005 * (search_margin + search_comp_margin)
            - penalties
        )
        best_search_idx = int(torch.argmax(search_score).detach().cpu())
        best_candidate_idx = int(search_candidate_idx[best_search_idx].detach().cpu())
        best_variant = search_variant[best_search_idx]
        v_dorm = candidates[best_candidate_idx]
        if best_variant == "minus":
            v_dorm = -v_dorm
        search_stats = {
            "search_candidates_requested": int(search_candidates),
            "search_candidates_evaluated": int(candidates.shape[0]),
            "search_vectors_evaluated": int(search_vecs.shape[0]),
            "search_examples": int(search_examples),
            "candidate_batch_size": int(candidate_batch_size),
            "search_diff_penalty": float(search_diff_penalty),
            "search_best_index": best_search_idx,
            "search_best_candidate_index": best_candidate_idx,
            "search_best_variant": best_variant,
            "search_best_acc": float(search_acc[best_search_idx].detach().cpu()),
            "search_best_comp_acc": float(
                search_comp_acc[best_search_idx].detach().cpu()
            ),
            "search_best_margin": float(search_margin[best_search_idx].detach().cpu()),
            "search_best_comp_margin": float(
                search_comp_margin[best_search_idx].detach().cpu()
            ),
            "search_best_score": float(search_score[best_search_idx].detach().cpu()),
            "search_best_candidate_diff_rms": float(
                candidate_diff[best_candidate_idx].detach().cpu()
            ),
            "dcm_rank": rank,
            "search_top": [
                {
                    "rank": int(rank_i),
                    "search_index": int(idx.detach().cpu()),
                    "candidate_index": int(search_candidate_idx[idx].detach().cpu()),
                    "variant": search_variant[int(idx.detach().cpu())],
                    "acc": float(search_acc[idx].detach().cpu()),
                    "comp_acc": float(search_comp_acc[idx].detach().cpu()),
                    "margin": float(search_margin[idx].detach().cpu()),
                    "comp_margin": float(search_comp_margin[idx].detach().cpu()),
                    "score": float(search_score[idx].detach().cpu()),
                    "candidate_diff_rms": float(
                        candidate_diff[search_candidate_idx[idx]].detach().cpu()
                    ),
                }
                for rank_i, idx in enumerate(
                    torch.topk(search_score, k=min(10, search_score.numel())).indices
                )
            ],
        }
        del basis, dcm_basis, candidates, plus_vecs, minus_vecs, search_vecs
    else:
        raise ValueError(f"Unknown dormant_mode: {dormant_mode}")
    v_dorm = v_dorm - torch.dot(v_dorm, v_disc) * v_disc
    v_dorm = v_dorm / v_dorm.norm().clamp_min(1e-8)

    v_plus = v_disc + v_dorm
    v_plus = v_plus / v_plus.norm().clamp_min(1e-8)
    v_minus = v_disc - v_dorm
    v_minus = v_minus / v_minus.norm().clamp_min(1e-8)

    def q_energy(v: torch.Tensor) -> float:
        return float(torch.sqrt((v @ gram @ v).clamp_min(0)).detach().cpu())

    def diff_rms(v: torch.Tensor) -> float:
        return float(torch.sqrt(((diff @ v) ** 2).mean()).detach().cpu())

    direction_stats = {
        "q_layers": q_layers,
        "kernel_dim": kernel_dim,
        "rowspace_dim": rowspace_dim,
        "dormant_mode": dormant_mode,
        "source_svd_tag": source_svd_tag,
        "pointer_result_tag": pointer_result_tag,
        "eig_min": float(eigvals[0].detach().cpu()),
        "eig_kernel_max": float(eigvals[kernel_dim - 1].detach().cpu()),
        "eig_rowspace_min": float(eigvals[-rowspace_dim].detach().cpu()),
        "eig_max": float(eigvals[-1].detach().cpu()),
        "cos_disc_dorm": float(torch.dot(v_disc, v_dorm).detach().cpu()),
        "diff_rms_disc": diff_rms(v_disc),
        "diff_rms_dorm": diff_rms(v_dorm),
        "diff_rms_plus": diff_rms(v_plus),
        "diff_rms_minus": diff_rms(v_minus),
        "q_energy_disc": q_energy(v_disc),
        "q_energy_dorm": q_energy(v_dorm),
        "q_energy_plus": q_energy(v_plus),
        "q_energy_minus": q_energy(v_minus),
    }
    direction_stats.update(search_stats)

    out_dir = VOL_MOUNT / "results" / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "v_disc": v_disc.detach().cpu(),
            "v_dorm": v_dorm.detach().cpu(),
            "v_plus": v_plus.detach().cpu(),
            "v_minus": v_minus.detach().cpu(),
            "eigvals": eigvals.detach().cpu(),
            "metadata": direction_stats,
        },
        out_dir / "directions.pt",
    )

    eye = torch.eye(hidden_dim, device="cuda", dtype=torch.float16)

    def projector(v: torch.Tensor) -> torch.Tensor:
        vh = v.to(device="cuda", dtype=torch.float16)
        return torch.outer(vh, vh)

    full_iia = validate(
        exp_name="answer_lookback-pointer",
        lm=lm,
        layer_idx=layer,
        validation_loader=valid_loader,
        projection=None,
        verbose=False,
        save_outputs=False,
        projection_type="full_rank",
        remote=False,
        bigtom=False,
    )

    patch_rows = []
    for name, vec in [("plus", v_plus), ("minus", v_minus)]:
        proj = projector(vec)
        comp = eye - proj
        ill_iia = validate(
            exp_name="answer_lookback-pointer",
            lm=lm,
            layer_idx=layer,
            validation_loader=valid_loader,
            projection=proj,
            verbose=False,
            save_outputs=False,
            projection_type="singular_vector",
            remote=False,
            bigtom=False,
        )
        comp_iia = validate(
            exp_name="answer_lookback-pointer",
            lm=lm,
            layer_idx=layer,
            validation_loader=valid_loader,
            projection=comp,
            verbose=False,
            save_outputs=False,
            projection_type="complement",
            remote=False,
            bigtom=False,
        )
        patch_rows.append(
            {
                "variant": name,
                "full_iia": float(full_iia),
                "illusory_iia": float(ill_iia),
                "complement_iia": float(comp_iia),
            }
        )
        del proj, comp
        free_gpu_cache()

    def probe(X: np.ndarray, y: np.ndarray) -> dict:
        n_classes = int(len(np.unique(y)))
        counts = np.bincount(y)
        baseline = float(counts.max() / counts.sum())
        if n_classes < 2 or len(y) < 16 or X.shape[1] < 1:
            return {
                "test_acc": baseline if X.shape[1] < 1 else float("nan"),
                "baseline": baseline,
                "n": int(len(y)),
                "n_classes": n_classes,
                "pca_dim": 0,
            }
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.25, random_state=probe_seed, stratify=y
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.25, random_state=probe_seed
            )
        effective_pca = min(pca_dim, X_tr.shape[0] - 1, X_tr.shape[1])
        solver = "liblinear" if n_classes == 2 else "lbfgs"
        steps = [("scale", StandardScaler(with_mean=True, with_std=True))]
        if effective_pca > 0:
            steps.append(("pca", PCA(n_components=effective_pca, random_state=probe_seed)))
        steps.append(("lr", LogisticRegression(max_iter=1000, C=1.0, solver=solver)))
        pipe = Pipeline(steps)
        pipe.fit(X_tr, y_tr)
        return {
            "test_acc": float(pipe.score(X_te, y_te)),
            "baseline": baseline,
            "n": int(len(y)),
            "n_classes": n_classes,
            "pca_dim": int(effective_pca),
        }

    samples = _build_samples(n_samples=probe_samples, template_idx=2, seed=42)
    prompts = [sample["prompt"] for sample in samples]
    labels = [sample["labels"] for sample in samples]
    y, keep_idx = _make_y(labels, "state_idx_when_known")
    keep = torch.tensor(keep_idx, dtype=torch.long)
    acts: list[torch.Tensor] = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(prompts), probe_batch_size), desc=f"probe cache L{layer}"):
            batch = prompts[i : i + probe_batch_size]
            with lm.trace() as tracer:
                with tracer.invoke(batch):
                    saved = lm.model.layers[layer].output[0][:, -1].clone().save()
            acts.append(saved.detach().cpu().float())
            del saved
            free_gpu_cache()
    X = torch.cat(acts, dim=0)[keep]
    probe_rows = []
    for name, vec in [("plus", v_plus.detach().cpu()), ("minus", v_minus.detach().cpu())]:
        coords = X @ vec
        x_proj_features = coords[:, None].numpy()
        x_orth = X - coords[:, None] * vec[None, :]
        probe_rows.append(
            {
                "variant": name,
                "probe_full": probe(X.numpy(), y),
                "probe_proj": probe(x_proj_features, y),
                "probe_orth": probe(x_orth.numpy(), y),
            }
        )

    payload = {
        "model_key": MODEL_KEY,
        "experiment": "answer_lookback-pointer",
        "layer": layer,
        "construction_size": construction_size,
        "validation_size": validation_size,
        "probe_samples": probe_samples,
        "dormant_mode": dormant_mode,
        "source_svd_tag": source_svd_tag,
        "pointer_result_tag": pointer_result_tag,
        "search_candidates": search_candidates,
        "search_examples": search_examples,
        "candidate_batch_size": candidate_batch_size,
        "search_diff_penalty": search_diff_penalty,
        "batch_size": batch_size,
        "probe_batch_size": probe_batch_size,
        "seed": seed,
        "probe_seed": probe_seed,
        "direction_stats": direction_stats,
        "patch_rows": patch_rows,
        "probe_rows": probe_rows,
        "directions_path": str(out_dir / "directions.pt"),
    }
    (out_dir / "makelov_illusion_audit.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    volume.commit()
    return payload


@app.function(volumes={str(VOL_MOUNT): volume}, timeout=5 * 60)
def remote_status() -> dict:
    volume.reload()
    result_root = VOL_MOUNT / "results"
    svd_root = VOL_MOUNT / "svd"
    snapshot_root = VOL_MOUNT / "svd_snapshots"
    result_files = sorted(str(p.relative_to(result_root)) for p in result_root.rglob("*.json"))
    svd_files = sorted(str(p.relative_to(svd_root)) for p in svd_root.rglob("*.pt"))
    snapshot_files = (
        sorted(str(p.relative_to(snapshot_root)) for p in snapshot_root.rglob("*.pt"))
        if snapshot_root.exists()
        else []
    )
    status = {
        "volume": VOLUME_NAME,
        "n_result_json": len(result_files),
        "n_svd_pt": len(svd_files),
        "n_svd_snapshot_pt": len(snapshot_files),
        "result_samples": result_files[:20],
        "svd_samples": svd_files[:20],
        "svd_snapshot_samples": snapshot_files[:20],
    }
    print(json.dumps(status, indent=2))
    return status


@app.local_entrypoint()
def sanity():
    print(remote_sanity.remote())


@app.local_entrypoint()
def smoke(
    experiment: str = "answer_lookback-pointer",
    layer: int = 0,
    n_samples: int = 20,
    train_size: int = 4,
    validation_size: int = 4,
    batch_size: int = 4,
):
    print(
        remote_smoke.remote(
            experiment=experiment,
            layer=layer,
            n_samples=n_samples,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
        )
    )


@app.local_entrypoint()
def sweep_smoke(
    experiment: str = "answer_lookback-pointer",
    layers: str = "[32]",
    lambdas: str = "[0.1, 1.0]",
    source_svd_tag: str = "paper_hparams",
    sweep_tag: str = "sweep_smoke",
    train_size: int = 24,
    validation_size: int = 8,
    batch_size: int = 4,
    learning_rate: float = 0.1,
    n_epochs: int = 1,
):
    parsed_layers = json.loads(layers)
    parsed_lambdas = json.loads(lambdas)
    print(
        json.dumps(
            remote_sweep_dcm.remote(
                experiment=experiment,
                layers=parsed_layers,
                lambdas=parsed_lambdas,
                source_svd_tag=source_svd_tag,
                sweep_tag=sweep_tag,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                verbose=False,
            ),
            indent=2,
        )
    )


@app.local_entrypoint()
def svd(
    experiment: str = "both",
    n_samples: int = 500,
    batch_size: int = 4,
    layers: str = "",
):
    parsed_layers = json.loads(layers) if layers else QWEN_14B_LAYERS
    experiments = (
        ["answer_lookback-pointer", "answer_lookback-payload"]
        if experiment == "both"
        else [experiment]
    )
    for exp in experiments:
        print(remote_compute_svd.remote(exp, parsed_layers, n_samples, batch_size))


@app.local_entrypoint()
def svd_snapshot(
    experiment: str = "both",
    n_samples: int = 80,
    batch_size: int = 4,
    save_tag: str = "svd_n80",
    layers: str = "",
    parallel: bool = True,
):
    parsed_layers = json.loads(layers) if layers else QWEN_14B_LAYERS
    experiments = (
        ["answer_lookback-pointer", "answer_lookback-payload"]
        if experiment == "both"
        else [experiment]
    )
    if parallel and len(experiments) > 1:
        calls = [
            remote_compute_svd_snapshot.spawn(
                exp, parsed_layers, n_samples, batch_size, save_tag
            )
            for exp in experiments
        ]
        for call in calls:
            print(call.get())
    else:
        for exp in experiments:
            print(
                remote_compute_svd_snapshot.remote(
                    exp, parsed_layers, n_samples, batch_size, save_tag
                )
            )


@app.local_entrypoint()
def train(
    experiment: str,
    layers: str = "",
    train_size: int = 80,
    validation_size: int = 80,
    batch_size: int = 4,
    learning_rate: float = 0.01,
    lamb: float = 0.1,
    n_epochs: int = 1,
    save_tag: str = "paper_hparams",
    verbose: bool = False,
):
    parsed_layers = json.loads(layers) if layers else QWEN_14B_LAYERS
    print(
        remote_train_dcm.remote(
            experiment=experiment,
            layers=parsed_layers,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lamb=lamb,
            n_epochs=n_epochs,
            save_tag=save_tag,
            verbose=verbose,
        )
    )


@app.local_entrypoint()
def full(
    n_samples: int = 500,
    train_size: int = 80,
    validation_size: int = 80,
    batch_size: int = 4,
    save_tag: str = "paper_hparams",
    layers: str = "",
    parallel: bool = True,
):
    parsed_layers = json.loads(layers) if layers else QWEN_14B_LAYERS
    experiments = ["answer_lookback-pointer", "answer_lookback-payload"]
    if parallel:
        calls = [
            remote_full_experiment.spawn(
                experiment=exp,
                layers=parsed_layers,
                n_samples=n_samples,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                learning_rate=0.01,
                lamb=0.1,
                n_epochs=1,
                save_tag=save_tag,
                verbose=False,
            )
            for exp in experiments
        ]
        for call in calls:
            print(call.get())
    else:
        for exp in experiments:
            print(
                remote_full_experiment.remote(
                    experiment=exp,
                    layers=parsed_layers,
                    n_samples=n_samples,
                    train_size=train_size,
                    validation_size=validation_size,
                    batch_size=batch_size,
                    learning_rate=0.01,
                    lamb=0.1,
                    n_epochs=1,
                    save_tag=save_tag,
                    verbose=False,
                )
            )


@app.local_entrypoint()
def sweep(
    experiment: str = "both",
    layers: str = "",
    lambdas: str = "[0.03, 0.1, 0.3, 1.0, 3.0, 10.0]",
    source_svd_tag: str = "paper_hparams",
    sweep_tag: str = "lambda_sweep_lr0p1",
    train_size: int = 80,
    validation_size: int = 80,
    batch_size: int = 4,
    learning_rate: float = 0.1,
    n_epochs: int = 1,
    parallel: bool = True,
):
    parsed_lambdas = json.loads(lambdas)
    experiments = (
        ["answer_lookback-pointer", "answer_lookback-payload"]
        if experiment == "both"
        else [experiment]
    )

    def exp_layers(exp: str) -> list[int]:
        if layers:
            return json.loads(layers)
        return ACTIVE_SWEEP_LAYERS[exp]

    if parallel and len(experiments) > 1:
        calls = [
            remote_sweep_dcm.spawn(
                experiment=exp,
                layers=exp_layers(exp),
                lambdas=parsed_lambdas,
                source_svd_tag=source_svd_tag,
                sweep_tag=sweep_tag,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                verbose=False,
            )
            for exp in experiments
        ]
        for call in calls:
            print(json.dumps(call.get(), indent=2))
    else:
        for exp in experiments:
            print(
                json.dumps(
                    remote_sweep_dcm.remote(
                        experiment=exp,
                        layers=exp_layers(exp),
                        lambdas=parsed_lambdas,
                        source_svd_tag=source_svd_tag,
                        sweep_tag=sweep_tag,
                        train_size=train_size,
                        validation_size=validation_size,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        n_epochs=n_epochs,
                        verbose=False,
                    ),
                    indent=2,
                )
            )


@app.local_entrypoint()
def probe_audit(
    layers: str = "",
    source_svd_tag: str = "svd_n80",
    pointer_result_tag: str = "full_dcm_svd_n80_lr0p1_pointer_lamb0p2/lamb_0p2",
    payload_result_tag: str = "full_dcm_svd_n80_lr0p1_payload_lamb0p1/lamb_0p1",
    out_tag: str = "probe_projection_audit/full_dcm_svd_n80_lr0p1",
    n_samples: int = 500,
    batch_size: int = 8,
    seed: int = 42,
    probe_seed: int = 0,
    pca_dim: int = 200,
):
    parsed_layers = json.loads(layers) if layers else QWEN_14B_LAYERS
    result = remote_probe_projection_audit.remote(
        layers=parsed_layers,
        source_svd_tag=source_svd_tag,
        pointer_result_tag=pointer_result_tag,
        payload_result_tag=payload_result_tag,
        out_tag=out_tag,
        n_samples=n_samples,
        batch_size=batch_size,
        seed=seed,
        probe_seed=probe_seed,
        pca_dim=pca_dim,
    )
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def dcm_pca_clusters(
    pointer_layers: str = "[30, 32, 34, 36, 38, 40]",
    payload_layers: str = "[40, 42, 44, 46, 47]",
    source_svd_tag: str = "svd_n80",
    pointer_result_tag: str = "full_dcm_svd_n80_lr0p1_pointer_lamb0p02/lamb_0p02",
    payload_result_tag: str = "full_dcm_svd_n80_lr0p1_payload_lamb0p1/lamb_0p1",
    out_tag: str = "dcm_pca_clusters/full_dcm_svd_n80_lr0p1",
    n_samples: int = 400,
    batch_size: int = 4,
    seed: int = 42,
    pca_components: int = 10,
    cluster_ks: str = "[2, 3, 4, 5, 6, 7, 8]",
):
    result = remote_dcm_pca_clusters.remote(
        pointer_layers=json.loads(pointer_layers),
        payload_layers=json.loads(payload_layers),
        source_svd_tag=source_svd_tag,
        pointer_result_tag=pointer_result_tag,
        payload_result_tag=payload_result_tag,
        out_tag=out_tag,
        n_samples=n_samples,
        batch_size=batch_size,
        seed=seed,
        pca_components=pca_components,
        cluster_ks=json.loads(cluster_ks),
    )
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def payload_target_probe_audit(
    layers: str = "[40, 42, 44, 46, 47]",
    source_svd_tag: str = "svd_n80",
    payload_result_tag: str = "full_dcm_svd_n80_lr0p1_payload_lamb0p1/lamb_0p1",
    out_tag: str = "payload_target_probe_audit/full_dcm_svd_n80_lr0p1",
    n_samples: int = 500,
    batch_size: int = 8,
    seed: int = 42,
    probe_seed: int = 0,
    pca_dim: int = 200,
    target_top_k: int = 12,
    target_min_count: int = 4,
):
    result = remote_payload_target_probe_audit.remote(
        layers=json.loads(layers),
        source_svd_tag=source_svd_tag,
        payload_result_tag=payload_result_tag,
        out_tag=out_tag,
        n_samples=n_samples,
        batch_size=batch_size,
        seed=seed,
        probe_seed=probe_seed,
        pca_dim=pca_dim,
        target_top_k=target_top_k,
        target_min_count=target_min_count,
    )
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def payload_grouped_patching_audit(
    layers: str = "[40, 42, 44, 46, 47]",
    source_svd_tag: str = "svd_n80",
    payload_result_tag: str = "full_dcm_svd_n80_lr0p1_payload_lamb0p1/lamb_0p1",
    out_tag: str = "payload_grouped_patching_audit/full_dcm_svd_n80_lr0p1",
    train_size: int = 80,
    validation_size: int = 160,
    batch_size: int = 4,
    seed: int = 0,
):
    result = remote_payload_grouped_patching_audit.remote(
        layers=json.loads(layers),
        source_svd_tag=source_svd_tag,
        payload_result_tag=payload_result_tag,
        out_tag=out_tag,
        train_size=train_size,
        validation_size=validation_size,
        batch_size=batch_size,
        seed=seed,
    )
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def payload_token_geometry(
    layers: str = "[40, 42, 44, 46, 47]",
    source_svd_tag: str = "svd_n80",
    payload_result_tag: str = "full_dcm_svd_n80_lr0p1_payload_lamb0p1/lamb_0p1",
    grouped_patching_tag: str = (
        "payload_grouped_patching_audit/"
        "full_dcm_svd_n80_lr0p1_v240_payload_layers"
    ),
    out_tag: str = "payload_token_geometry/full_dcm_svd_n80_lr0p1",
):
    result = remote_payload_token_geometry.remote(
        layers=json.loads(layers),
        source_svd_tag=source_svd_tag,
        payload_result_tag=payload_result_tag,
        grouped_patching_tag=grouped_patching_tag,
        out_tag=out_tag,
    )
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def pointer_payload_cross_patch(
    pointer_layer: int = 32,
    payload_layer: int = 46,
    source_svd_tag: str = "svd_n80",
    pointer_result_tag: str = "full_dcm_svd_n80_lr0p1_pointer_lamb0p02/lamb_0p02",
    payload_result_tag: str = "full_dcm_svd_n80_lr0p1_payload_lamb0p1/lamb_0p1",
    out_tag: str = "pointer_payload_cross_patch/full_dcm_svd_n80_lr0p1_l32_l46",
    n_examples: int = 80,
    pool_size: int = 600,
    batch_size: int = 4,
    seed: int = 123,
):
    result = remote_pointer_payload_cross_patch.remote(
        pointer_layer=pointer_layer,
        payload_layer=payload_layer,
        source_svd_tag=source_svd_tag,
        pointer_result_tag=pointer_result_tag,
        payload_result_tag=payload_result_tag,
        out_tag=out_tag,
        n_examples=n_examples,
        pool_size=pool_size,
        batch_size=batch_size,
        seed=seed,
    )
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def complement_audit(
    experiment: str = "both",
    layers: str = "",
    source_svd_tag: str = "svd_n80",
    pointer_result_tag: str = "full_dcm_svd_n80_lr0p1_pointer_lamb0p2/lamb_0p2",
    payload_result_tag: str = "full_dcm_svd_n80_lr0p1_payload_lamb0p1/lamb_0p1",
    out_tag: str = "causal_complement_audit/full_dcm_svd_n80_lr0p1",
    train_size: int = 80,
    validation_size: int = 80,
    batch_size: int = 4,
    seed: int = 0,
    parallel: bool = True,
):
    experiments = (
        ["answer_lookback-pointer", "answer_lookback-payload"]
        if experiment == "both"
        else [experiment]
    )
    parsed_layers = json.loads(layers) if layers else QWEN_14B_LAYERS

    def result_tag(exp: str) -> str:
        if exp == "answer_lookback-pointer":
            return pointer_result_tag
        if exp == "answer_lookback-payload":
            return payload_result_tag
        raise ValueError(f"No result tag configured for {exp}")

    if parallel and len(experiments) > 1:
        calls = [
            remote_causal_complement_audit.spawn(
                experiment=exp,
                layers=parsed_layers,
                source_svd_tag=source_svd_tag,
                result_tag=result_tag(exp),
                out_tag=out_tag,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                seed=seed,
            )
            for exp in experiments
        ]
        for call in calls:
            print(json.dumps(call.get(), indent=2))
    else:
        for exp in experiments:
            print(
                json.dumps(
                    remote_causal_complement_audit.remote(
                        experiment=exp,
                        layers=parsed_layers,
                        source_svd_tag=source_svd_tag,
                        result_tag=result_tag(exp),
                        out_tag=out_tag,
                        train_size=train_size,
                        validation_size=validation_size,
                        batch_size=batch_size,
                        seed=seed,
                    ),
                    indent=2,
                )
            )


@app.local_entrypoint()
def makelov_illusion(
    layer: int = 32,
    q_start_layer: int = 33,
    q_end_layer: int = 40,
    kernel_dim: int = 512,
    rowspace_dim: int = 2048,
    dormant_mode: str = "q_rowspace_min_diff",
    source_svd_tag: str = "svd_n80",
    pointer_result_tag: str = "full_dcm_svd_n80_lr0p1_pointer_lamb0p02/lamb_0p02",
    search_candidates: int = 96,
    search_examples: int = 32,
    candidate_batch_size: int = 16,
    search_diff_penalty: float = 0.0,
    construction_size: int = 80,
    validation_size: int = 80,
    probe_samples: int = 500,
    batch_size: int = 4,
    probe_batch_size: int = 8,
    seed: int = 0,
    probe_seed: int = 0,
    pca_dim: int = 200,
    out_tag: str = "makelov_illusion/l32_q33_40",
):
    print(
        json.dumps(
            remote_makelov_illusion_audit.remote(
                layer=layer,
                q_start_layer=q_start_layer,
                q_end_layer=q_end_layer,
                kernel_dim=kernel_dim,
                rowspace_dim=rowspace_dim,
                dormant_mode=dormant_mode,
                source_svd_tag=source_svd_tag,
                pointer_result_tag=pointer_result_tag,
                search_candidates=search_candidates,
                search_examples=search_examples,
                candidate_batch_size=candidate_batch_size,
                search_diff_penalty=search_diff_penalty,
                construction_size=construction_size,
                validation_size=validation_size,
                probe_samples=probe_samples,
                batch_size=batch_size,
                probe_batch_size=probe_batch_size,
                seed=seed,
                probe_seed=probe_seed,
                pca_dim=pca_dim,
                out_tag=out_tag,
            ),
            indent=2,
        )
    )


@app.local_entrypoint()
def status():
    print(remote_status.remote())
