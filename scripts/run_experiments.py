from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.analysis import append_results
from ser.audit import load_config
from ser.splits import PROTOCOLS, build_splits, path_column
from ser.train import train_and_evaluate
from ser.utils import ensure_dirs, get_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("run_experiments")

MODEL_IDS = ["M0", "M1", "M2", "M3"]


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply supported CLI hyperparameter overrides into the config dict."""
    train = cfg.setdefault("train", {})
    if args.batch_size is not None:
        train["batch_size"] = int(args.batch_size)
    if args.epochs is not None:
        train["max_epochs"] = int(args.epochs)
    if args.learning_rate is not None:
        train["learning_rate"] = float(args.learning_rate)
    if args.optimizer is not None:
        train["optimizer"] = str(args.optimizer)
    return cfg


def _results_paths(cfg: dict) -> list[Path]:
    """Primary results/results.csv plus configured results_dir mirror."""
    paths = [ROOT / "results" / "results.csv"]
    configured = ROOT / cfg["paths"]["results_dir"] / "results.csv"
    if configured.resolve() != paths[0].resolve():
        paths.append(configured)
    return paths


def _outputs_dir(cfg: dict) -> Path:
    return ROOT / cfg["paths"].get("outputs_dir", cfg["paths"]["results_dir"])


def _load_existing_artifact(run_dir: Path) -> dict[str, Any] | None:
    """Return a loadable metrics.json artifact, or None if missing/invalid."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        artifact = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not load %s: %s", metrics_path, exc)
        return None
    if not isinstance(artifact, dict) or "run_id" not in artifact:
        LOGGER.warning("Invalid artifact shape in %s", metrics_path)
        return None
    if "test_metrics" not in artifact:
        LOGGER.warning("Incomplete artifact (no test_metrics) in %s", metrics_path)
        return None
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SER experiment matrix (M0–M3 × protocols × seeds).")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--protocol", choices=list(PROTOCOLS) + ["all"], default="all")
    parser.add_argument("--models", nargs="+", choices=MODEL_IDS, default=MODEL_IDS)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", "--max-epochs", dest="epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default=None)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs whose artifacts/runs/<run_id>/metrics.json already exists; upsert into results.csv.",
    )
    parser.add_argument(
        "--gap-analysis",
        action="store_true",
        help="Run gap analysis automatically after experiments finish.",
    )
    args = parser.parse_args()

    cfg = _apply_cli_overrides(load_config(args.config), args)

    labels = cfg["labels"]
    seeds = args.seeds if args.seeds is not None else [int(cfg.get("seed", 42))]
    manifest_path = ROOT / cfg["paths"]["processed_manifest"]
    df = pd.read_csv(manifest_path)
    missing = sorted({"dataset", "speaker", "emotion", "label", "feature_path"} - set(df.columns))
    try:
        path_column(df)
    except KeyError:
        missing.append("filepath")
    if missing:
        raise RuntimeError(f"Manifest missing columns: {missing}. Run prepare_manifest.py then preprocess.py.")

    device = get_device()
    print(f"Device: {device}")
    protocols = list(PROTOCOLS) if args.protocol == "all" else [args.protocol]
    result_paths = _results_paths(cfg)
    outputs_dir = _outputs_dir(cfg)
    for path in result_paths:
        ensure_dirs(path.parent)

    all_artifacts: list[dict] = []
    skipped = 0
    total = 0
    planned = 0
    splits_by_job = {
        (seed, protocol): build_splits(df, protocol=protocol, seed=seed) for seed in seeds for protocol in protocols
    }
    for split_list in splits_by_job.values():
        planned += len(split_list) * len(args.models)

    print(
        f"Running {planned} jobs "
        f"(protocols={protocols}, models={args.models}, seeds={seeds}, skip_existing={args.skip_existing})"
    )

    for seed in seeds:
        for protocol in protocols:
            for item in splits_by_job[(seed, protocol)]:
                if isinstance(item, tuple):
                    held_out, split = item
                    extra_name = held_out
                else:
                    split = item
                    extra_name = ""
                for model_id in args.models:
                    total += 1
                    suffix = f"_{extra_name}" if extra_name else ""
                    run_id = f"{protocol}{suffix}_{model_id}_seed{seed}".replace("-", "_")
                    print(f"\n=== [{total}/{planned}] {run_id} ===")

                    if args.skip_existing:
                        existing = _load_existing_artifact(outputs_dir / run_id)
                        if existing is not None:
                            print(f"SKIP existing metrics.json -> {outputs_dir / run_id / 'metrics.json'}")
                            artifact = existing
                            skipped += 1
                            all_artifacts.append(artifact)
                            for path in result_paths:
                                append_results([artifact], path, labels=labels)
                            continue

                    artifact = train_and_evaluate(
                        cfg=cfg,
                        model_id=model_id,
                        split_name=split.name,
                        train_df=split.train,
                        val_df=split.val,
                        test_df=split.test,
                        labels=labels,
                        seed=seed,
                        run_id=run_id,
                        device=device,
                        max_epochs_override=args.epochs,
                    )
                    all_artifacts.append(artifact)
                    for path in result_paths:
                        append_results([artifact], path, labels=labels)

    print(f"\nResults saved -> {result_paths[0]}")
    for path in result_paths[1:]:
        print(f"Results mirrored -> {path}")
    print(f"Completed runs: {len(all_artifacts)} (skipped existing: {skipped})")

    if args.gap_analysis and all_artifacts:
        import runpy

        sys.argv = [
            str(ROOT / "scripts" / "gap_analysis.py"),
            "--results",
            str(result_paths[0].relative_to(ROOT)),
            "--config",
            args.config,
        ]
        runpy.run_path(str(ROOT / "scripts" / "gap_analysis.py"), run_name="__main__")


if __name__ == "__main__":
    main()
