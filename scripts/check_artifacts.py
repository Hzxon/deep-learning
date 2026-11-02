from __future__ import annotations

"""Artifact and repository-contract checks for CI and local pre-flight.

Does not run training. Fails hard on contract violations.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.analysis import RESULT_COLUMNS, normalize_results_frame
from ser.features import FEATURE_SHAPE
from ser.models import EXPECTED_INPUT_HW
from ser.splits import path_column  # noqa: E402

REQUIRED_FILES = [
    "docs/SER_PROJECT_SPEC.md",
    "docs/VALIDATION_PROTOCOL.md",
    "docs/WORKFLOW_GUIDE.md",
    "docs/CI_SPEC.md",
    "docs/PROJECT_STRUCTURE.md",
    "AGENTS.md",
    "RESEARCH_DESIGN.md",
    "README.md",
    "CHANGELOG.md",
    "PROJECT_STATE.yaml",
    "RESEARCH_OS_SUMMARY.md",
    "requirements.txt",
    "config/config.yaml",
    "config/ci.yaml",
    "scripts/workflow.py",
    "scripts/workflows.yaml",
    "scripts/prepare_manifest.py",
    "scripts/preprocess.py",
    "scripts/make_splits.py",
    "scripts/prepare_ci_fixtures.py",
    "scripts/validate_dataset.py",
    "scripts/validate_features.py",
    "scripts/validate_splits.py",
    "scripts/reproducibility_check.py",
    "scripts/post_training_audit.py",
    "scripts/audit_training.py",
    "scripts/inspect_manifest.py",
    "scripts/smoke_test.py",
    "scripts/run_experiments.py",
    "scripts/gap_analysis.py",
    "scripts/make_report.py",
    "scripts/check_artifacts.py",
    "src/ser/data.py",
    "src/ser/features.py",
    "src/ser/models.py",
    "src/ser/splits.py",
    "src/ser/train.py",
    "src/ser/metrics.py",
    "src/ser/analysis.py",
    "src/ser/plotting.py",
    "src/ser/utils.py",
    "src/ser/audit.py",
]


def _fail(msg: str) -> None:
    raise SystemExit(f"ARTIFACTS CHECK FAILED: {msg}")


def check_required_files() -> None:
    missing = [p for p in REQUIRED_FILES if not (ROOT / p).exists()]
    if missing:
        _fail(f"missing required files: {missing}")


def check_no_notebooks() -> None:
    notebooks = sorted(ROOT.rglob("*.ipynb"))
    notebooks = [p for p in notebooks if ".venv" not in p.parts and "site-packages" not in p.parts]
    if notebooks:
        _fail(f"notebooks are forbidden: {notebooks[:10]}")


def check_no_duplicate_canonical_docs() -> None:
    """Fail if a canonical docs/ file has been re-forked at the repository root."""
    forkable = ["SER_PROJECT_SPEC.md", "VALIDATION_PROTOCOL.md", "WORKFLOW_GUIDE.md", "CI_SPEC.md"]
    offenders = [name for name in forkable if (ROOT / name).exists()]
    if offenders:
        _fail(
            f"canonical docs must live only under docs/: remove root copies {offenders}. "
            "Link to docs/<file> instead of duplicating content."
        )


def check_config(config_path: Path) -> dict:
    if not config_path.exists():
        _fail(f"config not found: {config_path}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    labels = cfg.get("labels", [])
    expected = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
    if labels != expected:
        _fail(f"labels mismatch: {labels} != {expected}")
    mel = cfg.get("mel", {})
    if int(mel.get("n_mels", -1)) != FEATURE_SHAPE[0]:
        _fail(f"mel.n_mels must be {FEATURE_SHAPE[0]}")
    if int(mel.get("target_frames", -1)) != FEATURE_SHAPE[1]:
        _fail(f"mel.target_frames must be {FEATURE_SHAPE[1]}")
    if int(mel.get("n_fft", -1)) != 512 or int(mel.get("hop_length", -1)) != 256:
        _fail("mel n_fft/hop_length must be 512/256 per project spec")
    model = cfg.get("model", {})
    if int(model.get("lstm_hidden", -1)) != 128:
        _fail("model.lstm_hidden must be 128")
    if EXPECTED_INPUT_HW != (1, FEATURE_SHAPE[0], FEATURE_SHAPE[1]):
        _fail(f"EXPECTED_INPUT_HW mismatch: {EXPECTED_INPUT_HW}")
    return cfg


def check_manifest_and_features(cfg: dict, *, require_present: bool) -> None:
    manifest_path = ROOT / cfg["paths"]["processed_manifest"]
    if not manifest_path.exists():
        if require_present:
            _fail(f"manifest missing: {manifest_path}")
        print(f"SKIP manifest/features (not present): {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    path_col = path_column(df)
    required_cols = {path_col, "dataset", "speaker", "emotion", "label", "feature_path"}
    missing = required_cols - set(df.columns)
    if missing:
        _fail(f"manifest missing columns: {sorted(missing)}")
    if df.empty:
        _fail("manifest is empty")
    if not df[path_col].is_unique:
        _fail("duplicate audio paths in manifest")
    if set(df["emotion"].unique()) != set(cfg["labels"]):
        _fail("manifest emotions do not match config labels")
    if df["feature_path"].isna().any():
        _fail("manifest has null feature_path")

    missing_feats = [p for p in df["feature_path"] if not Path(p).exists()]
    if missing_feats:
        _fail(f"{len(missing_feats)} feature files missing (e.g. {missing_feats[0]})")

    # Spot-check shapes (all rows if small; otherwise sample).
    sample = df if len(df) <= 256 else df.sample(n=256, random_state=0)
    for _, row in sample.iterrows():
        arr = np.load(row["feature_path"])
        if tuple(arr.shape) != FEATURE_SHAPE:
            _fail(f"bad feature shape {arr.shape} for {row['feature_path']}")
        if arr.dtype != np.float32:
            _fail(f"bad feature dtype {arr.dtype} for {row['feature_path']}")
    print(f"OK manifest/features: rows={len(df)} shape_checks={len(sample)}")


# Minimum train_size for a production experiment row. CI/demo fixtures use ~100.
PRODUCTION_MIN_TRAIN_SIZE = 1000
FORBIDDEN_RUN_ID_SUBSTRINGS = ("demo_", "smoke_", "trainer_framework_check")


def check_results_csv() -> None:
    path = ROOT / "results" / "results.csv"
    if not path.exists():
        print("SKIP results/results.csv (not present)")
        return
    df = normalize_results_frame(pd.read_csv(path))
    if df.empty:
        _fail("results.csv exists but contains zero rows")
    missing = [c for c in RESULT_COLUMNS if c not in df.columns]
    if missing:
        _fail(f"results.csv missing columns: {missing}")
    if "run_id" in df.columns:
        run_ids = df["run_id"].astype(str)
        for needle in FORBIDDEN_RUN_ID_SUBSTRINGS:
            if run_ids.str.contains(needle, na=False).any():
                _fail(f"results.csv contains forbidden synthetic run_id substring: {needle!r}")
    if "train_size" in df.columns:
        sizes = pd.to_numeric(df["train_size"], errors="coerce")
        if sizes.isna().any():
            _fail("results.csv has non-numeric train_size")
        bad = sizes < PRODUCTION_MIN_TRAIN_SIZE
        if bool(bad.any()):
            examples = df.loc[bad, "run_id"].astype(str).head(5).tolist() if "run_id" in df.columns else []
            _fail(
                f"results.csv has train_size < {PRODUCTION_MIN_TRAIN_SIZE} "
                f"(synthetic/fixture-sized rows); examples={examples}"
            )
    if df[["accuracy", "uar", "macro_f1"]].isna().any().any():
        _fail("results.csv contains NaN metrics")
    for col in ("accuracy", "uar", "macro_f1"):
        vals = df[col].astype(float)
        if ((vals < 0) | (vals > 1)).any():
            _fail(f"results.csv {col} out of [0,1]")
    print(f"OK results.csv: rows={len(df)}")


def check_no_full_training_in_ci_marker() -> None:
    """Guardrail: CI fixtures config must not point at production results path."""
    ci_cfg_path = ROOT / "config" / "ci.yaml"
    if not ci_cfg_path.exists():
        return
    ci_cfg = yaml.safe_load(ci_cfg_path.read_text(encoding="utf-8"))
    results_dir = str(ci_cfg.get("paths", {}).get("results_dir", ""))
    if results_dir in {"results", "artifacts/results"}:
        _fail("config/ci.yaml must not write into production results directories")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check repository artifacts and contracts.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--require-manifest",
        action="store_true",
        help="Fail if the configured manifest/features are missing.",
    )
    args = parser.parse_args()

    check_required_files()
    check_no_notebooks()
    check_no_duplicate_canonical_docs()
    check_no_full_training_in_ci_marker()
    cfg = check_config(ROOT / args.config)
    # Always validate production config contracts too when checking ci.yaml.
    if Path(args.config).name != "config.yaml":
        check_config(ROOT / "config" / "config.yaml")
    check_manifest_and_features(cfg, require_present=bool(args.require_manifest))
    check_results_csv()
    print("ARTIFACTS CHECK PASSED")


if __name__ == "__main__":
    main()
