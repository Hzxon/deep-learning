#!/usr/bin/env python3
"""Gate 1 — Dataset integrity validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.audit import (  # noqa: E402
    CheckResult,
    GateResult,
    exit_code,
    render_gate_report,
    save_json,
    save_markdown,
)
from ser.data import DISCARDED_EMOTIONS, LABELS  # noqa: E402

EXPECTED_DATASETS = {"CREMA-D", "RAVDESS", "TESS", "SAVEE"}
EXPECTED_SPEAKERS = {"CREMA-D": 91, "RAVDESS": 24, "TESS": 2, "SAVEE": 4}


def _path_col(df: pd.DataFrame) -> str:
    return "filepath" if "filepath" in df.columns else "path"


def validate_dataset(
    df: pd.DataFrame,
    cfg: dict,
    *,
    allow_missing_audio: bool = False,
    max_audio_checks: int = 200,
) -> GateResult:
    checks: list[CheckResult] = []
    labels = list(cfg.get("labels", LABELS))
    path_col = _path_col(df)

    required = {path_col, "dataset", "speaker", "emotion", "label"}
    missing = sorted(required - set(df.columns))
    checks.append(CheckResult("metadata_columns", not missing, "ok" if not missing else str(missing)))

    checks.append(CheckResult("non_empty", len(df) > 0, f"n={len(df)}"))
    dupes = int(df[path_col].duplicated().sum()) if path_col in df.columns else -1
    checks.append(CheckResult("no_duplicate_audio", dupes == 0, f"duplicates={dupes}"))

    if "emotion" in df.columns:
        discarded = sorted(set(df["emotion"].unique()) & set(DISCARDED_EMOTIONS))
        checks.append(CheckResult("no_discarded_emotions", not discarded, f"discarded={discarded}"))
        emotion_ok = set(df["emotion"].unique()) == set(labels)
        checks.append(
            CheckResult(
                "emotion_labels",
                emotion_ok,
                f"found={sorted(df['emotion'].unique())} expected={labels}",
            )
        )

    if "dataset" in df.columns:
        datasets = set(df["dataset"].unique())
        checks.append(
            CheckResult(
                "dataset_count",
                datasets == EXPECTED_DATASETS,
                f"found={sorted(datasets)}",
            )
        )
        speaker_counts = df.groupby("dataset")["speaker"].nunique().to_dict()
        speaker_ok = True
        details = []
        for ds, expected in EXPECTED_SPEAKERS.items():
            got = int(speaker_counts.get(ds, 0))
            # CI fixtures intentionally use fewer speakers.
            if allow_missing_audio:
                details.append(f"{ds}={got}")
                continue
            if got != expected:
                speaker_ok = False
            details.append(f"{ds}={got}/{expected}")
        checks.append(
            CheckResult(
                "speaker_counts",
                speaker_ok if not allow_missing_audio else all(v > 0 for v in speaker_counts.values()),
                ", ".join(details),
            )
        )

    if "duration" in df.columns:
        durations = pd.to_numeric(df["duration"], errors="coerce")
        bad_dur = int(durations.isna().sum() + (durations <= 0).sum())
        checks.append(
            CheckResult(
                "duration_positive",
                bad_dur == 0,
                f"bad={bad_dur} mean={float(durations.mean()):.3f}s",
            )
        )

    # Audio corruption / sample-rate spot checks.
    sample_rate = int(cfg.get("sample_rate", 16000))
    audio_errors: list[str] = []
    missing_audio = 0
    checked = 0
    sample = df if len(df) <= max_audio_checks else df.sample(n=max_audio_checks, random_state=0)
    for _, row in sample.iterrows():
        audio_path = Path(str(row[path_col]))
        if not audio_path.exists():
            missing_audio += 1
            continue
        checked += 1
        try:
            info = sf.info(str(audio_path))
            if info.frames <= 0:
                audio_errors.append(f"empty:{audio_path.name}")
            # Original corpora have heterogeneous native rates; preprocess resamples.
            # Only flag unreadable / zero-length files here.
            _ = info.samplerate
        except Exception as exc:  # noqa: BLE001 — collect corruption evidence
            audio_errors.append(f"corrupt:{audio_path.name}:{exc}")

    if allow_missing_audio:
        checks.append(
            CheckResult(
                "audio_readable",
                len(audio_errors) == 0,
                f"checked={checked} missing={missing_audio} errors={len(audio_errors)}",
            )
        )
    else:
        checks.append(
            CheckResult(
                "audio_present",
                missing_audio == 0,
                f"missing={missing_audio} of sampled={len(sample)}",
            )
        )
        checks.append(
            CheckResult(
                "audio_readable",
                len(audio_errors) == 0,
                f"checked={checked} errors={audio_errors[:5]} target_sr={sample_rate}",
            )
        )

    passed = all(c.passed for c in checks)
    return GateResult(
        gate="dataset_integrity",
        passed=passed,
        checks=checks,
        message=f"n={len(df)} errors={len(audio_errors)} missing_audio={missing_audio}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset / manifest integrity.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--allow-missing-audio", action="store_true")
    parser.add_argument("--max-audio-checks", type=int, default=200)
    parser.add_argument("--report", default="VALIDATION_REPORT.md")
    parser.add_argument("--json", default="artifacts/reports/validation_dataset.json")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    manifest_path = ROOT / cfg["paths"]["processed_manifest"]
    if not manifest_path.exists():
        print(f"VALIDATION FAILED: manifest missing at {manifest_path}")
        raise SystemExit(1)

    df = pd.read_csv(manifest_path)
    gate = validate_dataset(
        df,
        cfg,
        allow_missing_audio=bool(args.allow_missing_audio),
        max_audio_checks=int(args.max_audio_checks),
    )

    report = render_gate_report(
        title="Validation Report — Dataset Integrity",
        workflow="validate_dataset",
        gates=[gate],
    )
    save_markdown(report, ROOT / args.report)
    save_json({"gate": gate.to_dict()}, ROOT / args.json)

    # Also keep a stable path for CI artifact uploads.
    save_json({"gate": gate.to_dict()}, ROOT / "artifacts" / "reports" / "validation_dataset.json")

    print(gate.message)
    for check in gate.checks:
        print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    print("DATASET VALIDATION " + ("PASSED" if gate.passed else "FAILED"))
    raise SystemExit(exit_code(gate.passed))


if __name__ == "__main__":
    main()
