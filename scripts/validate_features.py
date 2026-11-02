#!/usr/bin/env python3
"""Gate 2 — Feature integrity validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.audit import (  # noqa: E402
    CheckResult,
    GateResult,
    exit_code,
    load_config,  # noqa: E402
    render_gate_report,
    save_json,
    save_markdown,
)
from ser.features import FEATURE_SHAPE  # noqa: E402


def validate_features(df: pd.DataFrame, *, max_checks: int = 512) -> GateResult:
    checks: list[CheckResult] = []
    if "feature_path" not in df.columns:
        checks.append(CheckResult("feature_path_column", False, "feature_path missing"))
        return GateResult("feature_integrity", False, checks, "missing feature_path")

    missing = [p for p in df["feature_path"] if not Path(str(p)).exists()]
    checks.append(
        CheckResult(
            "features_exist",
            len(missing) == 0,
            f"missing={len(missing)}" + (f" e.g. {missing[0]}" if missing else ""),
        )
    )

    sample = df if len(df) <= max_checks else df.sample(n=max_checks, random_state=0)
    bad_shape = 0
    bad_dtype = 0
    nan_count = 0
    inf_count = 0
    range_notes: list[str] = []
    checked = 0
    for _, row in sample.iterrows():
        path = Path(str(row["feature_path"]))
        if not path.exists():
            continue
        arr = np.load(path)
        checked += 1
        if tuple(arr.shape) != FEATURE_SHAPE:
            bad_shape += 1
        if arr.dtype != np.float32:
            bad_dtype += 1
        if np.isnan(arr).any():
            nan_count += 1
        if np.isinf(arr).any():
            inf_count += 1
        # Log-mel in dB is typically negative; allow a wide but finite range.
        amin = float(np.min(arr))
        amax = float(np.max(arr))
        if amin < -160 or amax > 80:
            range_notes.append(f"{path.name}:[{amin:.1f},{amax:.1f}]")

    checks.append(CheckResult("shape_64x188", bad_shape == 0, f"bad_shape={bad_shape}/{checked}"))
    checks.append(CheckResult("dtype_float32", bad_dtype == 0, f"bad_dtype={bad_dtype}/{checked}"))
    checks.append(CheckResult("no_nan", nan_count == 0, f"nan_files={nan_count}"))
    checks.append(CheckResult("no_inf", inf_count == 0, f"inf_files={inf_count}"))
    checks.append(
        CheckResult(
            "normalization_range",
            len(range_notes) == 0,
            "ok" if not range_notes else f"out_of_range={range_notes[:5]}",
        )
    )

    passed = all(c.passed for c in checks)
    return GateResult(
        gate="feature_integrity",
        passed=passed,
        checks=checks,
        message=f"checked={checked} missing={len(missing)}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate log-mel feature integrity.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--max-checks", type=int, default=512)
    parser.add_argument("--report", default="artifacts/reports/VALIDATION_FEATURES.md")
    parser.add_argument("--json", default="artifacts/reports/validation_features.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path = ROOT / cfg["paths"]["processed_manifest"]
    if not manifest_path.exists():
        print(f"VALIDATION FAILED: manifest missing at {manifest_path}")
        raise SystemExit(1)

    df = pd.read_csv(manifest_path)
    gate = validate_features(df, max_checks=int(args.max_checks))
    save_markdown(
        render_gate_report(title="Validation Report — Feature Integrity", workflow="validate_features", gates=[gate]),
        ROOT / args.report,
    )
    save_json({"gate": gate.to_dict()}, ROOT / args.json)

    for check in gate.checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    print("FEATURE VALIDATION " + ("PASSED" if gate.passed else "FAILED"))
    raise SystemExit(exit_code(gate.passed))


if __name__ == "__main__":
    main()
