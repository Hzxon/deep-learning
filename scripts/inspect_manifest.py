#!/usr/bin/env python3
"""Inspect the processed manifest and print a human-readable summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from ser.data import DISCARDED_EMOTIONS, LABELS  # noqa: E402
from ser.splits import path_column  # noqa: E402


def inspect_manifest(df: pd.DataFrame, labels: list[str]) -> GateResult:
    checks: list[CheckResult] = []
    path_col = path_column(df)
    required = {path_col, "dataset", "speaker", "emotion", "label", "feature_path"}
    missing_cols = sorted(required - set(df.columns))
    checks.append(
        CheckResult(
            "required_columns",
            not missing_cols,
            "ok" if not missing_cols else f"missing={missing_cols}",
        )
    )
    checks.append(CheckResult("non_empty", len(df) > 0, f"rows={len(df)}"))
    discarded = sorted(set(df["emotion"].unique()) & set(DISCARDED_EMOTIONS)) if "emotion" in df.columns else []
    checks.append(
        CheckResult(
            "no_discarded_emotions",
            not discarded,
            "ok" if not discarded else f"found={discarded}",
        )
    )
    if "emotion" in df.columns:
        unexpected = sorted(set(df["emotion"].unique()) - set(labels))
        checks.append(
            CheckResult(
                "emotion_set",
                not unexpected and set(labels).issubset(set(df["emotion"].unique())),
                f"emotions={sorted(df['emotion'].unique())}",
            )
        )
    passed = all(c.passed for c in checks)
    return GateResult(
        gate="manifest_inspect",
        passed=passed,
        checks=checks,
        message=f"rows={len(df)} datasets={df['dataset'].nunique() if 'dataset' in df.columns else 0}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SER manifest statistics.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--report", default="artifacts/reports/manifest_inspect.md")
    parser.add_argument("--json", default="artifacts/reports/manifest_inspect.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path = ROOT / cfg["paths"]["processed_manifest"]
    if not manifest_path.exists():
        print(f"MANIFEST MISSING: {manifest_path}")
        raise SystemExit(1)

    df = pd.read_csv(manifest_path)
    gate = inspect_manifest(df, list(cfg.get("labels", LABELS)))

    print("=== Manifest inspection ===")
    print(f"path: {manifest_path}")
    print(f"rows: {len(df)}")
    if "dataset" in df.columns:
        print(df.groupby("dataset").agg(n=("emotion", "size"), speakers=("speaker", "nunique")).to_string())
    if "emotion" in df.columns:
        print("\nEmotion counts:")
        print(df["emotion"].value_counts().sort_index().to_string())

    payload = {"manifest": str(manifest_path), "gate": gate.to_dict()}
    save_json(payload, ROOT / args.json)
    save_markdown(
        render_gate_report(title="Manifest Inspection", workflow="inspect_manifest", gates=[gate]),
        ROOT / args.report,
    )
    print("MANIFEST INSPECT " + ("PASSED" if gate.passed else "FAILED"))
    raise SystemExit(exit_code(gate.passed))


if __name__ == "__main__":
    main()
