#!/usr/bin/env python3
"""Gate 3 — Split integrity validation."""

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
from ser.splits import PROTOCOLS, assert_split_integrity, iter_split_bundles, speaker_sets  # noqa: E402


def validate_splits(df: pd.DataFrame, *, seed: int) -> GateResult:
    checks: list[CheckResult] = []
    for protocol in PROTOCOLS:
        for fold_name, bundle in iter_split_bundles(df, protocol=protocol, seed=seed):
            try:
                assert_split_integrity(df, bundle, protocol=protocol)
                integrity_ok = True
                integrity_detail = "ok"
            except Exception as exc:  # noqa: BLE001
                integrity_ok = False
                integrity_detail = str(exc)
            checks.append(CheckResult(f"{fold_name}:integrity", integrity_ok, integrity_detail))

            train_s, val_s, test_s = speaker_sets(bundle)
            if protocol == "random":
                # Stratification: every split should contain all emotions when feasible.
                for split_name, split_df in (("train", bundle.train), ("val", bundle.val), ("test", bundle.test)):
                    n_emotions = split_df["emotion"].nunique() if "emotion" in split_df.columns else 0
                    checks.append(
                        CheckResult(
                            f"{fold_name}:{split_name}_stratification",
                            n_emotions == df["emotion"].nunique() or len(split_df) < df["emotion"].nunique(),
                            f"emotions={n_emotions}",
                        )
                    )
            else:
                overlap_tv = train_s & val_s
                overlap_tt = train_s & test_s
                overlap_vt = val_s & test_s
                checks.append(
                    CheckResult(
                        f"{fold_name}:no_speaker_overlap",
                        not (overlap_tv or overlap_tt or overlap_vt),
                        f"tv={len(overlap_tv)} tt={len(overlap_tt)} vt={len(overlap_vt)}",
                    )
                )

            if protocol == "leave_one_corpus_out":
                held_out = set(bundle.test["dataset"].unique())
                train_corpora = set(bundle.train["dataset"].unique()) | set(bundle.val["dataset"].unique())
                leakage = held_out & train_corpora
                checks.append(
                    CheckResult(
                        f"{fold_name}:no_corpus_leakage",
                        not leakage and len(held_out) == 1,
                        f"held_out={sorted(held_out)} leakage={sorted(leakage)}",
                    )
                )

    passed = all(c.passed for c in checks)
    return GateResult(
        gate="split_integrity",
        passed=passed,
        checks=checks,
        message=f"protocols={list(PROTOCOLS)} seed={seed} checks={len(checks)}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate train/val/test split integrity.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default="artifacts/reports/VALIDATION_SPLITS.md")
    parser.add_argument("--json", default="artifacts/reports/validation_splits.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path = ROOT / cfg["paths"]["processed_manifest"]
    if not manifest_path.exists():
        print(f"VALIDATION FAILED: manifest missing at {manifest_path}")
        raise SystemExit(1)

    df = pd.read_csv(manifest_path)
    gate = validate_splits(df, seed=int(args.seed))
    save_markdown(
        render_gate_report(title="Validation Report — Split Integrity", workflow="validate_splits", gates=[gate]),
        ROOT / args.report,
    )
    save_json({"gate": gate.to_dict()}, ROOT / args.json)

    failures = [c for c in gate.checks if not c.passed]
    for check in gate.checks:
        if not check.passed:
            print(f"[FAIL] {check.name}: {check.detail}")
    if not failures:
        print(f"OK all split checks ({len(gate.checks)})")
    print("SPLIT VALIDATION " + ("PASSED" if gate.passed else "FAILED"))
    raise SystemExit(exit_code(gate.passed))


if __name__ == "__main__":
    main()
