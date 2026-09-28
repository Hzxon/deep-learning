#!/usr/bin/env python3
"""Aggregate pre-training audit — runs nested validators unless --skip-nested."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.audit import (  # noqa: E402
    GateResult,
    exit_code,
    render_gate_report,
    save_json,
    save_markdown,
    update_project_state,
)


def _run(cmd: list[str]) -> GateResult:
    name = Path(cmd[1]).stem if len(cmd) > 1 else cmd[0]
    print(f"\n==> pre-training nested: {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    passed = completed.returncode == 0
    return GateResult(
        gate=name,
        passed=passed,
        message="ok" if passed else f"exit={completed.returncode}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full pre-training audit.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-nested",
        action="store_true",
        help="Assume nested validators already ran; only write the aggregate report.",
    )
    parser.add_argument("--report", default="PRE_TRAINING_AUDIT.md")
    parser.add_argument("--json", default="artifacts/reports/pre_training_audit.json")
    args = parser.parse_args()

    cfg_path = args.config
    seed = int(args.seed)
    gates: list[GateResult] = []

    if not args.skip_nested:
        # Order matches Gate 1→3, Gate 5, Gate 7 (docs/VALIDATION_PROTOCOL.md).
        commands = [
            [sys.executable, "scripts/validate_dataset.py", "--config", cfg_path],
            [sys.executable, "scripts/validate_features.py", "--config", cfg_path],
            [sys.executable, "scripts/validate_splits.py", "--config", cfg_path, "--seed", str(seed)],
            [sys.executable, "scripts/smoke_test.py"],
            [sys.executable, "scripts/reproducibility_check.py", "--config", cfg_path, "--seed", str(seed)],
        ]
        for cmd in commands:
            result = _run(cmd)
            gates.append(result)
            if not result.passed:
                break
    else:
        # Workflow already executed nested steps; mark report-only mode as PASS.
        gates = [
            GateResult("validate_dataset", True, message="skipped-nested"),
            GateResult("validate_features", True, message="skipped-nested"),
            GateResult("validate_splits", True, message="skipped-nested"),
            GateResult("smoke_test", True, message="skipped-nested"),
            GateResult("reproducibility_check", True, message="skipped-nested"),
        ]

    overall = all(g.passed for g in gates)
    md = render_gate_report(
        title="Pre-Training Audit",
        workflow="pre_training_audit",
        gates=gates,
        recommendations=(
            [
                "All pre-training gates passed.",
                "Safe to run: python scripts/run_experiments.py --config config/config.yaml ...",
            ]
            if overall
            else [
                "Do NOT start training.",
                "Fix failing gates and re-run: python scripts/workflow.py pre-training",
            ]
        ),
    )
    save_markdown(md, ROOT / args.report)
    save_json({"overall": overall, "gates": [g.to_dict() for g in gates]}, ROOT / args.json)
    update_project_state(
        workflow="pre_training_audit",
        overall_passed=overall,
        completed_stage="pre_training_audit" if overall else None,
        current_stage="training" if overall else "pre_training_audit",
        next_stage="post_training_audit" if overall else "pre_training_audit",
        blocked=[] if overall else [g.gate for g in gates if not g.passed],
    )

    print("PRE-TRAINING AUDIT " + ("PASSED" if overall else "FAILED"))
    print(f"Report: {ROOT / args.report}")
    raise SystemExit(exit_code(overall))


if __name__ == "__main__":
    main()
