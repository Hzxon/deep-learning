#!/usr/bin/env python3
"""YAML-driven Research OS workflow orchestrator.

Usage:
  python scripts/workflow.py pre-change
  python scripts/workflow.py pre-training
  python scripts/workflow.py post-training
  python scripts/workflow.py ci-local
  python scripts/workflow.py release
  python scripts/workflow.py list
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.audit import (  # noqa: E402
    GateResult,
    exit_code,
    render_gate_report,
    save_json,
    save_markdown,
    update_project_state,
    utc_now,
)

WORKFLOWS_PATH = ROOT / "scripts" / "workflows.yaml"


def load_workflows() -> dict[str, Any]:
    payload = yaml.safe_load(WORKFLOWS_PATH.read_text(encoding="utf-8"))
    workflows = payload.get("workflows")
    if not isinstance(workflows, dict) or not workflows:
        raise SystemExit(f"No workflows defined in {WORKFLOWS_PATH}")
    return workflows


def run_step(step: dict[str, Any], *, env: dict[str, str]) -> GateResult:
    name = str(step["name"])
    command = str(step["command"])
    gate = str(step.get("gate", name))
    print(f"\n==> [{gate}] {name}")
    print(f"$ {command}")
    completed = subprocess.run(
        command,
        shell=True,
        cwd=ROOT,
        env=env,
        check=False,
    )
    passed = completed.returncode == 0
    message = "ok" if passed else f"exit={completed.returncode}"
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    return GateResult(gate=gate, passed=passed, message=message)


def execute_workflow(name: str, workflows: dict[str, Any]) -> int:
    if name not in workflows:
        known = ", ".join(sorted(workflows))
        raise SystemExit(f"Unknown workflow '{name}'. Known: {known}")

    spec = workflows[name]
    steps = spec.get("steps") or []
    if not steps:
        raise SystemExit(f"Workflow '{name}' has no steps")

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    (ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)

    print(f"Research OS workflow: {name}")
    print(f"Started: {utc_now()}")
    print(spec.get("description", ""))

    results: list[GateResult] = []
    for step in steps:
        result = run_step(step, env=env)
        results.append(result)
        if not result.passed:
            print(f"\nWORKFLOW STOPPED: gate '{result.gate}' failed.")
            break

    overall = all(r.passed for r in results) and len(results) == len(steps)
    blocked = [] if overall else [f"{r.gate} failed" for r in results if not r.passed]

    update_project_state(
        workflow=name,
        overall_passed=overall,
        completed_stage=spec.get("completed_stage"),
        current_stage=spec.get("current_stage") if overall else name,
        next_stage=spec.get("next_stage") if overall else spec.get("current_stage"),
        blocked=blocked,
    )

    report_name = spec.get("report")
    if report_name:
        # The report always reflects the gates this run actually executed.
        md = render_gate_report(
            title="Pre-Training Audit" if name == "pre-training" else f"Workflow: {name}",
            workflow=name,
            gates=results,
            recommendations=spec.get("recommendations_pass" if overall else "recommendations_fail"),
        )
        save_markdown(md, ROOT / str(report_name))
        save_json(
            {"overall": overall, "gates": [r.to_dict() for r in results]},
            ROOT / "artifacts" / "reports" / f"{Path(str(report_name)).stem.lower()}.json",
        )

    summary_path = ROOT / "artifacts" / "reports" / f"workflow_{name.replace('-', '_')}.md"
    save_markdown(
        render_gate_report(
            title=f"Workflow Report — {name}",
            workflow=name,
            gates=results,
            recommendations=(
                ["Workflow completed successfully."]
                if overall
                else ["Fix the first failing gate, then re-run this workflow."]
            ),
        ),
        summary_path,
    )

    print("\n" + ("=" * 60))
    print(f"WORKFLOW {name}: {'PASS' if overall else 'FAIL'}")
    print("State updated: PROJECT_STATE.yaml")
    print(f"Report: {summary_path}")
    return exit_code(overall)


def main() -> None:
    workflows = load_workflows()
    parser = argparse.ArgumentParser(description="SER Research OS workflow orchestrator.")
    parser.add_argument(
        "workflow",
        nargs="?",
        default="list",
        help="Workflow name, or 'list' to show available workflows.",
    )
    args = parser.parse_args()

    if args.workflow in {"list", "ls", "--list"}:
        print("Available workflows:")
        for key, spec in workflows.items():
            print(f"  {key:16s}  {spec.get('description', '')}")
        raise SystemExit(0)

    raise SystemExit(execute_workflow(args.workflow, workflows))


if __name__ == "__main__":
    main()
