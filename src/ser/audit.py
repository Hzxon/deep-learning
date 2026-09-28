"""Shared helpers for Research OS validation, audits, and reports."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CheckResult:
    """One atomic validation check."""

    name: str
    passed: bool
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateResult:
    """Aggregated result for one quality gate."""

    gate: str
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "message": self.message,
            "checks": [asdict(c) for c in self.checks],
        }


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def project_root() -> Path:
    return ROOT


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_json(data: Any, path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def save_markdown(text: str, path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def repo_state_summary() -> dict[str, Any]:
    """Collect lightweight repository state for audit headers."""
    git_head = "unknown"
    try:
        git_head = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        git_head = "not-a-git-repo"
    return {
        "timestamp": utc_now(),
        "root": str(ROOT),
        "git_head": git_head,
        "python": sys.version.split()[0],
    }


def render_gate_report(
    *,
    title: str,
    workflow: str,
    gates: list[GateResult],
    recommendations: list[str] | None = None,
    extra_sections: list[tuple[str, str]] | None = None,
) -> str:
    """Render a standard Markdown audit/validation report."""
    state = repo_state_summary()
    overall = all(g.passed for g in gates)
    lines = [
        f"# {title}",
        "",
        f"- **Timestamp:** {state['timestamp']}",
        f"- **Repository:** `{state['root']}`",
        f"- **Git HEAD:** `{state['git_head']}`",
        f"- **Python:** {state['python']}",
        f"- **Workflow:** `{workflow}`",
        f"- **Overall:** {'PASS' if overall else 'FAIL'}",
        "",
        "## Gate summary",
        "",
        "| Gate | Result | Message |",
        "|---|---|---|",
    ]
    for gate in gates:
        status = "PASS" if gate.passed else "FAIL"
        msg = gate.message.replace("|", "\\|")
        lines.append(f"| {gate.gate} | {status} | {msg} |")

    lines.extend(["", "## Checks", ""])
    for gate in gates:
        lines.append(f"### {gate.gate}")
        lines.append("")
        if not gate.checks:
            lines.append("_No atomic checks recorded._")
            lines.append("")
            continue
        for check in gate.checks:
            mark = "PASS" if check.passed else "FAIL"
            detail = check.detail.strip() or "—"
            lines.append(f"- **{check.name}:** {mark} — {detail}")
        lines.append("")

    if extra_sections:
        for heading, body in extra_sections:
            lines.append(f"## {heading}")
            lines.append("")
            lines.append(body.rstrip())
            lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    recs = recommendations or (
        ["All gates passed. Safe to proceed."]
        if overall
        else ["Fix failing gates, then re-run the same workflow before continuing."]
    )
    for rec in recs:
        lines.append(f"- {rec}")
    lines.append("")
    return "\n".join(lines)


def load_project_state(path: Path | None = None) -> dict[str, Any]:
    state_path = path or (ROOT / "PROJECT_STATE.yaml")
    if not state_path.exists():
        return {
            "completed_stages": [],
            "current_stage": "environment_setup",
            "next_stage": "dataset_preparation",
            "blocked_conditions": [],
            "last_workflow": None,
            "last_updated": None,
        }
    return yaml.safe_load(state_path.read_text(encoding="utf-8")) or {}


def update_project_state(
    *,
    workflow: str,
    overall_passed: bool,
    completed_stage: str | None = None,
    current_stage: str | None = None,
    next_stage: str | None = None,
    blocked: list[str] | None = None,
    path: Path | None = None,
) -> Path:
    """Update PROJECT_STATE.yaml after a workflow run."""
    state_path = path or (ROOT / "PROJECT_STATE.yaml")
    state = load_project_state(state_path)
    completed = list(state.get("completed_stages") or [])
    if overall_passed and completed_stage and completed_stage not in completed:
        completed.append(completed_stage)
    state["completed_stages"] = completed
    if current_stage:
        state["current_stage"] = current_stage
    if next_stage:
        state["next_stage"] = next_stage
    state["blocked_conditions"] = blocked or ([] if overall_passed else [f"{workflow} failed"])
    state["last_workflow"] = workflow
    state["last_updated"] = utc_now()
    state["last_result"] = "PASS" if overall_passed else "FAIL"
    state_path.write_text(yaml.safe_dump(state, sort_keys=False), encoding="utf-8")
    return state_path


def exit_code(passed: bool) -> int:
    return 0 if passed else 1
