#!/usr/bin/env python3
"""Post-training health audit from history.csv + metrics.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.audit import (  # noqa: E402
    CheckResult,
    GateResult,
    discover_runs,
    exit_code,
    from_root,
    load_config,
    render_gate_report,
    save_json,
    save_markdown,
    update_project_state,
)


def audit_run(run_dir: Path) -> dict[str, Any]:
    history = pd.read_csv(run_dir / "history.csv")
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    flags: dict[str, bool] = {
        "overfitting": False,
        "underfitting": False,
        "exploding_gradients": False,
        "vanishing_gradients": False,
        "learning_plateau": False,
        "checkpoint_incorrect": False,
    }
    notes: list[str] = []

    if history.empty:
        flags["underfitting"] = True
        notes.append("empty history")
        return {"run_id": run_dir.name, "flags": flags, "notes": notes, "passed": False}

    train_loss = history["train_loss"].astype(float) if "train_loss" in history else None
    val_uar = history["val_uar"].astype(float) if "val_uar" in history else None
    train_uar = history["train_uar"].astype(float) if "train_uar" in history else None

    if train_loss is not None:
        if not np.isfinite(train_loss).all() or (train_loss > 50).any():
            flags["exploding_gradients"] = True
            notes.append("non-finite or extremely large train_loss")
        if len(train_loss) >= 3:
            early = float(train_loss.iloc[: max(1, len(train_loss) // 3)].mean())
            late = float(train_loss.iloc[-max(1, len(train_loss) // 3) :].mean())
            if early - late < 1e-4 and late > 1.5:
                flags["vanishing_gradients"] = True
                notes.append("train_loss barely decreased while remaining high")

    if train_uar is not None and val_uar is not None and len(history) >= 3:
        gap = float(train_uar.iloc[-1] - val_uar.iloc[-1])
        if gap > 0.15:
            flags["overfitting"] = True
            notes.append(f"train/val UAR gap={gap:.3f}")
        if float(train_uar.max()) < 0.25 and float(val_uar.max()) < 0.25:
            flags["underfitting"] = True
            notes.append("both train and val UAR stayed below 0.25")

    if val_uar is not None and len(val_uar) >= 5:
        recent = val_uar.iloc[-5:]
        if float(recent.max() - recent.min()) < 0.005 and float(val_uar.max()) < 0.55:
            flags["learning_plateau"] = True
            notes.append("validation UAR plateaued at a low level")

    best_epoch_hist = None
    if val_uar is not None and not val_uar.empty and "epoch" in history.columns:
        best_epoch_hist = int(history.loc[val_uar.idxmax(), "epoch"])
    claimed = metrics.get("best_epoch")
    if claimed is not None and best_epoch_hist is not None:
        if abs(int(claimed) - int(best_epoch_hist)) > 1:
            flags["checkpoint_incorrect"] = True
            notes.append(f"best_epoch metrics={claimed} history_argmax={best_epoch_hist}")

    ckpt = run_dir / "best_model.pt"
    is_complete = bool(metrics.get("test_metrics")) or claimed is not None
    if not ckpt.exists():
        flags["checkpoint_incorrect"] = True
        notes.append("best_model.pt missing")

    # Incomplete stubs (history only) are warned, not hard failures.
    severe = flags["exploding_gradients"] or (is_complete and not ckpt.exists())
    return {
        "run_id": run_dir.name,
        "flags": flags,
        "notes": notes,
        "best_epoch_history": best_epoch_hist,
        "best_epoch_metrics": claimed,
        "n_epochs": int(len(history)),
        "passed": not severe,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit training health from run artifacts.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run-dir", default=None, help="Optional single run directory.")
    parser.add_argument("--report", default="POST_TRAINING_AUDIT.md")
    parser.add_argument("--json", default="artifacts/reports/training_health.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    outputs_dir = ROOT / cfg["paths"]["outputs_dir"]
    run_dirs = [from_root(args.run_dir)] if args.run_dir else discover_runs(outputs_dir)

    audits = [audit_run(p) for p in run_dirs]
    if not audits:
        gate = GateResult(
            gate="post_training",
            passed=True,
            checks=[CheckResult("runs_found", True, "no runs yet — nothing to audit")],
            message="no runs",
        )
        recommendations = ["No training runs found. Run training before post-training audit."]
    else:
        checks = [
            CheckResult(
                a["run_id"],
                bool(a["passed"]),
                ", ".join(a["notes"]) if a["notes"] else "healthy",
            )
            for a in audits
        ]
        gate = GateResult(
            gate="post_training",
            passed=all(a["passed"] for a in audits),
            checks=checks,
            message=f"runs={len(audits)}",
        )
        recommendations = []
        for a in audits:
            active = [k for k, v in a["flags"].items() if v]
            if active:
                recommendations.append(f"{a['run_id']}: flags={active}")
        if not recommendations:
            recommendations = ["All audited runs look healthy."]

    health = {"runs": audits, "overall_passed": gate.passed}
    save_json(health, args.json)

    extra = [
        (
            "Flag definitions",
            (
                "- **overfitting:** final train UAR exceeds val UAR by > 0.15\n"
                "- **underfitting:** train and val UAR both < 0.25\n"
                "- **exploding_gradients:** non-finite or extremely large loss\n"
                "- **vanishing_gradients:** train loss barely decreases while high\n"
                "- **learning_plateau:** recent val UAR flat and low\n"
                "- **checkpoint_incorrect:** missing checkpoint or best epoch mismatch"
            ),
        )
    ]
    save_markdown(
        render_gate_report(
            title="Post-Training Audit",
            workflow="post_training_audit",
            gates=[gate],
            recommendations=recommendations,
            extra_sections=extra,
        ),
        ROOT / args.report,
    )
    update_project_state(
        workflow="post_training_audit",
        overall_passed=gate.passed,
        completed_stage="post_training_audit" if gate.passed else None,
        current_stage="gap_analysis" if gate.passed else "post_training_audit",
        next_stage="report_generation" if gate.passed else "post_training_audit",
    )

    print(f"Audited runs: {len(audits)}")
    for a in audits:
        print(f"  {a['run_id']}: passed={a['passed']} flags={[k for k, v in a['flags'].items() if v]}")
    print("POST-TRAINING AUDIT " + ("PASSED" if gate.passed else "FAILED"))
    raise SystemExit(exit_code(gate.passed))


if __name__ == "__main__":
    main()
