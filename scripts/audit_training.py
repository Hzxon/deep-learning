#!/usr/bin/env python3
"""Generate annotated training curves for completed runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.audit import exit_code, save_json  # noqa: E402
from ser.plotting import plot_training_curves  # noqa: E402


def annotate_best_epoch(history_csv: Path, out_dir: Path, run_id: str) -> list[Path]:
    """Plot curves and mark the best validation-UAR epoch."""
    hist = pd.read_csv(history_csv)
    if "epoch" not in hist.columns or "val_uar" not in hist.columns:
        return plot_training_curves(history_csv, out_dir, run_id=run_id)

    best_idx = hist["val_uar"].astype(float).idxmax()
    best_epoch = int(hist.loc[best_idx, "epoch"])
    best_uar = float(hist.loc[best_idx, "val_uar"])

    written = plot_training_curves(history_csv, out_dir, run_id=run_id)
    # Overlay annotation on UAR curve if present.
    uar_path = out_dir / f"{run_id}_uar_curve.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist["epoch"], hist["val_uar"], marker="o", label="val_uar")
    if "train_uar" in hist.columns:
        ax.plot(hist["epoch"], hist["train_uar"], marker="o", label="train_uar")
    ax.axvline(best_epoch, color="red", linestyle="--", label=f"best epoch={best_epoch}")
    ax.scatter([best_epoch], [best_uar], color="red", zorder=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("UAR")
    ax.set_title(f"{run_id}: UAR (best={best_uar:.4f} @ {best_epoch})")
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(uar_path, dpi=200)
    plt.close(fig)
    if uar_path not in written:
        written.append(uar_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit training curves and annotate best epoch.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--json", default="artifacts/reports/audit_training.json")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    outputs_dir = ROOT / cfg["paths"]["outputs_dir"]
    plots_dir = ROOT / cfg["paths"].get("plots_dir", "artifacts/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        run_dirs = [
            p
            for p in sorted(outputs_dir.iterdir())
            if p.is_dir() and (p / "history.csv").exists() and not p.name.startswith("demo_")
        ] if outputs_dir.exists() else []

    summary: list[dict] = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        run_id = run_dir.name
        if metrics_path.exists():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            run_id = str(payload.get("run_id", run_id))
        out = plots_dir / run_dir.name
        written = annotate_best_epoch(run_dir / "history.csv", out, run_id)
        summary.append({"run_id": run_id, "plots": [str(p) for p in written]})
        print(f"OK {run_id}: wrote {len(written)} plots -> {out}")

    save_json({"runs": summary}, ROOT / args.json)
    print(f"AUDIT TRAINING DONE runs={len(summary)}")
    raise SystemExit(exit_code(True))


if __name__ == "__main__":
    main()
