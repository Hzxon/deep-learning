from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

# Non-interactive backend required for CLI / CI / headless macOS runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import normalize_results_frame

LOGGER = logging.getLogger(__name__)


def _save_fig(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_metric_by_protocol_model(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    *,
    title: str | None = None,
) -> Path | None:
    """Grouped bar chart: protocols on x-axis, models as series."""
    df = normalize_results_frame(df)
    if metric not in df.columns or df.empty:
        return None
    pivot = df.groupby(["protocol", "model"])[metric].mean().unstack("model")
    if pivot.empty:
        return None
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel(metric)
    ax.set_xlabel("Evaluation protocol")
    ax.set_title(title or f"{metric} by protocol and model")
    ax.legend(title="Model")
    ax.tick_params(axis="x", labelrotation=20)
    return _save_fig(ax.get_figure(), out_path)


def plot_metric_by_model(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    *,
    protocol: str | None = None,
    title: str | None = None,
) -> Path | None:
    """Bar chart comparing models for one metric (optionally one protocol)."""
    df = normalize_results_frame(df)
    if protocol is not None:
        df = df[df["protocol"] == protocol]
    if df.empty or metric not in df.columns:
        return None
    means = df.groupby("model")[metric].mean()
    stds = df.groupby("model")[metric].std(ddof=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(means.index.astype(str), means.values, yerr=stds.values, capsize=4)
    ax.set_ylabel(metric)
    ax.set_xlabel("Model")
    ax.set_title(title or f"{metric} by model" + (f" ({protocol})" if protocol else ""))
    return _save_fig(fig, out_path)


def plot_training_curves(history_csv: Path, out_dir: Path, run_id: str | None = None) -> list[Path]:
    """Plot loss / accuracy / UAR / Macro-F1 curves from a history.csv file."""
    history_csv = Path(history_csv)
    if not history_csv.exists():
        return []
    hist = pd.read_csv(history_csv)
    if "epoch" not in hist.columns:
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = run_id or history_csv.parent.name
    written: list[Path] = []

    curve_specs = [
        ("loss", ["train_loss", "val_loss"], "Loss"),
        ("accuracy", ["train_accuracy", "val_accuracy"], "Accuracy"),
        ("uar", ["train_uar", "val_uar"], "UAR"),
        ("macro_f1", ["train_macro_f1", "val_macro_f1"], "Macro-F1"),
    ]
    for name, cols, ylabel in curve_specs:
        present = [c for c in cols if c in hist.columns]
        if not present:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for col in present:
            ax.plot(hist["epoch"], hist[col], marker="o", label=col)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{prefix}: {ylabel}")
        ax.legend()
        written.append(_save_fig(fig, out_dir / f"{prefix}_{name}_curve.png"))
    return written


def plot_confusion_matrix(
    matrix: list[list[float]] | np.ndarray,
    labels: list[str],
    out_path: Path,
    *,
    title: str = "Confusion Matrix",
) -> Path:
    """Save a confusion-matrix heatmap PNG."""
    cm = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    return _save_fig(fig, out_path)


def plot_confusion_from_metrics_json(metrics_json: Path, labels: list[str], out_path: Path) -> Path | None:
    path = Path(metrics_json)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("test_metrics", payload)
    cm = metrics.get("confusion_matrix")
    if cm is None:
        return None
    title = f"Confusion Matrix ({payload.get('run_id', path.stem)})"
    return plot_confusion_matrix(cm, labels=labels, out_path=out_path, title=title)


def generate_all_result_plots(
    df: pd.DataFrame,
    plots_dir: Path,
    *,
    labels: list[str] | None = None,
    outputs_dir: Path | None = None,
) -> list[Path]:
    """Generate protocol/model comparison plots and optional run-level curves."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = normalize_results_frame(df)
    written: list[Path] = []

    for metric, title in (
        ("uar", "UAR by protocol and model"),
        ("macro_f1", "Macro-F1 by protocol and model"),
        ("accuracy", "Accuracy by protocol and model"),
    ):
        path = plot_metric_by_protocol_model(
            df, metric, plots_dir / f"{metric}_by_protocol_model.png", title=title
        )
        if path is not None:
            written.append(path)

    for protocol in ("random", "speaker_independent"):
        path = plot_metric_by_model(
            df,
            "uar",
            plots_dir / f"uar_by_model_{protocol}.png",
            protocol=protocol,
            title=f"UAR by model ({protocol})",
        )
        if path is not None:
            written.append(path)

    # Per-class recall under LOCO.
    label_cols = [c for c in df.columns if c.endswith("_recall")]
    loco = df[df["protocol"] == "leave_one_corpus_out"]
    if label_cols and not loco.empty:
        class_mean = loco.groupby("model")[label_cols].mean().T
        class_mean.index = [x.replace("_recall", "") for x in class_mean.index]
        ax = class_mean.plot(kind="bar", figsize=(10, 5))
        ax.set_ylabel("Recall")
        ax.set_xlabel("Emotion")
        ax.set_title("Per-emotion recall under leave-one-corpus-out")
        ax.legend(title="Model")
        written.append(_save_fig(ax.get_figure(), plots_dir / "cross_corpus_per_class_recall.png"))

    if outputs_dir is not None and Path(outputs_dir).exists():
        for run_dir in sorted(Path(outputs_dir).iterdir()):
            if not run_dir.is_dir():
                continue
            history = run_dir / "history.csv"
            if history.exists():
                written.extend(plot_training_curves(history, plots_dir / "curves", run_id=run_dir.name))
            metrics_json = run_dir / "metrics.json"
            if metrics_json.exists() and labels:
                cm_path = plot_confusion_from_metrics_json(
                    metrics_json,
                    labels=labels,
                    out_path=plots_dir / "confusion" / f"{run_dir.name}_confusion.png",
                )
                if cm_path is not None:
                    written.append(cm_path)

    LOGGER.info("Wrote %d plots under %s", len(written), plots_dir)
    return written
