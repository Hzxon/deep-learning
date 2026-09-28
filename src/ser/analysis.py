from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

RESULT_COLUMNS = [
    "protocol",
    "model",
    "seed",
    "dataset_fold",
    "accuracy",
    "uar",
    "macro_f1",
    "precision",
    "recall",
]


def parse_protocol_and_fold(split_name: str) -> tuple[str, str]:
    """Map a split bundle name to ``(protocol, dataset_fold)``."""
    name = str(split_name)
    if name == "random":
        return "random", "all"
    if name == "speaker_independent":
        return "speaker_independent", "all"
    if name.startswith("leave_one_corpus_out"):
        # leave_one_corpus_out_CREMA-D or leave_one_corpus_out_CREMA_D
        suffix = name.replace("leave_one_corpus_out_", "", 1)
        # Prefer original corpus spelling when present.
        for corpus in ("CREMA-D", "RAVDESS", "TESS", "SAVEE"):
            compact = corpus.replace("-", "_")
            if suffix in {corpus, compact} or suffix.replace("-", "_") == compact:
                return "leave_one_corpus_out", corpus
        return "leave_one_corpus_out", suffix
    return name, "all"


def artifact_to_result_row(artifact: dict[str, Any], labels: list[str] | None = None) -> dict[str, Any]:
    """Convert a training artifact dict into a flat results.csv row."""
    metrics = artifact["test_metrics"]
    protocol, fold = parse_protocol_and_fold(artifact["split"])
    # Prefer held-out corpus from test_datasets for LOCO when available.
    if protocol == "leave_one_corpus_out":
        test_ds = artifact.get("test_datasets") or []
        if isinstance(test_ds, list) and len(test_ds) == 1:
            fold = str(test_ds[0])
        elif isinstance(test_ds, str) and test_ds:
            fold = test_ds.split(";")[0]

    row: dict[str, Any] = {
        "run_id": artifact.get("run_id", ""),
        "protocol": protocol,
        "model": artifact["model_id"],
        "model_id": artifact["model_id"],  # backward compatible
        "seed": int(artifact["seed"]),
        "dataset_fold": fold,
        "split": artifact["split"],
        "device": artifact.get("device", ""),
        "best_epoch": artifact.get("best_epoch"),
        "best_val_uar": artifact.get("best_val_uar"),
        "accuracy": float(metrics["accuracy"]),
        "uar": float(metrics["uar"]),
        "macro_f1": float(metrics["macro_f1"]),
        "precision": float(metrics.get("macro_precision", metrics.get("precision", 0.0))),
        "recall": float(metrics.get("macro_recall", metrics.get("recall", metrics["uar"]))),
        "train_size": artifact.get("train_size"),
        "val_size": artifact.get("val_size"),
        "test_size": artifact.get("test_size"),
        "train_speakers": artifact.get("train_speakers"),
        "val_speakers": artifact.get("val_speakers"),
        "test_speakers": artifact.get("test_speakers"),
        "train_datasets": ";".join(artifact.get("train_datasets") or []),
        "test_datasets": ";".join(artifact.get("test_datasets") or []),
        "elapsed_seconds": artifact.get("elapsed_seconds"),
    }

    per_class = metrics.get("per_class", {})
    label_names = labels or []
    for idx, label in enumerate(label_names):
        per = per_class.get(str(idx), {})
        row[f"{label}_precision"] = float(per.get("precision", 0.0))
        row[f"{label}_recall"] = float(per.get("recall", 0.0))
        row[f"{label}_f1"] = float(per.get("f1", 0.0))
    return row


def append_results(
    artifacts: list[dict[str, Any]],
    path: Path,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Append/upsert experiment rows into ``results.csv`` and return the full frame."""
    rows = [artifact_to_result_row(art, labels=labels) for art in artifacts]
    new = pd.DataFrame(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, new], ignore_index=True)
        if "run_id" in combined.columns:
            combined = combined.drop_duplicates(subset=["run_id"], keep="last")
        else:
            combined = combined.drop_duplicates(
                subset=["protocol", "model", "seed", "dataset_fold"],
                keep="last",
            )
    else:
        combined = new
    # Stable column order: required first, then extras.
    ordered = [c for c in RESULT_COLUMNS if c in combined.columns]
    extras = [c for c in combined.columns if c not in ordered]
    combined = combined[ordered + extras]
    combined.to_csv(path, index=False)
    return combined


def _mean_std(series: pd.Series) -> tuple[float, float]:
    mean = float(series.mean()) if len(series) else float("nan")
    std = float(series.std(ddof=0)) if len(series) else float("nan")
    return mean, std


def format_mean_std(mean: float, std: float, digits: int = 4) -> str:
    if np.isnan(mean):
        return "n/a"
    if np.isnan(std) or len(np.atleast_1d(std)) == 0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def normalize_results_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure protocol/model/dataset_fold columns exist for older CSVs."""
    out = df.copy()
    if "model" not in out.columns and "model_id" in out.columns:
        out["model"] = out["model_id"]
    if "model_id" not in out.columns and "model" in out.columns:
        out["model_id"] = out["model"]
    if "protocol" not in out.columns and "split" in out.columns:
        parsed = out["split"].map(parse_protocol_and_fold)
        out["protocol"] = [p for p, _ in parsed]
        out["dataset_fold"] = [f for _, f in parsed]
    if "dataset_fold" not in out.columns:
        out["dataset_fold"] = "all"
    if "precision" not in out.columns:
        out["precision"] = out.get("macro_precision", out.get("uar"))
    if "recall" not in out.columns:
        out["recall"] = out.get("macro_recall", out.get("uar"))
    return out


def build_rq1_table(df: pd.DataFrame) -> pd.DataFrame:
    """RQ1: Random vs Speaker-Independent (mean ± std across seeds)."""
    df = normalize_results_frame(df)
    rows: list[dict[str, Any]] = []
    models = sorted(df["model"].unique().tolist())
    for model in models:
        sub = df[df["model"] == model]
        random_df = sub[sub["protocol"] == "random"]
        si_df = sub[sub["protocol"] == "speaker_independent"]
        row: dict[str, Any] = {"model": model}
        for metric in ("accuracy", "uar", "macro_f1"):
            r_mean, r_std = _mean_std(random_df[metric])
            s_mean, s_std = _mean_std(si_df[metric])
            row[f"random_{metric}"] = r_mean
            row[f"random_{metric}_std"] = r_std
            row[f"speaker_independent_{metric}"] = s_mean
            row[f"speaker_independent_{metric}_std"] = s_std
            row[f"delta_{metric}"] = s_mean - r_mean  # SI - random (usually negative)
            row[f"random_{metric}_fmt"] = format_mean_std(r_mean, r_std)
            row[f"speaker_independent_{metric}_fmt"] = format_mean_std(s_mean, s_std)
            row[f"delta_{metric}_fmt"] = f"{(s_mean - r_mean):.4f}" if not np.isnan(s_mean - r_mean) else "n/a"
        rows.append(row)
    return pd.DataFrame(rows)


def build_rq2_table(df: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
    """RQ2: Leave-One-Corpus-Out table with train corpora / test corpus."""
    df = normalize_results_frame(df)
    loco = df[df["protocol"] == "leave_one_corpus_out"].copy()
    if loco.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "train_corpora",
                "test_corpus",
                "accuracy",
                "uar",
                "macro_f1",
                "accuracy_fmt",
                "uar_fmt",
                "macro_f1_fmt",
            ]
        )
    if model is not None:
        loco = loco[loco["model"] == model]

    all_corpora = sorted({c for c in loco["dataset_fold"].unique().tolist() if c and c != "all"})
    rows: list[dict[str, Any]] = []
    group_cols = ["model", "dataset_fold"]
    for (model_id, test_corpus), g in loco.groupby(group_cols, dropna=False):
        train_corpora = [c for c in all_corpora if c != test_corpus]
        # Prefer explicit train_datasets column when present.
        if "train_datasets" in g.columns and g["train_datasets"].notna().any():
            joined = str(g["train_datasets"].dropna().iloc[0])
            if joined:
                train_corpora = [x for x in joined.split(";") if x]
        row: dict[str, Any] = {
            "model": model_id,
            "train_corpora": " + ".join(train_corpora),
            "test_corpus": test_corpus,
        }
        for metric in ("accuracy", "uar", "macro_f1"):
            mean, std = _mean_std(g[metric])
            row[metric] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_fmt"] = format_mean_std(mean, std)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "test_corpus"]).reset_index(drop=True)


def build_rq3_table(df: pd.DataFrame, protocol: str = "speaker_independent") -> pd.DataFrame:
    """RQ3: 2x2 ablation (M0–M3) with delta improvements for one protocol."""
    df = normalize_results_frame(df)
    sub = df[df["protocol"] == protocol].copy()
    if sub.empty:
        # Fall back to any available non-LOCO protocol.
        candidates = [p for p in ("speaker_independent", "random") if p in set(df["protocol"])]
        if candidates:
            protocol = candidates[0]
            sub = df[df["protocol"] == protocol].copy()

    rows: list[dict[str, Any]] = []
    for model in ("M0", "M1", "M2", "M3"):
        g = sub[sub["model"] == model]
        row: dict[str, Any] = {
            "protocol": protocol,
            "model": model,
            "augmentation": model in {"M1", "M3"},
            "bilstm": model in {"M2", "M3"},
        }
        for metric in ("accuracy", "uar", "macro_f1"):
            mean, std = _mean_std(g[metric])
            row[metric] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_fmt"] = format_mean_std(mean, std)
        rows.append(row)
    table = pd.DataFrame(rows)

    # Delta improvements vs M0 and pairwise contrasts.
    means = {r["model"]: r for r in rows if r["model"] in {"M0", "M1", "M2", "M3"}}
    if len(means) == 4:
        for metric in ("accuracy", "uar", "macro_f1"):
            m0 = means["M0"][metric]
            for model in ("M0", "M1", "M2", "M3"):
                table.loc[table["model"] == model, f"delta_vs_M0_{metric}"] = means[model][metric] - m0
        contrasts = {
            "aug_effect_cnn": means["M1"]["uar"] - means["M0"]["uar"],
            "temporal_effect_no_aug": means["M2"]["uar"] - means["M0"]["uar"],
            "aug_effect_cnn_bilstm": means["M3"]["uar"] - means["M2"]["uar"],
            "temporal_effect_with_aug": means["M3"]["uar"] - means["M1"]["uar"],
            "aug_x_temporal_interaction": (means["M3"]["uar"] - means["M2"]["uar"])
            - (means["M1"]["uar"] - means["M0"]["uar"]),
        }
        table.attrs["contrasts"] = contrasts
    return table


def dataframe_to_markdown(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Render a DataFrame as a GitHub-flavored Markdown table."""
    if df.empty:
        return "_No data available._\n"
    view = df if columns is None else df.loc[:, [c for c in columns if c in df.columns]]
    headers = list(view.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in view.iterrows():
        cells = [str(row[c]) for c in headers]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_rq_markdown_report(
    df: pd.DataFrame,
    out_path: Path,
    *,
    rq3_protocol: str = "speaker_independent",
) -> Path:
    """Write a Markdown report containing RQ1/RQ2/RQ3 tables."""
    df = normalize_results_frame(df)
    rq1 = build_rq1_table(df)
    rq2 = build_rq2_table(df)
    rq3 = build_rq3_table(df, protocol=rq3_protocol)

    parts = [
        "# SER Gap Analysis\n",
        "## RQ1. Random vs Speaker-Independent\n",
        "Metrics are mean ± std across seeds. Δ is speaker-independent − random.\n",
        dataframe_to_markdown(
            rq1,
            columns=[
                "model",
                "random_accuracy_fmt",
                "speaker_independent_accuracy_fmt",
                "delta_accuracy_fmt",
                "random_uar_fmt",
                "speaker_independent_uar_fmt",
                "delta_uar_fmt",
                "random_macro_f1_fmt",
                "speaker_independent_macro_f1_fmt",
                "delta_macro_f1_fmt",
            ],
        ).replace("_fmt", ""),
        "\n## RQ2. Leave-One-Corpus-Out\n",
        dataframe_to_markdown(
            rq2,
            columns=["model", "train_corpora", "test_corpus", "accuracy_fmt", "uar_fmt", "macro_f1_fmt"],
        ).replace("_fmt", ""),
        f"\n## RQ3. Ablation (M0–M3) on `{rq3_protocol}`\n",
        dataframe_to_markdown(
            rq3,
            columns=[
                "model",
                "augmentation",
                "bilstm",
                "accuracy_fmt",
                "uar_fmt",
                "macro_f1_fmt",
                "delta_vs_M0_uar",
            ],
        ).replace("_fmt", ""),
    ]
    contrasts = rq3.attrs.get("contrasts")
    if contrasts:
        parts.append("\n### RQ3 UAR contrasts\n")
        contrast_df = pd.DataFrame(
            [{"contrast": k, "delta_uar": f"{v:.4f}"} for k, v in contrasts.items()]
        )
        parts.append(dataframe_to_markdown(contrast_df))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path
