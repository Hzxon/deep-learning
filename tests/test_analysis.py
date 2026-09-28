from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.analysis import (
    append_results,
    build_rq1_table,
    build_rq2_table,
    build_rq3_table,
    parse_protocol_and_fold,
    write_rq_markdown_report,
)
from ser.plotting import generate_all_result_plots, plot_training_curves


def _fake_artifact(model_id: str, split: str, seed: int, uar: float) -> dict:
    labels = list(range(6))
    per = {
        str(i): {"precision": uar, "recall": uar, "f1": uar, "support": 10}
        for i in labels
    }
    cm = np.eye(6, dtype=int).tolist()
    protocol, fold = parse_protocol_and_fold(split)
    test_datasets = [fold] if protocol == "leave_one_corpus_out" else ["CREMA-D", "RAVDESS", "TESS", "SAVEE"]
    train_datasets = [c for c in ["CREMA-D", "RAVDESS", "TESS", "SAVEE"] if c not in test_datasets] or ["CREMA-D"]
    return {
        "run_id": f"{split}_{model_id}_seed{seed}".replace("-", "_"),
        "model_id": model_id,
        "split": split,
        "seed": seed,
        "device": "cpu",
        "best_epoch": 3,
        "best_val_uar": uar,
        "test_metrics": {
            "accuracy": uar + 0.01,
            "uar": uar,
            "macro_f1": uar - 0.01,
            "macro_precision": uar,
            "macro_recall": uar,
            "per_class": per,
            "confusion_matrix": cm,
        },
        "train_size": 100,
        "val_size": 20,
        "test_size": 30,
        "train_speakers": 10,
        "val_speakers": 2,
        "test_speakers": 3,
        "train_datasets": train_datasets,
        "test_datasets": test_datasets,
        "elapsed_seconds": 1.0,
    }


@pytest.fixture
def synthetic_results(tmp_path: Path) -> pd.DataFrame:
    rows = []
    base = {
        "M0": 0.55,
        "M1": 0.58,
        "M2": 0.60,
        "M3": 0.63,
    }
    for seed in (42, 43):
        for model, uar in base.items():
            rows.append(_fake_artifact(model, "random", seed, uar + 0.05 + 0.01 * (seed - 42)))
            rows.append(_fake_artifact(model, "speaker_independent", seed, uar + 0.01 * (seed - 42)))
            for corpus in ("CREMA-D", "RAVDESS", "TESS", "SAVEE"):
                rows.append(
                    _fake_artifact(
                        model,
                        f"leave_one_corpus_out_{corpus}",
                        seed,
                        uar - 0.10 + 0.01 * (seed - 42),
                    )
                )
    path = tmp_path / "results.csv"
    labels = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
    df = append_results(rows, path, labels=labels)
    assert path.exists()
    assert set(["protocol", "model", "seed", "dataset_fold", "accuracy", "uar", "macro_f1", "precision", "recall"]).issubset(df.columns)
    return df


def test_parse_protocol_and_fold() -> None:
    assert parse_protocol_and_fold("random") == ("random", "all")
    assert parse_protocol_and_fold("speaker_independent") == ("speaker_independent", "all")
    assert parse_protocol_and_fold("leave_one_corpus_out_CREMA-D")[0] == "leave_one_corpus_out"
    assert parse_protocol_and_fold("leave_one_corpus_out_CREMA-D")[1] == "CREMA-D"


def test_rq_tables_and_markdown(synthetic_results: pd.DataFrame, tmp_path: Path) -> None:
    rq1 = build_rq1_table(synthetic_results)
    rq2 = build_rq2_table(synthetic_results)
    rq3 = build_rq3_table(synthetic_results, protocol="speaker_independent")
    assert not rq1.empty
    assert set(rq1["model"]) == {"M0", "M1", "M2", "M3"}
    assert "delta_uar" in rq1.columns
    assert not rq2.empty
    assert set(rq2["test_corpus"]) == {"CREMA-D", "RAVDESS", "TESS", "SAVEE"}
    assert not rq3.empty
    assert "contrasts" in rq3.attrs
    md = write_rq_markdown_report(synthetic_results, tmp_path / "gap_analysis.md")
    text = md.read_text(encoding="utf-8")
    assert "RQ1" in text and "RQ2" in text and "RQ3" in text
    assert "|" in text


def test_plots_generation(synthetic_results: pd.DataFrame, tmp_path: Path) -> None:
    plots = tmp_path / "plots"
    # Create a fake run history for curve plotting.
    run_dir = tmp_path / "runs" / "demo_run"
    run_dir.mkdir(parents=True)
    hist = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_loss": [1.2, 1.0, 0.8],
            "val_loss": [1.3, 1.1, 0.9],
            "train_accuracy": [0.3, 0.4, 0.5],
            "val_accuracy": [0.25, 0.35, 0.45],
            "train_uar": [0.3, 0.4, 0.5],
            "val_uar": [0.25, 0.35, 0.45],
            "train_macro_f1": [0.2, 0.3, 0.4],
            "val_macro_f1": [0.2, 0.3, 0.35],
            "learning_rate": [1e-3, 1e-3, 5e-4],
        }
    )
    hist.to_csv(run_dir / "history.csv", index=False)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": "demo_run",
                "test_metrics": {
                    "confusion_matrix": np.eye(6).tolist(),
                },
            }
        ),
        encoding="utf-8",
    )
    written = generate_all_result_plots(
        synthetic_results,
        plots,
        labels=["angry", "disgust", "fear", "happy", "neutral", "sad"],
        outputs_dir=tmp_path / "runs",
    )
    assert written
    assert (plots / "uar_by_protocol_model.png").exists()
    assert any(p.name.endswith("_loss_curve.png") for p in written)
    curves = plot_training_curves(run_dir / "history.csv", plots / "curves2", run_id="x")
    assert curves
