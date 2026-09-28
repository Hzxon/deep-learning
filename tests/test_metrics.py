from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.metrics import classification_metrics


def test_metrics_perfect_prediction() -> None:
    y = np.array([0, 1, 2, 3, 4, 5])
    out = classification_metrics(y, y, labels=list(range(6)), target_names=["a", "d", "f", "h", "n", "s"])
    assert out["accuracy"] == pytest.approx(1.0)
    assert out["uar"] == pytest.approx(1.0)
    assert out["macro_f1"] == pytest.approx(1.0)
    assert out["macro_precision"] == pytest.approx(1.0)
    assert out["macro_recall"] == pytest.approx(1.0)
    assert np.array(out["confusion_matrix"]).shape == (6, 6)


def test_metrics_handles_errors() -> None:
    y_true = np.array([0, 1, 2, 3, 4, 5])
    y_pred = np.array([0, 0, 2, 3, 4, 5])
    out = classification_metrics(y_true, y_pred, labels=list(range(6)))
    assert 0.0 < out["accuracy"] < 1.0
    assert "per_class" in out
    assert "classification_report" in out
