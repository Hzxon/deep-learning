"""Golden: metric definitions, model geometry and feature shape. Protected by testgate."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ser.features import FEATURE_SHAPE, _enforce_shape, validate_feature_shape
from ser.metrics import classification_metrics
from ser.models import EXPECTED_INPUT_HW, build_model, uses_augmentation, verify_model_shapes

LABELS = list(range(6))


def test_uar_is_macro_recall_not_accuracy() -> None:
    # Imbalanced: class 0 dominates, class 1 always wrong.
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    out = classification_metrics(y_true, y_pred, labels=[0, 1])
    assert out["accuracy"] == pytest.approx(0.8)
    assert out["uar"] == pytest.approx(0.5)
    assert out["macro_recall"] == pytest.approx(0.5)
    assert out["macro_precision"] == pytest.approx(0.4)
    assert out["macro_f1"] == pytest.approx((2 * 0.8 * 1.0 / 1.8) / 2)
    assert out["confusion_matrix"] == [[8, 0], [2, 0]]
    assert out["per_class"]["1"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 2}
    assert out["per_class"]["0"]["support"] == 8


def test_metrics_uses_target_names_in_report() -> None:
    y = np.array([0, 1, 2, 3, 4, 5])
    names = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
    out = classification_metrics(y, y, labels=LABELS, target_names=names)
    for name in names:
        assert out["classification_report"][name]["recall"] == pytest.approx(1.0)
    assert np.array(out["confusion_matrix"]).trace() == 6


def test_feature_shape_contract() -> None:
    assert FEATURE_SHAPE == (64, 188)
    assert EXPECTED_INPUT_HW == (1, 64, 188)
    validate_feature_shape(np.zeros((64, 188), dtype=np.float32))
    with pytest.raises(ValueError):
        validate_feature_shape(np.zeros((64, 187), dtype=np.float32))


def test_enforce_shape_pads_and_crops() -> None:
    short = np.ones((64, 100), dtype=np.float64)
    out = _enforce_shape(short, n_mels=64, n_frames=188)
    assert out.shape == (64, 188) and out.dtype == np.float32
    assert out[:, :100].sum() == 64 * 100 and out[:, 100:].sum() == 0
    long = np.arange(64 * 200, dtype=np.float32).reshape(64, 200)
    np.testing.assert_array_equal(_enforce_shape(long, 64, 188), long[:, :188])
    with pytest.raises(ValueError):
        _enforce_shape(np.ones((32, 188)), 64, 188)
    with pytest.raises(ValueError):
        _enforce_shape(np.ones(188), 64, 188)


@pytest.mark.parametrize("model_id", ["M0", "M1", "M2", "M3"])
def test_models_emit_six_logits(model_id: str) -> None:
    torch.manual_seed(0)
    model = build_model(model_id, num_classes=6, cfg={"model": {"channels": [8, 16, 32], "lstm_hidden": 16}})
    shapes = verify_model_shapes(model, batch_size=3, num_classes=6)
    assert shapes["input"] == (3, *EXPECTED_INPUT_HW)
    assert shapes["logits"] == (3, 6)


def test_augmentation_only_for_m1_m3() -> None:
    assert [uses_augmentation(m) for m in ("M0", "M1", "M2", "M3")] == [False, True, False, True]
    with pytest.raises(ValueError):
        build_model("M9", num_classes=6, cfg={})
