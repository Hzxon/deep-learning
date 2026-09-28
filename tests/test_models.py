from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.models import (
    EXPECTED_INPUT_HW,
    CNNBackbone,
    CNNBiLSTMClassifier,
    CNNClassifier,
    build_model,
    count_parameters,
    format_parameter_count,
    uses_augmentation,
    verify_model_shapes,
)


@pytest.fixture
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))


def test_shared_backbone_is_identical_instance_pattern(cfg: dict) -> None:
    backbone = CNNBackbone()
    cnn = CNNClassifier(num_classes=6, backbone=backbone)
    bilstm = CNNBiLSTMClassifier(num_classes=6, backbone=backbone, hidden_size=128)
    assert cnn.backbone is backbone
    assert bilstm.backbone is backbone


def test_cnn_and_bilstm_shapes(cfg: dict) -> None:
    cnn = build_model("M0", num_classes=6, cfg=cfg)
    bilstm = build_model("M2", num_classes=6, cfg=cfg)

    cnn_shapes = verify_model_shapes(cnn, batch_size=4, num_classes=6)
    bilstm_shapes = verify_model_shapes(bilstm, batch_size=4, num_classes=6)

    assert cnn_shapes["input"] == (4, *EXPECTED_INPUT_HW)
    assert cnn_shapes["backbone"][1] == 128
    assert cnn_shapes["backbone"][2:] == (8, 23)  # 64/8, 188→23 after 3 pools
    assert cnn_shapes["logits"] == (4, 6)

    assert bilstm_shapes["input"] == (4, *EXPECTED_INPUT_HW)
    assert bilstm_shapes["backbone"][1] == 128
    assert bilstm_shapes["logits"] == (4, 6)
    assert bilstm.embed_dim == 256  # type: ignore[attr-defined]


def test_input_shape_assertion_rejects_bad_tensor() -> None:
    model = CNNClassifier(num_classes=6)
    bad = torch.randn(2, 1, 64, 100)
    with pytest.raises(AssertionError, match="Expected input"):
        model(bad)


def test_parameter_counting(cfg: dict) -> None:
    model = build_model("M0", num_classes=6, cfg=cfg)
    counts = count_parameters(model)
    assert counts["total"] > 0
    assert counts["trainable"] == counts["total"]
    assert counts["frozen"] == 0
    text = format_parameter_count(model)
    assert "trainable=" in text


def test_uses_augmentation_mapping() -> None:
    assert uses_augmentation("M0") is False
    assert uses_augmentation("M1") is True
    assert uses_augmentation("M2") is False
    assert uses_augmentation("M3") is True


def test_build_all_variants(cfg: dict) -> None:
    for model_id in ("M0", "M1", "M2", "M3"):
        model = build_model(model_id, num_classes=6, cfg=cfg)
        shapes = verify_model_shapes(model, batch_size=2, num_classes=6)
        assert shapes["logits"] == (2, 6)
