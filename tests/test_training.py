from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.models import build_model
from ser.utils import seed_everything


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))


def test_seed_everything_is_deterministic(cfg: dict) -> None:
    seed_everything(123)
    a = torch.randn(4)
    seed_everything(123)
    b = torch.randn(4)
    assert torch.allclose(a, b)


def test_reproducibility_check_script(cfg: dict) -> None:
    mod = _load_script("reproducibility_check")
    gate = mod.reproducibility_check(cfg, seed=42)
    assert gate.passed


def test_build_model_train_eval_toggle(cfg: dict) -> None:
    model = build_model("M0", num_classes=6, cfg=cfg)
    model.train()
    x = torch.randn(2, 1, 64, 188)
    logits = model(x)
    assert logits.shape == (2, 6)
    model.eval()
    with torch.no_grad():
        logits2 = model(x)
    assert logits2.shape == (2, 6)
