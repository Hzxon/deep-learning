from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.data import DISCARDED_EMOTIONS, LABEL_TO_ID, LABELS

# Import inspect helper
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def ci_manifest(tmp_path: Path) -> pd.DataFrame:
    from prepare_ci_fixtures import build_ci_manifest

    return build_ci_manifest(tmp_path / "features")


def test_label_mapping_is_six_class() -> None:
    assert LABELS == ["angry", "disgust", "fear", "happy", "neutral", "sad"]
    assert len(LABEL_TO_ID) == 6
    assert "calm" in DISCARDED_EMOTIONS


def test_ci_manifest_columns(ci_manifest: pd.DataFrame) -> None:
    required = {"filepath", "dataset", "speaker", "emotion", "label", "feature_path"}
    assert required.issubset(ci_manifest.columns)
    assert set(ci_manifest["emotion"].unique()) == set(LABELS)
    assert ci_manifest["filepath"].is_unique


def test_inspect_manifest_gate(ci_manifest: pd.DataFrame) -> None:
    # Local import after path bootstrap
    import importlib.util

    path = ROOT / "scripts" / "inspect_manifest.py"
    spec = importlib.util.spec_from_file_location("inspect_manifest", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    gate = mod.inspect_manifest(ci_manifest, LABELS)
    assert gate.passed
