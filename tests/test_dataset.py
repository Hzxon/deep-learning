from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ser.data import LABELS


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config" / "ci.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def ci_df(tmp_path: Path) -> pd.DataFrame:
    from prepare_ci_fixtures import build_ci_manifest

    return build_ci_manifest(tmp_path / "features")


def test_validate_dataset_on_ci_manifest(cfg: dict, ci_df: pd.DataFrame) -> None:
    mod = _load_script("validate_dataset")
    gate = mod.validate_dataset(ci_df, cfg, allow_missing_audio=True, max_audio_checks=32)
    assert gate.passed
    assert set(ci_df["emotion"].unique()) == set(LABELS)


def test_validate_dataset_rejects_duplicates(cfg: dict, ci_df: pd.DataFrame) -> None:
    mod = _load_script("validate_dataset")
    bad = pd.concat([ci_df, ci_df.iloc[:1]], ignore_index=True)
    gate = mod.validate_dataset(bad, cfg, allow_missing_audio=True, max_audio_checks=8)
    assert not gate.passed
