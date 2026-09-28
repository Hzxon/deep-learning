from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ser.features import FEATURE_SHAPE


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def feature_df(tmp_path: Path) -> pd.DataFrame:
    from prepare_ci_fixtures import build_ci_manifest

    return build_ci_manifest(tmp_path / "features")


def test_feature_shape_contract(feature_df: pd.DataFrame) -> None:
    path = Path(feature_df.iloc[0]["feature_path"])
    arr = np.load(path)
    assert arr.shape == FEATURE_SHAPE
    assert arr.dtype == np.float32


def test_validate_features_pass(feature_df: pd.DataFrame) -> None:
    mod = _load_script("validate_features")
    gate = mod.validate_features(feature_df, max_checks=64)
    assert gate.passed


def test_validate_features_detects_nan(feature_df: pd.DataFrame) -> None:
    mod = _load_script("validate_features")
    path = Path(feature_df.iloc[0]["feature_path"])
    bad = np.load(path)
    bad[0, 0] = np.nan
    np.save(path, bad.astype(np.float32))
    gate = mod.validate_features(feature_df.iloc[:1], max_checks=1)
    assert not gate.passed
