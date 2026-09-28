from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def ci_df(tmp_path: Path):
    from prepare_ci_fixtures import build_ci_manifest

    return build_ci_manifest(tmp_path / "features")


def test_validate_splits_on_ci_manifest(ci_df) -> None:
    mod = _load_script("validate_splits")
    gate = mod.validate_splits(ci_df, seed=42)
    assert gate.passed
