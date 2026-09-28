from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.splits import (
    SplitBundle,
    assert_no_speaker_leakage,
    assert_partition_integrity,
    assert_split_integrity,
    build_splits,
    iter_split_bundles,
    leave_one_corpus_out_splits,
    random_split,
    save_all_splits,
    speaker_independent_split,
    speaker_sets,
)


def _synthetic_manifest(rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Build a small multi-corpus frame that mirrors real speaker-count constraints."""
    rng = rng or np.random.default_rng(0)
    rows: list[dict] = []
    corpus_speakers = {
        "CREMA-D": [f"CREMA-D_{i:02d}" for i in range(10)],
        "RAVDESS": [f"RAVDESS_{i:02d}" for i in range(8)],
        "TESS": ["TESS_OAF", "TESS_YAF"],
        "SAVEE": [f"SAVEE_{s}" for s in ("DC", "JE", "JK", "KL")],
    }
    label = 0
    idx = 0
    for dataset, speakers in corpus_speakers.items():
        for speaker in speakers:
            for emotion_id in range(6):
                for _ in range(2):
                    rows.append(
                        {
                            "filepath": f"/fake/{dataset}/{speaker}_{emotion_id}_{idx}.wav",
                            "path": f"/fake/{dataset}/{speaker}_{emotion_id}_{idx}.wav",
                            "dataset": dataset,
                            "speaker": speaker,
                            "emotion": ["angry", "disgust", "fear", "happy", "neutral", "sad"][emotion_id],
                            "label": emotion_id,
                            "feature_path": f"/fake/feat/{idx}.npy",
                            "duration": 2.5,
                            "split": "",
                        }
                    )
                    idx += 1
                    label = (label + 1) % 6
    return pd.DataFrame(rows)


@pytest.fixture
def manifest() -> pd.DataFrame:
    return _synthetic_manifest()


def test_random_split_ratios_and_is_reproducible(manifest: pd.DataFrame) -> None:
    a = random_split(manifest, seed=42)
    b = random_split(manifest, seed=42)
    c = random_split(manifest, seed=43)
    assert_partition_integrity(manifest, a)
    assert list(a.train["filepath"]) == list(b.train["filepath"])
    assert list(a.train["filepath"]) != list(c.train["filepath"])
    total = len(a.train) + len(a.val) + len(a.test)
    assert total == len(manifest)
    # Rough 70/10/20
    assert 0.60 <= len(a.train) / total <= 0.80
    assert 0.05 <= len(a.val) / total <= 0.20
    assert 0.10 <= len(a.test) / total <= 0.30


def test_speaker_independent_has_no_leakage(manifest: pd.DataFrame) -> None:
    bundle = speaker_independent_split(manifest, seed=42)
    assert_split_integrity(manifest, bundle, protocol="speaker_independent")
    train_s, val_s, test_s = speaker_sets(bundle)
    assert train_s.isdisjoint(val_s)
    assert train_s.isdisjoint(test_s)
    assert val_s.isdisjoint(test_s)


def test_tess_special_case_no_val_speaker(manifest: pd.DataFrame) -> None:
    bundle = speaker_independent_split(manifest, seed=7)
    tess = manifest[manifest["dataset"] == "TESS"]
    tess_speakers = set(tess["speaker"].unique())
    train_s, val_s, test_s = speaker_sets(bundle)
    assert len(tess_speakers & train_s) == 1
    assert len(tess_speakers & test_s) == 1
    assert len(tess_speakers & val_s) == 0


def test_leave_one_corpus_out_four_folds_and_no_leakage(manifest: pd.DataFrame) -> None:
    folds = leave_one_corpus_out_splits(manifest, seed=42)
    assert len(folds) == 4
    held_out_names = {name for name, _ in folds}
    assert held_out_names == {"CREMA-D", "RAVDESS", "TESS", "SAVEE"}
    for held_out, bundle in folds:
        assert_split_integrity(manifest, bundle, protocol="leave_one_corpus_out")
        assert set(bundle.test["dataset"].unique()) == {held_out}
        assert held_out not in set(bundle.train["dataset"].unique())
        assert held_out not in set(bundle.val["dataset"].unique())
        # Entire train pool is retained (no discarded pool-test speakers).
        pool = manifest[manifest["dataset"] != held_out]
        assert len(bundle.train) + len(bundle.val) == len(pool)
        assert_no_speaker_leakage(bundle)


def test_loco_does_not_drop_train_pool_samples(manifest: pd.DataFrame) -> None:
    for held_out, bundle in leave_one_corpus_out_splits(manifest, seed=1):
        pool_paths = set(manifest.loc[manifest["dataset"] != held_out, "filepath"])
        used = set(bundle.train["filepath"]) | set(bundle.val["filepath"])
        assert pool_paths == used


def test_build_splits_protocols(manifest: pd.DataFrame) -> None:
    assert len(build_splits(manifest, "random", 0)) == 1
    assert len(build_splits(manifest, "speaker_independent", 0)) == 1
    assert len(build_splits(manifest, "leave_one_corpus_out", 0)) == 4
    with pytest.raises(ValueError):
        build_splits(manifest, "unknown", 0)


def test_save_all_splits_writes_csvs(manifest: pd.DataFrame, tmp_path: Path) -> None:
    written = save_all_splits(manifest, splits_dir=tmp_path, seed=42)
    assert len(written) == 6  # random + SI + 4 LOCO folds
    for out in written:
        assert (out / "train.csv").exists()
        assert (out / "val.csv").exists()
        assert (out / "test.csv").exists()
        assert (out / "all.csv").exists()
        assert (out / "summary.json").exists()
        train = pd.read_csv(out / "train.csv")
        val = pd.read_csv(out / "val.csv")
        test = pd.read_csv(out / "test.csv")
        reloaded = SplitBundle(train=train, val=val, test=test, name=out.name)
        protocol = (
            "leave_one_corpus_out"
            if out.name.startswith("leave_one_corpus_out")
            else "speaker_independent"
            if out.name.startswith("speaker_independent")
            else "random"
        )
        assert_partition_integrity(manifest, reloaded)
        if protocol != "random":
            assert_no_speaker_leakage(reloaded)


def test_real_manifest_if_present() -> None:
    """Optional integration check against the project manifest."""
    manifest_path = ROOT / "data" / "processed" / "manifest.csv"
    if not manifest_path.exists():
        pytest.skip("processed manifest not available")
    df = pd.read_csv(manifest_path)
    for protocol in ("random", "speaker_independent", "leave_one_corpus_out"):
        for fold_name, bundle in iter_split_bundles(df, protocol=protocol, seed=42):
            assert_split_integrity(df, bundle, protocol=protocol)
            _ = fold_name
