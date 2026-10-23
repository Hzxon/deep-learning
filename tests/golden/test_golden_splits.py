"""Golden: split protocols must never leak speakers or corpora. Protected by testgate."""

from __future__ import annotations

import pandas as pd
import pytest

from ser.splits import build_splits, leave_one_corpus_out_splits, random_split, speaker_independent_split


def _manifest() -> pd.DataFrame:
    rows = []
    layout = {"RAVDESS": 10, "CREMA-D": 10, "TESS": 2, "SAVEE": 4}
    for dataset, n_speakers in layout.items():
        for s in range(n_speakers):
            for label in range(6):
                for k in range(2):
                    rows.append(
                        {
                            "filepath": f"{dataset}/{s}/{label}_{k}.wav",
                            "dataset": dataset,
                            "speaker": f"{dataset}_{s:02d}",
                            "label": label,
                        }
                    )
    return pd.DataFrame(rows)


def _speakers(df: pd.DataFrame) -> set[str]:
    return set(df["speaker"])


def test_random_split_partitions_every_row_once() -> None:
    df = _manifest()
    b = random_split(df, seed=42)
    assert len(b.train) + len(b.val) + len(b.test) == len(df)
    all_paths = pd.concat([b.train, b.val, b.test])["filepath"]
    assert all_paths.is_unique
    assert set(all_paths) == set(df["filepath"])
    assert len(b.test) == pytest.approx(0.2 * len(df), abs=2)
    assert len(b.val) == pytest.approx(0.1 * len(df), abs=2)
    assert set(b.train["split"]) == {"train"} and set(b.test["split"]) == {"test"}


def test_random_split_is_reproducible() -> None:
    df = _manifest()
    a = random_split(df, seed=7)
    b = random_split(df, seed=7)
    assert list(a.test["filepath"]) == list(b.test["filepath"])


@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_speaker_independent_is_disjoint(seed: int) -> None:
    df = _manifest()
    b = speaker_independent_split(df, seed=seed)
    tr, va, te = _speakers(b.train), _speakers(b.val), _speakers(b.test)
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(b.train) + len(b.val) + len(b.test) == len(df)
    assert b.name == "speaker_independent"


def test_speaker_independent_tess_has_no_val_speaker() -> None:
    b = speaker_independent_split(_manifest(), seed=3)
    assert "TESS" not in set(b.val["dataset"])
    assert b.train["speaker"][b.train["dataset"] == "TESS"].nunique() == 1
    assert b.test["speaker"][b.test["dataset"] == "TESS"].nunique() == 1


def test_speaker_independent_ten_speakers_quota() -> None:
    b = speaker_independent_split(_manifest(), seed=11)
    counts = [
        part["speaker"][part["dataset"] == "RAVDESS"].nunique() for part in (b.train, b.val, b.test)
    ]
    assert counts == [7, 1, 2]


def test_loco_holds_out_each_corpus_completely() -> None:
    df = _manifest()
    folds = leave_one_corpus_out_splits(df, seed=42)
    assert [name for name, _ in folds] == sorted(df["dataset"].unique())
    for held_out, b in folds:
        assert set(b.test["dataset"]) == {held_out}
        assert len(b.test) == (df["dataset"] == held_out).sum()
        assert held_out not in set(b.train["dataset"]) | set(b.val["dataset"])
        assert not (_speakers(b.train) & _speakers(b.val))
        assert len(b.train) + len(b.val) == (df["dataset"] != held_out).sum()
        assert len(b.val) > 0
        assert b.name == f"leave_one_corpus_out_{held_out}"


def test_build_splits_dispatch() -> None:
    df = _manifest()
    assert len(build_splits(df, "random", 1)) == 1
    assert len(build_splits(df, "speaker_independent", 1)) == 1
    assert len(build_splits(df, "leave_one_corpus_out", 1)) == 4
    with pytest.raises(ValueError, match="Unknown protocol"):
        build_splits(df, "kfold", 1)
