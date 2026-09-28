from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
TEST_FRAC = 0.20

PROTOCOLS = ("random", "speaker_independent", "leave_one_corpus_out")


@dataclass(frozen=True)
class SplitBundle:
    """Train / validation / test partitions for one protocol fold."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    name: str


def _tag_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    out = df.copy()
    out["split"] = split_name
    return out.reset_index(drop=True)


def random_split(
    df: pd.DataFrame,
    seed: int,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> SplitBundle:
    """Protocol A: sample-level stratified 70 / 10 / 20 split by emotion label."""
    if not np.isclose(train_frac + val_frac + (1.0 - train_frac - val_frac), 1.0):
        raise ValueError("train_frac + val_frac must be < 1")
    train_df, temp_df = train_test_split(
        df,
        test_size=1.0 - train_frac,
        random_state=seed,
        stratify=df["label"],
    )
    val_ratio_in_temp = val_frac / (1.0 - train_frac)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - val_ratio_in_temp,
        random_state=seed,
        stratify=temp_df["label"],
    )
    return SplitBundle(
        _tag_split(train_df, "train"),
        _tag_split(val_df, "val"),
        _tag_split(test_df, "test"),
        "random",
    )


def _speaker_counts(n_speakers: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    """Round speaker quotas to 70/10/20 with at least one train and one test when possible."""
    if n_speakers <= 0:
        raise ValueError("n_speakers must be positive")
    if n_speakers == 1:
        return 1, 0, 0
    if n_speakers == 2:
        # TESS-style: one train, one test, no validation speaker.
        return 1, 0, 1

    n_test = max(1, int(round(n_speakers * fractions[2])))
    n_val = max(1, int(round(n_speakers * fractions[1])))
    n_train = n_speakers - n_test - n_val
    if n_train < 1:
        n_train = 1
        overflow = n_train + n_val + n_test - n_speakers
        if overflow > 0 and n_val > 1:
            n_val -= overflow
        elif overflow > 0:
            n_test = max(1, n_test - overflow)
    # Re-check after adjustments.
    while n_train + n_val + n_test > n_speakers:
        if n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    n_train = n_speakers - n_val - n_test
    return n_train, n_val, n_test


def _assign_speakers_three_way(
    speakers: list[str],
    seed: int,
    fractions: tuple[float, float, float] = (TRAIN_FRAC, VAL_FRAC, TEST_FRAC),
) -> dict[str, str]:
    """Assign speakers to train/val/test with reproducible shuffling."""
    rng = np.random.default_rng(seed)
    order = list(speakers)
    rng.shuffle(order)
    n_train, n_val, n_test = _speaker_counts(len(order), fractions)
    assignment: dict[str, str] = {}
    idx = 0
    for _ in range(n_train):
        assignment[order[idx]] = "train"
        idx += 1
    for _ in range(n_val):
        assignment[order[idx]] = "val"
        idx += 1
    for _ in range(n_test):
        assignment[order[idx]] = "test"
        idx += 1
    assert idx == len(order)
    return assignment


def _assign_speakers_two_way(
    speakers: list[str],
    seed: int,
    train_frac: float = 0.875,
) -> dict[str, str]:
    """Assign speakers to train/val only (used for LOCO train pools)."""
    rng = np.random.default_rng(seed)
    order = list(speakers)
    rng.shuffle(order)
    n = len(order)
    if n == 1:
        return {order[0]: "train"}
    if n == 2:
        return {order[0]: "train", order[1]: "val"}
    n_train = max(1, int(round(n * train_frac)))
    n_train = min(n_train, n - 1)  # keep at least one val speaker
    assignment: dict[str, str] = {}
    for i, speaker in enumerate(order):
        assignment[speaker] = "train" if i < n_train else "val"
    return assignment


def _partition_by_speaker_map(df: pd.DataFrame, assignment: dict[str, str], name: str) -> SplitBundle:
    roles = df["speaker"].map(assignment)
    missing = roles.isna().sum()
    if missing:
        raise RuntimeError(f"{missing} rows have speakers missing from assignment map")
    train_df = df.loc[roles == "train"]
    val_df = df.loc[roles == "val"]
    test_df = df.loc[roles == "test"]
    return SplitBundle(
        _tag_split(train_df, "train"),
        _tag_split(val_df, "val"),
        _tag_split(test_df, "test"),
        name,
    )


def speaker_independent_split(
    df: pd.DataFrame,
    seed: int,
    fractions: tuple[float, float, float] = (TRAIN_FRAC, VAL_FRAC, TEST_FRAC),
) -> SplitBundle:
    """Protocol B: speaker-disjoint 70/10/20 with per-corpus assignment.

    TESS (2 speakers): one speaker → train, one speaker → test.
    No TESS speaker is placed in validation (val is merged into train capacity).
    """
    assignment: dict[str, str] = {}
    for dataset, ddf in df.groupby("dataset", sort=True):
        speakers = sorted(ddf["speaker"].unique().tolist())
        # Offset seed per corpus so assignments are stable but not identical.
        corpus_seed = seed + (sum(ord(c) for c in str(dataset)) % 997)
        if str(dataset).upper() == "TESS" or len(speakers) == 2:
            # Explicit AGENTS rule: train + test only.
            rng = np.random.default_rng(corpus_seed)
            order = speakers.copy()
            rng.shuffle(order)
            assignment[order[0]] = "train"
            assignment[order[1]] = "test"
            LOGGER.info("TESS-style split for %s: train=%s test=%s val=<none>", dataset, order[0], order[1])
            continue
        local = _assign_speakers_three_way(speakers, seed=corpus_seed, fractions=fractions)
        assignment.update(local)

    return _partition_by_speaker_map(df, assignment, "speaker_independent")


def leave_one_corpus_out_splits(df: pd.DataFrame, seed: int) -> list[tuple[str, SplitBundle]]:
    """Protocol C: four folds; held-out corpus is test; train/val from the rest.

    Validation speakers are chosen only from the train pool. The held-out corpus
    is never used for model selection. No train-pool samples are discarded.
    """
    outputs: list[tuple[str, SplitBundle]] = []
    for held_out in sorted(df["dataset"].unique()):
        train_pool = df[df["dataset"] != held_out].copy()
        test_df = df[df["dataset"] == held_out].copy()
        assignment: dict[str, str] = {}
        for dataset, ddf in train_pool.groupby("dataset", sort=True):
            speakers = sorted(ddf["speaker"].unique().tolist())
            corpus_seed = seed + (sum(ord(c) for c in f"{held_out}:{dataset}") % 997)
            local = _assign_speakers_two_way(speakers, seed=corpus_seed)
            assignment.update(local)
        roles = train_pool["speaker"].map(assignment)
        train_df = train_pool.loc[roles == "train"]
        val_df = train_pool.loc[roles == "val"]
        # Any unassigned role should not happen; fail loudly if it does.
        if roles.isna().any() or (roles == "test").any():
            raise RuntimeError(f"Invalid LOCO assignment for held-out corpus {held_out}")
        bundle = SplitBundle(
            _tag_split(train_df, "train"),
            _tag_split(val_df, "val"),
            _tag_split(test_df, "test"),
            f"leave_one_corpus_out_{held_out}",
        )
        outputs.append((str(held_out), bundle))
    return outputs


def build_splits(df: pd.DataFrame, protocol: str, seed: int) -> list[SplitBundle] | list[tuple[str, SplitBundle]]:
    """Build split bundles for a protocol.

    Random / speaker_independent → ``list[SplitBundle]``
    leave_one_corpus_out → ``list[tuple[held_out_corpus, SplitBundle]]``
    """
    if protocol == "random":
        return [random_split(df, seed)]
    if protocol == "speaker_independent":
        return [speaker_independent_split(df, seed)]
    if protocol == "leave_one_corpus_out":
        return leave_one_corpus_out_splits(df, seed)
    raise ValueError(f"Unknown protocol: {protocol}. Expected one of {PROTOCOLS}")


def iter_split_bundles(
    df: pd.DataFrame,
    protocol: str,
    seed: int,
) -> Iterable[tuple[str, SplitBundle]]:
    """Yield ``(fold_name, bundle)`` for any protocol."""
    built = build_splits(df, protocol=protocol, seed=seed)
    for item in built:
        if isinstance(item, tuple):
            held_out, bundle = item
            yield bundle.name, bundle
            _ = held_out
        else:
            yield item.name, item


def speaker_sets(bundle: SplitBundle) -> tuple[set[str], set[str], set[str]]:
    return (
        set(bundle.train["speaker"].unique()),
        set(bundle.val["speaker"].unique()),
        set(bundle.test["speaker"].unique()),
    )


def path_column(df: pd.DataFrame) -> str:
    if "filepath" in df.columns:
        return "filepath"
    if "path" in df.columns:
        return "path"
    raise KeyError("DataFrame requires filepath or path column")


def assert_no_speaker_leakage(bundle: SplitBundle, *, allow_empty_val: bool = True) -> None:
    """Fail if any speaker appears in more than one partition."""
    train_s, val_s, test_s = speaker_sets(bundle)
    if not allow_empty_val and len(val_s) == 0:
        raise AssertionError(f"{bundle.name}: validation speaker set is empty")
    overlaps = [
        ("train/val", train_s & val_s),
        ("train/test", train_s & test_s),
        ("val/test", val_s & test_s),
    ]
    for label, overlap in overlaps:
        if overlap:
            raise AssertionError(f"{bundle.name}: speaker leakage in {label}: {sorted(overlap)[:10]}")


def assert_partition_integrity(df: pd.DataFrame, bundle: SplitBundle) -> None:
    """Verify coverage, disjointness, and no duplicate audio paths."""
    col = path_column(df)
    parts = [bundle.train, bundle.val, bundle.test]
    for part in parts:
        if part.empty and part is bundle.test:
            raise AssertionError(f"{bundle.name}: test partition is empty")
        if not part[col].is_unique:
            raise AssertionError(f"{bundle.name}: duplicate paths inside a partition")

    all_paths = pd.concat([p[col] for p in parts], ignore_index=True)
    if not all_paths.is_unique:
        raise AssertionError(f"{bundle.name}: path overlap across partitions")

    expected = set(df[col].tolist())
    got = set(all_paths.tolist())
    if expected != got:
        missing = sorted(expected - got)[:5]
        extra = sorted(got - expected)[:5]
        raise AssertionError(
            f"{bundle.name}: partition coverage mismatch "
            f"(missing={len(expected - got)} extra={len(got - expected)} "
            f"examples_missing={missing} examples_extra={extra})"
        )


def assert_split_integrity(
    df: pd.DataFrame,
    bundle: SplitBundle,
    *,
    protocol: str,
    require_speaker_disjoint: bool | None = None,
) -> None:
    """Full integrity checks for one split bundle."""
    assert_partition_integrity(df, bundle)
    disjoint = require_speaker_disjoint
    if disjoint is None:
        disjoint = protocol != "random"
    if disjoint:
        assert_no_speaker_leakage(bundle)

    if protocol == "leave_one_corpus_out":
        test_datasets = set(bundle.test["dataset"].unique())
        train_datasets = set(bundle.train["dataset"].unique()) | set(bundle.val["dataset"].unique())
        if len(test_datasets) != 1:
            raise AssertionError(f"{bundle.name}: LOCO test must contain exactly one corpus, got {test_datasets}")
        held_out = next(iter(test_datasets))
        if held_out in train_datasets:
            raise AssertionError(f"{bundle.name}: held-out corpus {held_out} leaked into train/val")
        if set(bundle.test[path_column(bundle.test)]) & set(bundle.train[path_column(bundle.train)]):
            raise AssertionError(f"{bundle.name}: LOCO path leakage")

    # Labels present in every non-empty partition.
    for part_name, part in (("train", bundle.train), ("val", bundle.val), ("test", bundle.test)):
        if part.empty:
            continue
        if part["label"].isna().any():
            raise AssertionError(f"{bundle.name}: NaN labels in {part_name}")


def split_summary(bundle: SplitBundle) -> dict:
    """Compact JSON-serializable summary for logging / artifacts."""
    train_s, val_s, test_s = speaker_sets(bundle)
    return {
        "name": bundle.name,
        "train_samples": int(len(bundle.train)),
        "val_samples": int(len(bundle.val)),
        "test_samples": int(len(bundle.test)),
        "train_speakers": int(len(train_s)),
        "val_speakers": int(len(val_s)),
        "test_speakers": int(len(test_s)),
        "train_datasets": sorted(bundle.train["dataset"].unique().tolist()) if len(bundle.train) else [],
        "val_datasets": sorted(bundle.val["dataset"].unique().tolist()) if len(bundle.val) else [],
        "test_datasets": sorted(bundle.test["dataset"].unique().tolist()) if len(bundle.test) else [],
    }


def save_split_bundle(bundle: SplitBundle, out_dir: Path) -> Path:
    """Write train/val/test CSVs and a summary JSON under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle.train.to_csv(out_dir / "train.csv", index=False)
    bundle.val.to_csv(out_dir / "val.csv", index=False)
    bundle.test.to_csv(out_dir / "test.csv", index=False)
    combined = pd.concat([bundle.train, bundle.val, bundle.test], ignore_index=True)
    combined.to_csv(out_dir / "all.csv", index=False)
    summary = split_summary(bundle)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_dir


def save_all_splits(
    df: pd.DataFrame,
    splits_dir: str | Path,
    seed: int,
    protocols: Iterable[str] = PROTOCOLS,
) -> list[Path]:
    """Generate and persist all requested protocol splits under ``splits_dir``."""
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for protocol in protocols:
        for fold_name, bundle in iter_split_bundles(df, protocol=protocol, seed=seed):
            assert_split_integrity(df, bundle, protocol=protocol)
            out = splits_dir / f"{fold_name}_seed{seed}"
            save_split_bundle(bundle, out)
            written.append(out)
            LOGGER.info(
                "Saved %s -> %s (train=%d val=%d test=%d)",
                fold_name,
                out,
                len(bundle.train),
                len(bundle.val),
                len(bundle.test),
            )
    return written
