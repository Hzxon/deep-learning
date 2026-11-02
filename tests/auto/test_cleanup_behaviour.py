"""Behaviour of the augmentation, seeding, artifact and audit fixes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from ser import data as data_mod
from ser.audit import ROOT, discover_runs, from_root, load_config, save_json
from ser.data import LABELS, SERDataset, attach_feature_paths, seed_augmentation_worker
from ser.features import (
    FEATURE_SHAPE,
    PitchShift,
    RandomGain,
    SpecAugment,
    TimeStretch,
    build_augmentations,
)
from ser.splits import random_split, speaker_independent_split
from ser.train import train_and_evaluate
from ser.utils import seed_everything


def _ramp(n_mels: int = 8, n_frames: int = 20) -> np.ndarray:
    """Spectrogram whose row i holds the value -i (dB, so 0 is loudest)."""
    return -np.repeat(np.arange(n_mels, dtype=np.float32)[:, None], n_frames, axis=1)


def test_random_gain_is_additive_in_db() -> None:
    x = _ramp()
    out = RandomGain(probability=1.0, max_db=6.0)(x, np.random.default_rng(0))
    offset = out - x
    assert np.allclose(offset, offset.flat[0])
    assert 0.0 < abs(offset.flat[0]) <= 6.0
    assert out.dtype == np.float32


def test_random_gain_skips_when_disabled() -> None:
    x = _ramp()
    assert RandomGain(probability=1.0, max_db=0.0)(x, np.random.default_rng(0)) is x
    assert RandomGain(probability=0.0, max_db=6.0)(x, np.random.default_rng(0)) is x


@pytest.mark.parametrize("seed", range(6))
def test_pitch_shift_repeats_edges_instead_of_wrapping(seed: int) -> None:
    x = _ramp()
    out = PitchShift(probability=1.0, max_bins=2)(x, np.random.default_rng(seed))
    rows = out[:, 0]
    # Rows stay monotone non-increasing: nothing from the bottom appears at the top.
    assert np.all(np.diff(rows) <= 0)
    assert rows[0] == x[0, 0] or rows[-1] == x[-1, 0]


def test_pitch_shift_up_and_down() -> None:
    x = _ramp()

    class Fixed:
        def __init__(self, shift: int) -> None:
            self.shift = shift

        def random(self) -> float:
            return 0.0

        def integers(self, low: int, high: int) -> int:
            return self.shift

    up = PitchShift(1.0, 2)(x, Fixed(2))
    assert list(up[:, 0]) == [0, 0, 0, -1, -2, -3, -4, -5]
    down = PitchShift(1.0, 2)(x, Fixed(-2))
    assert list(down[:, 0]) == [-2, -3, -4, -5, -6, -7, -7, -7]
    assert PitchShift(1.0, 2)(x, Fixed(0)) is x


def test_time_stretch_pads_with_quietest_value() -> None:
    x = _ramp()

    class Fast:
        def random(self) -> float:
            return 0.0

        def uniform(self, low: float, high: float) -> float:
            return 2.0

    out = TimeStretch(1.0, 1.5, 2.5)(x, Fast())
    assert out.shape == x.shape
    assert np.all(out[:, 10:] == x.min())
    assert np.allclose(out[:, 0], x[:, 0])


def test_spec_augment_masks_with_quietest_value_per_axis() -> None:
    x = _ramp() + 5.0
    time_only = SpecAugment(time_mask_p=1.0, freq_mask_p=0.0, time_mask_max=20, freq_mask_max=8)
    rng = np.random.default_rng(3)
    out = time_only(x, rng)
    changed = out != x
    assert changed.any()
    assert np.all(out[changed] == x.min())
    # A time mask covers whole columns.
    cols = np.where(changed.any(axis=0))[0]
    assert np.all(out[:, cols] == x.min())

    freq_only = SpecAugment(time_mask_p=0.0, freq_mask_p=1.0, time_mask_max=20, freq_mask_max=8)
    out = freq_only(x, np.random.default_rng(5))
    rows = np.where((out != x).any(axis=1))[0]
    assert rows.size > 0 and np.all(out[rows] == x.min())

    off = SpecAugment(time_mask_p=0.0, freq_mask_p=0.0, time_mask_max=20, freq_mask_max=8)
    assert np.array_equal(off(x, np.random.default_rng(0)), x)


def test_spec_augment_uses_per_axis_probability_only() -> None:
    aug = build_augmentations({"time_mask_p": 1.0, "time_mask_max": 5, "freq_mask_p": 0.0}, seed=0)
    spec = aug.transforms[-1]
    assert isinstance(spec, SpecAugment)
    assert (spec.time_mask_p, spec.freq_mask_p) == (1.0, 0.0)
    hits = sum(bool((spec(_ramp(), np.random.default_rng(s)) != _ramp()).any()) for s in range(50))
    # With p=1 nearly every draw masks (a width of 0 is possible); no outer gate halves it.
    assert hits >= 40


def test_compose_reseed_restarts_stream() -> None:
    aug = build_augmentations({"noise_p": 1.0, "noise_std": 1.0}, seed=1)
    x = np.zeros(FEATURE_SHAPE, dtype=np.float32)
    aug.reseed(99)
    first = aug(x)
    aug.reseed(99)
    assert np.array_equal(aug(x), first)
    aug.reseed(100)
    assert not np.array_equal(aug(x), first)


def _feature_df(tmp_path: Path, n: int = 4) -> pd.DataFrame:
    rows = []
    for i in range(n):
        p = tmp_path / f"f{i}.npy"
        np.save(p, np.full(FEATURE_SHAPE, -float(i), dtype=np.float32))
        rows.append({"feature_path": str(p), "label": i % 6})
    return pd.DataFrame(rows)


def test_worker_init_reseeds_each_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds = SERDataset(_feature_df(tmp_path), augment=True, aug_cfg={"noise_p": 1.0, "noise_std": 1.0}, seed=0)
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: SimpleNamespace(seed=1234, dataset=ds))
    seed_augmentation_worker(0)
    assert ds.transforms is not None
    assert ds.transforms.rng.random() == np.random.default_rng(1234).random()


def test_worker_init_is_noop_outside_workers_and_without_augmentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_augmentation_worker(0)  # main process: get_worker_info() is None
    plain = SERDataset(_feature_df(tmp_path), augment=False)
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: SimpleNamespace(seed=1, dataset=plain))
    seed_augmentation_worker(0)
    assert plain.transforms is None


def test_seed_everything_enables_deterministic_cudnn() -> None:
    seed_everything(5)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    a = torch.rand(3)
    seed_everything(5)
    assert torch.equal(a, torch.rand(3))


def test_attach_feature_paths_accepts_legacy_path_column(tmp_path: Path) -> None:
    df = pd.DataFrame({"path": ["a.wav", "b.wav"], "dataset": ["TESS", "SAVEE"]})
    out = attach_feature_paths(df, tmp_path)
    assert list(out["filepath"]) == ["a.wav", "b.wav"]
    assert out["feature_path"].str.endswith(".npy").all()
    assert data_mod.path_column(out) == "filepath"


@pytest.mark.parametrize(("train_frac", "val_frac"), [(0.0, 0.0), (0.7, 0.3), (0.9, 0.2)])
def test_random_split_rejects_bad_fractions(train_frac: float, val_frac: float) -> None:
    df = pd.DataFrame({"label": [0, 1] * 10, "speaker": ["s"] * 20})
    with pytest.raises(ValueError):
        random_split(df, seed=0, train_frac=train_frac, val_frac=val_frac)


def test_audit_path_helpers(tmp_path: Path) -> None:
    assert from_root("config/config.yaml") == ROOT / "config" / "config.yaml"
    assert from_root(tmp_path) == tmp_path
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("labels: [a, b]\n")
    assert load_config(cfg_file) == {"labels": ["a", "b"]}
    written = save_json({"a": 1}, tmp_path / "x" / "y.json")
    assert json.loads(written.read_text()) == {"a": 1}


def test_discover_runs_filters(tmp_path: Path) -> None:
    assert discover_runs(tmp_path / "missing") == []
    for name, has_history in [("b_run", True), ("a_run", True), ("demo_x", True), ("empty", False)]:
        d = tmp_path / name
        d.mkdir()
        if has_history:
            (d / "history.csv").write_text("epoch\n1\n")
    (tmp_path / "file.txt").write_text("x")
    assert [p.name for p in discover_runs(tmp_path)] == ["a_run", "b_run"]


def _synthetic_manifest(tmp_path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for ds, n_spk in [("CREMA-D", 4), ("TESS", 2)]:
        for s in range(n_spk):
            for label in range(len(LABELS)):
                p = tmp_path / f"{ds}_{s}_{label}.npy"
                np.save(p, (rng.normal(size=FEATURE_SHAPE) - 40 + label).astype(np.float32))
                rows.append({"feature_path": str(p), "dataset": ds, "speaker": f"{ds}_{s}", "label": label})
    return pd.DataFrame(rows)


def _cfg(tmp_path: Path, tag: str) -> dict:
    """Minimal self-contained config (the testgate sandbox has no config/ dir)."""
    return {
        "labels": LABELS,
        "augmentation": {"noise_p": 0.5, "noise_std": 0.1, "time_mask_p": 0.5, "time_mask_max": 10},
        "model": {"channels": [4, 8, 8], "dropout": 0.3},
        "train": {
            "batch_size": 8,
            "learning_rate": 1e-3,
            "max_epochs": 1,
            "mixed_precision": False,
            "tensorboard": False,
            "num_workers": 0,
        },
        "paths": {
            "outputs_dir": str(tmp_path / tag / "runs"),
            "results_dir": str(tmp_path / tag / "results"),
            "tensorboard_dir": str(tmp_path / tag / "tb"),
        },
    }


def test_train_and_evaluate_is_order_independent_and_writes_run_dir_only(tmp_path: Path) -> None:
    split = speaker_independent_split(_synthetic_manifest(tmp_path), seed=1)
    common = dict(
        model_id="M1",
        split_name=split.name,
        train_df=split.train,
        val_df=split.val,
        test_df=split.test,
        labels=LABELS,
        seed=7,
        run_id="r",
        device=torch.device("cpu"),
        max_epochs_override=1,
    )
    first = train_and_evaluate(cfg=_cfg(tmp_path, "a"), **common)
    torch.rand(100)  # disturb global RNG state between runs
    np.random.rand(100)
    second = train_and_evaluate(cfg=_cfg(tmp_path, "b"), **common)
    assert first["test_metrics"]["confusion_matrix"] == second["test_metrics"]["confusion_matrix"]
    assert first["test_loss"] == pytest.approx(second["test_loss"])

    run_dir = tmp_path / "a" / "runs" / "r"
    assert {"best_model.pt", "history.json", "history.csv", "metrics.json", "config.yaml", "report.txt"} <= {
        p.name for p in run_dir.iterdir()
    }
    assert not (tmp_path / "a" / "results").exists()
    assert json.loads((run_dir / "metrics.json").read_text())["run_id"] == "r"
