from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import librosa
import numpy as np

LOGGER = logging.getLogger(__name__)

# Fixed research feature geometry (16 kHz, 3 s, hop=256, centered STFT).
FEATURE_SHAPE: tuple[int, int] = (64, 188)


def load_audio_fixed(
    path: str | Path,
    sample_rate: int = 16000,
    duration_seconds: float = 3.0,
    trim_top_db: int = 30,
) -> np.ndarray:
    """Load mono audio, trim silence, pad/crop to a fixed length, and peak-normalize."""
    y, sr = librosa.load(str(path), sr=None, mono=True)
    y = y.astype(np.float32, copy=False)
    if sr != sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate).astype(np.float32, copy=False)
    if y.size > 32:
        y, _ = librosa.effects.trim(y, top_db=trim_top_db)
    target_len = int(round(sample_rate * duration_seconds))
    if y.size >= target_len:
        start = (y.size - target_len) // 2
        y = y[start : start + target_len]
    else:
        y = np.pad(y, (0, target_len - y.size))
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.0:
        y = (y / peak).astype(np.float32, copy=False)
    return y


def _enforce_shape(log_mel: np.ndarray, n_mels: int, n_frames: int) -> np.ndarray:
    """Pad or crop a log-mel spectrogram to exactly ``(n_mels, n_frames)``."""
    if log_mel.ndim != 2:
        raise ValueError(f"Expected 2D log-mel, got shape {log_mel.shape}")
    mel = log_mel
    if mel.shape[0] != n_mels:
        raise ValueError(f"Expected {n_mels} mel bins, got {mel.shape[0]}")
    cur = mel.shape[1]
    if cur < n_frames:
        mel = np.pad(mel, ((0, 0), (0, n_frames - cur)), mode="constant")
    elif cur > n_frames:
        mel = mel[:, :n_frames]
    return mel.astype(np.float32, copy=False)


def extract_log_mel(y: np.ndarray, sample_rate: int, mel_cfg: dict) -> np.ndarray:
    """Extract a fixed-shape log-mel spectrogram from a waveform."""
    n_mels = int(mel_cfg["n_mels"])
    n_frames = int(mel_cfg.get("target_frames", FEATURE_SHAPE[1]))
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_fft=int(mel_cfg["n_fft"]),
        win_length=int(mel_cfg["win_length"]),
        hop_length=int(mel_cfg["hop_length"]),
        n_mels=n_mels,
        fmin=float(mel_cfg["fmin"]),
        fmax=float(mel_cfg["fmax"]),
        power=float(mel_cfg.get("power", 2.0)),
        center=True,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    return _enforce_shape(log_mel, n_mels=n_mels, n_frames=n_frames)


def preprocess_file(
    path: str | Path,
    sample_rate: int,
    duration_seconds: float,
    mel_cfg: dict,
) -> np.ndarray:
    """Full preprocessing: fixed waveform window → log-mel ``(n_mels, n_frames)``."""
    y = load_audio_fixed(path, sample_rate=sample_rate, duration_seconds=duration_seconds)
    return extract_log_mel(y, sample_rate=sample_rate, mel_cfg=mel_cfg)


def validate_feature_shape(arr: np.ndarray, expected: tuple[int, int] = FEATURE_SHAPE) -> None:
    """Raise if ``arr`` is not exactly the research feature shape."""
    if tuple(arr.shape) != expected:
        raise ValueError(f"Feature shape {tuple(arr.shape)} != expected {expected}")


class SpecTransform(ABC):
    """Base class for log-mel spectrogram augmentations."""

    @abstractmethod
    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class TimeShift(SpecTransform):
    """Circular shift along the time axis."""

    def __init__(self, probability: float, max_shift: int) -> None:
        self.probability = float(probability)
        self.max_shift = int(max_shift)

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.max_shift <= 0 or rng.random() >= self.probability:
            return x
        shift = int(rng.integers(-self.max_shift, self.max_shift + 1))
        if shift == 0:
            return x
        return np.roll(x, shift=shift, axis=-1)


class RandomGain(SpecTransform):
    """Multiply spectrogram by a random gain in linear scale."""

    def __init__(self, probability: float, max_db: float) -> None:
        self.probability = float(probability)
        self.max_db = float(max_db)

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.max_db <= 0 or rng.random() >= self.probability:
            return x
        db = float(rng.uniform(-self.max_db, self.max_db))
        return (x * (10.0 ** (db / 20.0))).astype(np.float32, copy=False)


class GaussianNoise(SpecTransform):
    """Add zero-mean Gaussian noise to the spectrogram."""

    def __init__(self, probability: float, std: float) -> None:
        self.probability = float(probability)
        self.std = float(std)

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.std <= 0 or rng.random() >= self.probability:
            return x
        noise = rng.normal(0.0, self.std, size=x.shape).astype(np.float32)
        return (x + noise).astype(np.float32, copy=False)


class PitchShift(SpecTransform):
    """Approximate pitch shift by rolling along the frequency axis."""

    def __init__(self, probability: float, max_bins: int) -> None:
        self.probability = float(probability)
        self.max_bins = int(max_bins)

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.max_bins <= 0 or rng.random() >= self.probability:
            return x
        shift = int(rng.integers(-self.max_bins, self.max_bins + 1))
        if shift == 0:
            return x
        return np.roll(x, shift=shift, axis=-2)


class TimeStretch(SpecTransform):
    """Approximate time stretch via linear interpolation on the time axis."""

    def __init__(self, probability: float, min_rate: float, max_rate: float) -> None:
        self.probability = float(probability)
        self.min_rate = float(min_rate)
        self.max_rate = float(max_rate)

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.max_rate <= self.min_rate or rng.random() >= self.probability:
            return x
        rate = float(rng.uniform(self.min_rate, self.max_rate))
        if abs(rate - 1.0) < 1e-6:
            return x
        n_mels, n_frames = x.shape
        src_t = np.linspace(0.0, 1.0, num=n_frames, dtype=np.float64)
        new_len = max(2, int(round(n_frames / rate)))
        dst_t = np.linspace(0.0, 1.0, num=new_len, dtype=np.float64)
        stretched = np.vstack([np.interp(dst_t, src_t, x[i]) for i in range(n_mels)]).astype(np.float32)
        return _enforce_shape(stretched, n_mels=n_mels, n_frames=n_frames)


class SpecAugment(SpecTransform):
    """Time and frequency masking (SpecAugment)."""

    def __init__(
        self,
        probability: float,
        time_mask_max: int,
        freq_mask_max: int,
        n_time_masks: int = 1,
        n_freq_masks: int = 1,
        time_mask_p: float | None = None,
        freq_mask_p: float | None = None,
    ) -> None:
        self.probability = float(probability)
        self.time_mask_max = int(time_mask_max)
        self.freq_mask_max = int(freq_mask_max)
        self.n_time_masks = int(n_time_masks)
        self.n_freq_masks = int(n_freq_masks)
        # Optional legacy per-axis probabilities from earlier configs.
        self.time_mask_p = float(time_mask_p) if time_mask_p is not None else self.probability
        self.freq_mask_p = float(freq_mask_p) if freq_mask_p is not None else self.probability

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() >= self.probability:
            return x
        out = x.copy()
        n_mels, n_frames = out.shape
        if rng.random() < self.time_mask_p:
            for _ in range(self.n_time_masks):
                max_width = min(self.time_mask_max, n_frames)
                width = int(rng.integers(0, max_width + 1))
                if width <= 0:
                    continue
                start = int(rng.integers(0, n_frames - width + 1))
                out[:, start : start + width] = 0.0
        if rng.random() < self.freq_mask_p:
            for _ in range(self.n_freq_masks):
                max_width = min(self.freq_mask_max, n_mels)
                width = int(rng.integers(0, max_width + 1))
                if width <= 0:
                    continue
                start = int(rng.integers(0, n_mels - width + 1))
                out[start : start + width, :] = 0.0
        return out


class ComposeAugmentations:
    """Apply a sequence of spectrogram transforms with a seeded RNG."""

    def __init__(self, transforms: Sequence[SpecTransform], seed: int) -> None:
        self.transforms = list(transforms)
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = x
        for transform in self.transforms:
            out = transform(out, self.rng)
        return out.astype(np.float32, copy=False)


def build_augmentations(aug_cfg: dict, seed: int) -> ComposeAugmentations:
    """Build the default augmentation pipeline from a config dictionary."""
    stretch = aug_cfg.get("time_stretch_range", [0.9, 1.1])
    if len(stretch) != 2:
        raise ValueError("augmentation.time_stretch_range must be [min, max]")
    transforms: list[SpecTransform] = [
        TimeShift(
            probability=float(aug_cfg.get("time_shift_p", 0.0)),
            max_shift=int(aug_cfg.get("time_shift_max", 0)),
        ),
        RandomGain(
            probability=float(aug_cfg.get("random_gain_p", 0.0)),
            max_db=float(aug_cfg.get("random_gain_db", 0.0)),
        ),
        GaussianNoise(
            probability=float(aug_cfg.get("noise_p", 0.0)),
            std=float(aug_cfg.get("noise_std", 0.0)),
        ),
        PitchShift(
            probability=float(aug_cfg.get("pitch_shift_p", 0.0)),
            max_bins=int(aug_cfg.get("pitch_shift_max", 0)),
        ),
        TimeStretch(
            probability=float(aug_cfg.get("time_stretch_p", 0.0)),
            min_rate=float(stretch[0]),
            max_rate=float(stretch[1]),
        ),
        SpecAugment(
            probability=float(aug_cfg.get("spec_augment_p", aug_cfg.get("time_mask_p", 0.0))),
            time_mask_max=int(aug_cfg.get("time_mask_max", 0)),
            freq_mask_max=int(aug_cfg.get("freq_mask_max", 0)),
            n_time_masks=int(aug_cfg.get("n_time_masks", 1)),
            n_freq_masks=int(aug_cfg.get("n_freq_masks", 1)),
            time_mask_p=float(aug_cfg["time_mask_p"]) if "time_mask_p" in aug_cfg else None,
            freq_mask_p=float(aug_cfg["freq_mask_p"]) if "freq_mask_p" in aug_cfg else None,
        ),
    ]
    return ComposeAugmentations(transforms, seed=seed)
