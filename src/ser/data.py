from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .features import FEATURE_SHAPE, ComposeAugmentations, build_augmentations
from .splits import path_column

LOGGER = logging.getLogger(__name__)

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
LABEL_TO_ID = {x: i for i, x in enumerate(LABELS)}
DISCARDED_EMOTIONS = frozenset({"calm", "surprise", "pleasant_surprise", "pleasant_surprised"})


@dataclass(frozen=True)
class Sample:
    """One labeled utterance after emotion normalization."""

    filepath: str
    dataset: str
    speaker: str
    emotion: str
    label: int
    duration: float


def _wav_files(root: Path) -> list[Path]:
    """Recursively discover WAV files under a dataset root."""
    if not root.exists():
        LOGGER.warning("Dataset root does not exist: %s", root)
        return []
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    LOGGER.info("Found %d WAV files under %s", len(files), root)
    return files


def _audio_duration(path: Path) -> float:
    """Return duration in seconds without fully decoding the file."""
    try:
        return float(sf.info(str(path)).duration)
    except Exception:
        LOGGER.debug("Failed to read duration for %s", path, exc_info=True)
        return float("nan")


def _resolve_root(root_value: str | Path, project_root: Path | None = None) -> Path:
    root = Path(root_value)
    if not root.is_absolute() and project_root is not None:
        root = project_root / root
    return root


def parse_ravdess(path: Path) -> tuple[str, str] | None:
    """Parse RAVDESS filenames (speech or song modality).

    Supports both flat ``Actor_XX/`` and Kaggle ``audio_speech_actors_01-24/Actor_XX/`` layouts.
    """
    parts = path.stem.split("-")
    if len(parts) != 7:
        return None
    try:
        emotion_id = int(parts[2])
    except ValueError:
        return None
    # 1=neutral, 2=calm(discard), 3=happy, 4=sad, 5=angry, 6=fear, 7=disgust, 8=surprise(discard)
    emotion_map = {1: "neutral", 3: "happy", 4: "sad", 5: "angry", 6: "fear", 7: "disgust"}
    emotion = emotion_map.get(emotion_id)
    if emotion is None:
        return None
    actor_match = re.search(r"Actor[_-]?(\d+)", str(path.parent), re.I)
    speaker = actor_match.group(1).zfill(2) if actor_match else parts[-1].zfill(2)
    return f"RAVDESS_{speaker}", emotion


def parse_cremad(path: Path) -> tuple[str, str] | None:
    """Parse CREMA-D filenames such as ``1001_DFA_ANG_XX.wav``."""
    tokens = path.stem.split("_")
    if len(tokens) < 3:
        return None
    emotion_map = {
        "ANG": "angry",
        "DIS": "disgust",
        "FEA": "fear",
        "HAP": "happy",
        "NEU": "neutral",
        "SAD": "sad",
    }
    emotion = emotion_map.get(tokens[2].upper())
    if emotion is None:
        return None
    speaker = tokens[0]
    return f"CREMA-D_{speaker}", emotion


def parse_tess(path: Path) -> tuple[str, str] | None:
    """Parse TESS layouts including nested Kaggle folder names.

    Accepts ``OAF_angry/`` / ``YAF_Fear/`` and discards pleasant-surprise variants.
    """
    joined = str(path).replace("\\", "/")
    m = re.search(r"/(OAF|YAF)_([^/]+)/[^/]+\.wav$", joined, re.I)
    if not m:
        # Fallback: emotion encoded in the filename, e.g. OAF_back_angry.wav
        m_file = re.match(r"(OAF|YAF)_.+_([A-Za-z]+)$", path.stem, re.I)
        if not m_file:
            return None
        speaker_prefix = m_file.group(1).upper()
        folder_emotion = m_file.group(2).lower()
    else:
        speaker_prefix = m.group(1).upper()
        folder_emotion = m.group(2).lower()

    folder_emotion = folder_emotion.replace("pleasant_surprised", "surprise").replace(
        "pleasant_surprise", "surprise"
    )
    if folder_emotion == "ps":
        folder_emotion = "surprise"
    if folder_emotion in DISCARDED_EMOTIONS:
        return None
    emotion_map = {
        "angry": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "happy": "happy",
        "neutral": "neutral",
        "sad": "sad",
    }
    emotion = emotion_map.get(folder_emotion)
    if emotion is None:
        return None
    return f"TESS_{speaker_prefix}", emotion


def parse_savee(path: Path) -> tuple[str, str] | None:
    """Parse SAVEE layouts.

    Layout A (official / Kaggle AudioData): ``AudioData/{DC,JE,JK,KL}/a01.wav``
    Layout B (flattened): ``DC_a01.wav`` or ``ALL/DC_a01.wav``
    """
    emotion_map = {
        "a": "angry",
        "d": "disgust",
        "f": "fear",
        "h": "happy",
        "n": "neutral",
        "sa": "sad",
        "su": None,
    }

    parent = path.parent.name.upper()
    if parent in {"DC", "JE", "JK", "KL"}:
        m = re.match(r"((?:sa|su|[adfhn]))(\d+)\.wav$", path.name, re.I)
        if not m:
            return None
        code = m.group(1).lower()
        emotion = emotion_map.get(code)
        if emotion is None:
            return None
        return f"SAVEE_{parent}", emotion

    m = re.match(r"([A-Za-z]+)_((?:sa|su|[adfhn]))(\d+)\.wav$", path.name, re.I)
    if not m:
        return None
    speaker = m.group(1).upper()
    if speaker not in {"DC", "JE", "JK", "KL"}:
        return None
    code = m.group(2).lower()
    emotion = emotion_map.get(code)
    if emotion is None:
        return None
    return f"SAVEE_{speaker}", emotion


PARSERS: dict[str, Callable[[Path], tuple[str, str] | None]] = {
    "RAVDESS": parse_ravdess,
    "CREMA-D": parse_cremad,
    "TESS": parse_tess,
    "SAVEE": parse_savee,
}

ROOT_NAME_TO_DATASET = {
    "ravdess": "RAVDESS",
    "cremad": "CREMA-D",
    "crema-d": "CREMA-D",
    "crema_d": "CREMA-D",
    "tess": "TESS",
    "savee": "SAVEE",
}


def _dedupe_key(dataset: str, speaker: str, path: Path) -> str:
    """Identity for Kaggle duplicate trees (same speaker + basename)."""
    return f"{dataset}|{speaker}|{path.name.lower()}"


def _path_preference(path: Path) -> tuple[int, int, str]:
    """Lower is better when choosing among duplicate Kaggle copies."""
    text = str(path).replace("\\", "/").lower()
    # Prefer the nested official RAVDESS speech package when both trees exist.
    preferred = 0 if "audio_speech_actors" in text else 1
    # Prefer AudioData / AudioWAV package folders over ad-hoc copies.
    if "/audiodata/" in text or "/audiowav/" in text:
        preferred = min(preferred, 0)
    return (preferred, len(text), text)


def build_manifest(
    roots: dict[str, str | Path],
    project_root: Path | None = None,
) -> pd.DataFrame:
    """Scan dataset roots and build a unified six-class metadata table.

    Duplicate WAV copies (common in Kaggle RAVDESS extracts) are collapsed using
    ``(dataset, speaker, basename)``.
    """
    best: dict[str, Sample] = {}
    for root_key, root_value in roots.items():
        dataset = ROOT_NAME_TO_DATASET.get(root_key.lower(), root_key.upper())
        parser = PARSERS.get(dataset)
        if parser is None:
            raise ValueError(f"No parser registered for dataset: {dataset}")
        root = _resolve_root(root_value, project_root=project_root)
        files = _wav_files(root)
        if not files:
            LOGGER.warning("No WAV files found under %s", root)
            continue
        kept = 0
        skipped = 0
        for path in files:
            parsed = parser(path)
            if parsed is None:
                skipped += 1
                continue
            speaker, emotion = parsed
            if emotion not in LABEL_TO_ID:
                skipped += 1
                continue
            sample = Sample(
                filepath=str(path.resolve()),
                dataset=dataset,
                speaker=speaker,
                emotion=emotion,
                label=LABEL_TO_ID[emotion],
                duration=_audio_duration(path),
            )
            key = _dedupe_key(dataset, speaker, path)
            prev = best.get(key)
            if prev is None or _path_preference(path) < _path_preference(Path(prev.filepath)):
                best[key] = sample
            kept += 1
        LOGGER.info("%s: accepted=%d skipped=%d unique_keys=%d", dataset, kept, skipped, len(best))

    if not best:
        raise RuntimeError("No valid samples found. Check dataset roots and file layouts.")

    rows = [
        {
            "filepath": s.filepath,
            "path": s.filepath,  # backward-compatible alias used by existing scripts
            "dataset": s.dataset,
            "speaker": s.speaker,
            "emotion": s.emotion,
            "label": s.label,
            "duration": s.duration,
            "split": "",
        }
        for s in best.values()
    ]
    df = (
        pd.DataFrame(rows)
        .sort_values(["dataset", "speaker", "filepath"])
        .reset_index(drop=True)
    )
    return df


def attach_feature_paths(df: pd.DataFrame, features_dir: str | Path) -> pd.DataFrame:
    """Attach deterministic feature paths under ``features/{dataset}/{sha1}.npy``."""
    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    path_col = path_column(df)
    feature_paths: list[str] = []
    for raw_path, dataset in zip(df[path_col].tolist(), df["dataset"].tolist(), strict=True):
        digest = hashlib.sha1(str(raw_path).encode("utf-8")).hexdigest()
        out = features_dir / str(dataset) / f"{digest}.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        feature_paths.append(str(out.resolve()))
    out_df = df.copy()
    out_df["feature_path"] = feature_paths
    if "filepath" not in out_df.columns and "path" in out_df.columns:
        out_df["filepath"] = out_df["path"]
    if "path" not in out_df.columns and "filepath" in out_df.columns:
        out_df["path"] = out_df["filepath"]
    return out_df


class SERDataset(Dataset):
    """PyTorch dataset over precomputed log-mel features.

    Augmentation is applied only when ``augment=True`` (training split).
    Validation and test must pass ``augment=False``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        augment: bool = False,
        aug_cfg: dict | None = None,
        mean: float = 0.0,
        std: float = 1.0,
        seed: int = 42,
        transforms: ComposeAugmentations | None = None,
        expected_shape: tuple[int, int] = FEATURE_SHAPE,
    ) -> None:
        if "feature_path" not in df.columns:
            raise ValueError("DataFrame must contain a feature_path column")
        if "label" not in df.columns:
            raise ValueError("DataFrame must contain a label column")
        self.df = df.reset_index(drop=True)
        self.augment = bool(augment)
        self.mean = float(mean)
        self.std = max(float(std), 1e-6)
        self.expected_shape = expected_shape
        if transforms is not None:
            self.transforms = transforms
        elif self.augment:
            if aug_cfg is None:
                raise ValueError("aug_cfg is required when augment=True and transforms is None")
            self.transforms = build_augmentations(aug_cfg, seed=seed)
        else:
            self.transforms = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        arr = np.load(row["feature_path"]).astype(np.float32, copy=False)
        if tuple(arr.shape) != self.expected_shape:
            raise ValueError(
                f"Unexpected feature shape {arr.shape} for {row['feature_path']}; "
                f"expected {self.expected_shape}"
            )
        if self.transforms is not None:
            arr = self.transforms(arr)
        x = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)
        x = (x - self.mean) / self.std
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return x, y


def seed_augmentation_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` giving each worker its own augmentation stream.

    Without it every worker inherits a copy of the same RNG, so all workers draw
    identical augmentations and the stream restarts each epoch. ``worker_info.seed``
    differs per worker and per epoch and derives from the loader's seeded generator.
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    if isinstance(dataset, SERDataset) and dataset.transforms is not None:
        dataset.transforms.reseed(info.seed)
