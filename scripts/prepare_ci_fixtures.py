from __future__ import annotations

"""Prepare lightweight synthetic fixtures so CI can run the validate_* gates without raw audio."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.audit import load_config  # noqa: E402
from ser.data import LABEL_TO_ID  # noqa: E402
from ser.utils import ensure_dirs  # noqa: E402


def build_ci_manifest(features_root: Path) -> pd.DataFrame:
    """Create a small multi-corpus manifest compatible with all split protocols."""
    rng = np.random.default_rng(0)
    labels = list(LABEL_TO_ID.keys())
    corpus_speakers = {
        "CREMA-D": [f"CREMA-D_{i:02d}" for i in range(6)],
        "RAVDESS": [f"RAVDESS_{i:02d}" for i in range(6)],
        "TESS": ["TESS_OAF", "TESS_YAF"],
        "SAVEE": [f"SAVEE_{s}" for s in ("DC", "JE", "JK", "KL")],
    }
    rows: list[dict] = []
    idx = 0
    for dataset, speakers in corpus_speakers.items():
        ds_dir = features_root / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        for speaker in speakers:
            for label_id, emotion in enumerate(labels):
                for _ in range(2):
                    stem = f"{speaker}_{emotion}_{idx:04d}"
                    feat = ds_dir / f"{stem}.npy"
                    arr = rng.normal(0.0, 1.0, size=(64, 188)).astype(np.float32)
                    arr += float(label_id) * 0.02
                    np.save(feat, arr)
                    fake_audio = str((ds_dir / f"{stem}.wav").resolve())
                    rows.append(
                        {
                            "filepath": fake_audio,
                            "path": fake_audio,
                            "dataset": dataset,
                            "speaker": speaker,
                            "emotion": emotion,
                            "label": label_id,
                            "duration": 3.0,
                            "split": "",
                            "feature_path": str(feat.resolve()),
                        }
                    )
                    idx += 1
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CI synthetic manifest and features.")
    parser.add_argument("--config", default="config/ci.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    features_dir = ROOT / cfg["paths"]["features_dir"]
    manifest_path = ROOT / cfg["paths"]["processed_manifest"]
    ensure_dirs(features_dir, manifest_path.parent)

    df = build_ci_manifest(features_dir)
    df.to_csv(manifest_path, index=False)
    print(f"CI fixtures ready: samples={len(df)} -> {manifest_path}")
    print(df.groupby("dataset")["speaker"].nunique().to_string())


if __name__ == "__main__":
    main()
