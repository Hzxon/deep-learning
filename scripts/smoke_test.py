from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.splits import build_splits
from ser.train import train_and_evaluate
from ser.utils import get_device, seed_everything


def make_fake_manifest(root: Path) -> pd.DataFrame:
    feature_dir = root / "synthetic_features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(7)
    labels = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
    for ds, n_speakers in [("RAVDESS", 6), ("CREMA-D", 6), ("TESS", 2), ("SAVEE", 4)]:
        for speaker_idx in range(n_speakers):
            speaker = f"{ds}_S{speaker_idx:02d}"
            for label_id, emotion in enumerate(labels):
                for sample_idx in range(2):
                    path = (feature_dir / f"{ds}_{speaker_idx}_{label_id}_{sample_idx}.npy").resolve()
                    x = rng.normal(0, 1, size=(64, 188)).astype(np.float32)
                    x += label_id * 0.05
                    np.save(path, x)
                    rows.append({
                        "path": str(path.with_suffix(".wav")),
                        "dataset": ds,
                        "speaker": speaker,
                        "emotion": emotion,
                        "label": label_id,
                        "feature_path": str(path),
                    })
    return pd.DataFrame(rows)


def main() -> None:
    seed_everything(42)
    tmp = ROOT / ".smoke_test"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    df = make_fake_manifest(tmp)
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    cfg["paths"]["checkpoints_dir"] = str(tmp / "checkpoints")
    cfg["paths"]["results_dir"] = str(tmp / "results")
    cfg["paths"]["outputs_dir"] = str(tmp / "runs")
    cfg["paths"]["tensorboard_dir"] = str(tmp / "tensorboard")
    cfg["train"]["tensorboard"] = False
    split = build_splits(df, "speaker_independent", 42)[0]
    assert not (set(split.train.speaker) & set(split.val.speaker))
    assert not (set(split.train.speaker) & set(split.test.speaker))
    assert not (set(split.val.speaker) & set(split.test.speaker))
    assert torch.randn(1, 1, 64, 188).shape[-2:] == (64, 188)

    result = train_and_evaluate(
        cfg=cfg,
        model_id="M0",
        split_name=split.name,
        train_df=split.train,
        val_df=split.val,
        test_df=split.test,
        labels=cfg["labels"],
        seed=42,
        run_id="smoke_M0",
        device=get_device(),
        max_epochs_override=1,
    )
    print("SMOKE TEST PASSED")
    print({k: result[k] for k in ["run_id", "test_loss", "best_epoch", "best_val_uar"]})
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
