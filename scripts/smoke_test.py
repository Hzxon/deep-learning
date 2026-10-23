"""End-to-end smoke test: synthetic features -> speaker-independent split -> 1-epoch M0 run."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.audit import load_config
from ser.data import LABELS
from ser.features import FEATURE_SHAPE
from ser.splits import assert_no_speaker_leakage, build_splits
from ser.train import train_and_evaluate
from ser.utils import get_device


def make_fake_manifest(root: Path) -> pd.DataFrame:
    feature_dir = root / "synthetic_features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(7)
    for ds, n_speakers in [("RAVDESS", 6), ("CREMA-D", 6), ("TESS", 2), ("SAVEE", 4)]:
        for speaker_idx in range(n_speakers):
            speaker = f"{ds}_S{speaker_idx:02d}"
            for label_id, emotion in enumerate(LABELS):
                for sample_idx in range(2):
                    path = (feature_dir / f"{ds}_{speaker_idx}_{label_id}_{sample_idx}.npy").resolve()
                    x = rng.normal(0, 1, size=FEATURE_SHAPE).astype(np.float32)
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
    parser = argparse.ArgumentParser(description="Run the end-to-end smoke test.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    tmp = ROOT / ".smoke_test"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        _run(tmp, load_config(args.config))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run(tmp: Path, cfg: dict) -> None:
    df = make_fake_manifest(tmp)
    cfg["paths"]["results_dir"] = str(tmp / "results")
    cfg["paths"]["outputs_dir"] = str(tmp / "runs")
    cfg["paths"]["tensorboard_dir"] = str(tmp / "tensorboard")
    cfg["train"]["tensorboard"] = False
    split = build_splits(df, "speaker_independent", 42)[0]
    assert_no_speaker_leakage(split)

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


if __name__ == "__main__":
    main()
