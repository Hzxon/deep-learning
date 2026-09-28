from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.data import attach_feature_paths
from ser.features import FEATURE_SHAPE, preprocess_file, validate_feature_shape
from ser.utils import ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("preprocess")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract fixed-shape log-mel features.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute features even when the .npy file already exists.",
    )
    args = parser.parse_args()

    with (ROOT / args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    expected_shape = (
        int(cfg["mel"]["n_mels"]),
        int(cfg["mel"].get("target_frames", FEATURE_SHAPE[1])),
    )
    if expected_shape != FEATURE_SHAPE:
        LOGGER.warning(
            "Config feature shape %s differs from research default %s",
            expected_shape,
            FEATURE_SHAPE,
        )

    manifest_path = ROOT / cfg["paths"]["processed_manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run prepare_manifest.py first.")

    df = pd.read_csv(manifest_path)
    df = attach_feature_paths(df, ROOT / cfg["paths"]["features_dir"])
    ensure_dirs(ROOT / cfg["paths"]["features_dir"])

    path_col = "filepath" if "filepath" in df.columns else "path"
    failures: list[dict[str, str]] = []
    computed = 0
    skipped = 0

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Preprocessing"):
        audio_path = getattr(row, path_col)
        feature_path = Path(row.feature_path)
        try:
            if feature_path.exists() and not args.overwrite:
                arr = np.load(feature_path)
                skipped += 1
            else:
                arr = preprocess_file(
                    audio_path,
                    sample_rate=int(cfg["sample_rate"]),
                    duration_seconds=float(cfg["duration_seconds"]),
                    mel_cfg=cfg["mel"],
                )
                np.save(feature_path, arr.astype(np.float32))
                computed += 1
            validate_feature_shape(arr, expected=expected_shape)
        except Exception as exc:
            failures.append({"path": str(audio_path), "error": repr(exc)})

    if failures:
        fail_path = ROOT / Path(cfg["paths"]["processed_manifest"]).with_suffix(".failures.csv")
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        bad = {x["path"] for x in failures}
        df = df[~df[path_col].isin(bad)].copy()
        print(f"WARNING: {len(failures)} files failed. See {fail_path}")

    df.to_csv(manifest_path, index=False)
    print(f"Saved processed manifest -> {manifest_path}")
    print(f"Feature shape: {expected_shape}")
    print(f"Computed={computed} skipped_existing={skipped} failed={len(failures)} kept={len(df)}")


if __name__ == "__main__":
    main()
