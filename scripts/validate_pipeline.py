from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.splits import PROTOCOLS, assert_split_integrity, iter_split_bundles, speaker_sets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    df = pd.read_csv(ROOT / cfg["paths"]["processed_manifest"])

    path_col = "filepath" if "filepath" in df.columns else "path"
    assert set(cfg["labels"]) == set(df["emotion"].unique()), "Manifest labels do not match config labels."
    assert df[path_col].is_unique, "Duplicate audio paths found."
    assert df["speaker"].notna().all()
    assert "feature_path" in df.columns, "feature_path missing. Run preprocess.py."
    assert df["feature_path"].apply(lambda p: Path(p).exists()).all(), "Some features are missing. Run preprocess.py."

    for protocol in PROTOCOLS:
        for fold_name, bundle in iter_split_bundles(df, protocol=protocol, seed=args.seed):
            assert_split_integrity(df, bundle, protocol=protocol)
            train_s, val_s, test_s = speaker_sets(bundle)
            if protocol != "random":
                assert not (train_s & val_s), f"{fold_name}: train/val speaker leakage"
                assert not (train_s & test_s), f"{fold_name}: train/test speaker leakage"
                assert not (val_s & test_s), f"{fold_name}: val/test speaker leakage"
            print(
                f"OK {fold_name}: "
                f"n={len(bundle.train)+len(bundle.val)+len(bundle.test)} "
                f"speakers={len(train_s)}/{len(val_s)}/{len(test_s)}"
            )

    print("VALIDATION PASSED")
    print(f"samples={len(df)} speakers={df.speaker.nunique()} datasets={df.dataset.nunique()}")


if __name__ == "__main__":
    main()
