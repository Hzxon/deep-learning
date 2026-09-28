from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.splits import PROTOCOLS, assert_split_integrity, iter_split_bundles, save_all_splits, split_summary
from ser.utils import ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducible SER train/val/test splits.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--protocol",
        choices=list(PROTOCOLS) + ["all"],
        default="all",
        help="Which protocol(s) to generate.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument(
        "--splits-dir",
        default=None,
        help="Override output directory (default: paths.splits_dir from config).",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    manifest_path = ROOT / cfg["paths"]["processed_manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    splits_dir = Path(args.splits_dir) if args.splits_dir else ROOT / cfg["paths"]["splits_dir"]
    if not splits_dir.is_absolute():
        splits_dir = ROOT / splits_dir
    ensure_dirs(splits_dir)

    protocols = list(PROTOCOLS) if args.protocol == "all" else [args.protocol]
    written = save_all_splits(df, splits_dir=splits_dir, seed=seed, protocols=protocols)

    print(f"\nWrote {len(written)} split folders under {splits_dir}")
    for protocol in protocols:
        for fold_name, bundle in iter_split_bundles(df, protocol=protocol, seed=seed):
            assert_split_integrity(df, bundle, protocol=protocol)
            summary = split_summary(bundle)
            leak = "n/a (random)" if protocol == "random" else "none"
            print(
                f"  {fold_name}: "
                f"train={summary['train_samples']} "
                f"val={summary['val_samples']} "
                f"test={summary['test_samples']} "
                f"speakers(train/val/test)="
                f"{summary['train_speakers']}/{summary['val_speakers']}/{summary['test_speakers']} "
                f"speaker_leakage={leak}"
            )
    print("SPLIT GENERATION PASSED")


if __name__ == "__main__":
    main()
