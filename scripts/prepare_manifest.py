from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.data import attach_feature_paths, build_manifest
from ser.utils import ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified SER metadata manifest.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg_path = ROOT / args.config
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    roots = {k: v for k, v in cfg["paths"].items() if k in {"ravdess", "cremad", "tess", "savee"}}
    df = build_manifest(roots, project_root=ROOT)
    df = attach_feature_paths(df, ROOT / cfg["paths"]["features_dir"])
    out = ROOT / cfg["paths"]["processed_manifest"]
    ensure_dirs(out.parent)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} samples -> {out}")
    print("\nDataset counts:")
    print(df.groupby(["dataset", "emotion"]).size().unstack(fill_value=0))
    print("\nSpeakers:")
    print(df.groupby("dataset")["speaker"].nunique())


if __name__ == "__main__":
    main()
