from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SER result tables and plots.")
    parser.add_argument("--results", default="results/results.csv")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--rq3-protocol", default="speaker_independent")
    args = parser.parse_args()
    sys.argv = [
        str(ROOT / "scripts" / "gap_analysis.py"),
        "--results",
        args.results,
        "--config",
        args.config,
        "--rq3-protocol",
        args.rq3_protocol,
    ]
    runpy.run_path(str(ROOT / "scripts" / "gap_analysis.py"), run_name="__main__")


if __name__ == "__main__":
    main()
