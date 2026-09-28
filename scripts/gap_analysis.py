from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.analysis import (
    build_rq1_table,
    build_rq2_table,
    build_rq3_table,
    normalize_results_frame,
    write_rq_markdown_report,
)
from ser.plotting import generate_all_result_plots
from ser.utils import ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("gap_analysis")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RQ1/RQ2/RQ3 tables and plots from results.csv.")
    parser.add_argument("--results", default="results/results.csv")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--rq3-protocol",
        default="speaker_independent",
        help="Protocol used for the RQ3 ablation table.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    results_path = Path(args.results)
    if not results_path.is_absolute():
        results_path = ROOT / results_path
    if not results_path.exists():
        # Fall back to configured artifacts path.
        alt = ROOT / cfg["paths"]["results_dir"] / "results.csv"
        if alt.exists():
            results_path = alt
        else:
            raise FileNotFoundError(
                f"Results not found at {args.results} or {alt}. Run scripts/run_experiments.py first."
            )

    df = normalize_results_frame(pd.read_csv(results_path))

    # results_dir is the single canonical home for research-facing tables
    # (results.csv, RQ tables, gap_analysis.md). artifacts/reports/ is reserved
    # for Research-OS validation/workflow audit reports — do not mix the two.
    results_dir = ROOT / cfg["paths"]["results_dir"]
    plots_dir = ROOT / cfg["paths"]["plots_dir"]
    ensure_dirs(results_dir, plots_dir)

    # CSV tables
    rq1 = build_rq1_table(df)
    rq2 = build_rq2_table(df)
    rq3 = build_rq3_table(df, protocol=args.rq3_protocol)

    rq1.to_csv(results_dir / "rq1_random_vs_speaker_independent.csv", index=False)
    rq2.to_csv(results_dir / "rq2_leave_one_corpus_out.csv", index=False)
    rq3.to_csv(results_dir / "rq3_ablation.csv", index=False)

    if rq3.attrs.get("contrasts"):
        contrast_df = pd.DataFrame(
            [{"contrast": k, "delta_uar": v} for k, v in rq3.attrs["contrasts"].items()]
        )
        contrast_df.to_csv(results_dir / "rq3_ablation_deltas.csv", index=False)

    summary = (
        df.groupby(["protocol", "model", "dataset_fold"], dropna=False)[["accuracy", "uar", "macro_f1", "precision", "recall"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(results_dir / "summary_by_protocol_model.csv", index=False)

    # Markdown table
    md_path = write_rq_markdown_report(
        df,
        results_dir / "gap_analysis.md",
        rq3_protocol=args.rq3_protocol,
    )

    # Plots
    outputs_dir = ROOT / cfg["paths"].get("outputs_dir", "artifacts/runs")
    written_plots = generate_all_result_plots(
        df,
        plots_dir,
        labels=list(cfg.get("labels", [])),
        outputs_dir=outputs_dir,
    )

    print(f"Results source -> {results_path}")
    print(f"Markdown report -> {md_path}")
    print(f"CSV tables -> {results_dir}")
    print(f"Plots ({len(written_plots)}) -> {plots_dir}")
    print("GAP ANALYSIS PASSED")


if __name__ == "__main__":
    main()
