# Changelog

All notable changes to this SER research repository are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- Augmentation now matches log-mel (dB) features. **Runs trained before this
  change are not comparable and must be retrained.**
  - `RandomGain` adds a dB offset instead of multiplying.
  - `TimeStretch` padding and `SpecAugment` masks use the quietest value, not 0 dB (the loudest).
  - `PitchShift` repeats the edge bin instead of wrapping high frequencies to the bottom.
  - `SpecAugment` applies `time_mask_p` / `freq_mask_p` once; `spec_augment_p` is removed.
- `train_and_evaluate` seeds every run, so results no longer depend on run order or
  `--skip-existing`. cuDNN is deterministic, the train loader has a seeded generator, and
  each DataLoader worker gets its own augmentation stream.
- `workflow.py pre-training` writes `PRE_TRAINING_AUDIT.md` from the gates it ran;
  `pre_training_audit.py --skip-nested` used to report PASS for unchecked gates.
- `audit_training.py` fails when a run produces no plots.
- `smoke_test.py` takes `--config` (CI uses `config/ci.yaml`) and always cleans up.
- `run_experiments.py`: `--max-epochs` is a true alias of `--epochs`; splits are built once.

### Removed

- `scripts/validate_pipeline.py` and `scripts/pre_training_audit.py` (superseded by the
  `validate_*` gates and `workflow.py`).
- Duplicate outputs: `checkpoints_dir/<run>.pt`, `results/<run>.json`,
  `results/<run>_history.json`, root `training_health.json` and `VALIDATION_REPORT.md`
  (now under `artifacts/reports/`). The run directory is the single source of truth.
- Unused code: model/dataset aliases, `utils.resolve_path`, `monitor_metric`.
- One-off reviews `IMPLEMENTATION_AUDIT.md` and `RESEARCH_OS_REVIEW.md`.

### Added

- Research Operating System: `scripts/workflow.py`, `scripts/workflows.yaml`,
  validation gates, Cursor rules/prompts/commands, and GitHub Actions
  (`validate.yml`, `nightly-audit.yml`).
- Canonical documentation under `docs/` (`SER_PROJECT_SPEC.md`,
  `VALIDATION_PROTOCOL.md`, `WORKFLOW_GUIDE.md`, `CI_SPEC.md`,
  `PROJECT_STRUCTURE.md`).

### Changed

- Per-run outputs live under `artifacts/runs/` (`paths.outputs_dir`); research
  tables remain in `results/`; Research-OS audit reports in `artifacts/reports/`.
