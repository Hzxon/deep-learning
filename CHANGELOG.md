# Changelog

All notable changes to this SER research repository are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

<!-- testgate smoke check 2 -->
