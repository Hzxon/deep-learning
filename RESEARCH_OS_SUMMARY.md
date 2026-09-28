# Research OS Summary

**Installed:** 2026-09-28  
**Purpose:** Validation-first automation for the SER thesis repository.

This document explains every Research OS component so future contributors can operate the repository without memorizing ad-hoc commands.

---

## What was installed

### Documentation (`docs/`)

| File | Role |
|---|---|
| `docs/SER_PROJECT_SPEC.md` | Authoritative research + engineering specification |
| `docs/VALIDATION_PROTOCOL.md` | Eight validation gates with recovery procedures |
| `docs/WORKFLOW_GUIDE.md` | Fresh-clone → final-report playbook |
| `docs/CI_SPEC.md` | Local + GitHub Actions CI architecture |
| `docs/PROJECT_STRUCTURE.md` | Canonical map of every file/folder and its single owner |

Each of these lives in exactly **one** place under `docs/`. Nothing is mirrored to the
repository root anymore — `scripts/check_artifacts.py` now fails CI if a root fork of any
of these four files reappears (`check_no_duplicate_canonical_docs`).

### Cursor automation (`.cursor/`)

| Path | Role |
|---|---|
| `rules/00-research-os.mdc` | Workflow dispatcher (always apply) |
| `rules/01-project-spec.mdc` | Spec authority |
| `rules/02-validation-gates.mdc` | Gate stop/fix/rerun policy |
| `rules/03-training-workflow.mdc` | Pre/post training audits |
| `rules/05-ci-policy.mdc` | Local CI must pass |
| `prompts/*.md` | Reusable audit/review prompt templates |
| `commands/*.sh` | Thin wrappers around `scripts/workflow.py` |

### Workflow orchestrator

| File | Role |
|---|---|
| `scripts/workflow.py` | YAML-driven orchestrator |
| `scripts/workflows.yaml` | Ordered steps for each workflow |
| `PROJECT_STATE.yaml` | Stage machine updated after every workflow |

Supported commands:

```bash
python scripts/workflow.py pre-change
python scripts/workflow.py pre-training
python scripts/workflow.py post-training
python scripts/workflow.py ci-local
python scripts/workflow.py release
python scripts/workflow.py list
```

### Validation & audit scripts

| Script | Gate / job |
|---|---|
| `inspect_manifest.py` | Manifest summary |
| `validate_dataset.py` | Gate 1 Dataset |
| `validate_features.py` | Gate 2 Features |
| `validate_splits.py` | Gate 3 Splits |
| `reproducibility_check.py` | Gate 7 Reproducibility |
| `pre_training_audit.py` | Aggregate pre-training report |
| `post_training_audit.py` | Training health / `POST_TRAINING_AUDIT.md` |
| `audit_training.py` | Annotated training-curve PNGs |
| `check_artifacts.py` | Gate 8 Artifacts (extended) |
| `smoke_test.py` | Gate 5 Smoke (existing) |
| `validate_pipeline.py` | Legacy combined validator (kept) |

Shared helpers live in `src/ser/audit.py`.

### Tests (`tests/`)

| File | Focus |
|---|---|
| `test_manifest.py` | Label/manifest contracts |
| `test_dataset.py` | Dataset validator |
| `test_features.py` | Feature validator |
| `test_splits.py` | Split logic (existing) |
| `test_splits_validation.py` | Split validator script |
| `test_models.py` | Model contracts (existing) |
| `test_metrics.py` | Metric contracts |
| `test_training.py` | Seeding / reproducibility helpers |
| `test_analysis.py` | Gap analysis (existing) |

### CI

| File | Role |
|---|---|
| `.github/workflows/validate.yml` | PR/push quality gate |
| `.github/workflows/nightly-audit.yml` | Scheduled validation + report upload |

---

## Automatic workflows

### Code change → `pre-change`

Ruff → Pytest → Artifact contract check → update `PROJECT_STATE.yaml`.

### Before training → `pre-training`

Dataset → Features → Splits → Smoke → Reproducibility → `PRE_TRAINING_AUDIT.md`.

**Training is forbidden until this PASSes.**

### After training → `post-training`

`post_training_audit.py` + `audit_training.py` → `POST_TRAINING_AUDIT.md` + plots.

### Release → `release`

Runs `ci-local` then artifact checks.

---

## Quality gates

1. **Dataset** — metadata, labels, speakers, audio readability  
2. **Features** — exist, `(64,188)`, float32, no NaN/Inf  
3. **Splits** — stratification / speaker / corpus integrity  
4. **Model** — unit tests for CNN / CNN-BiLSTM contracts  
5. **Training** — smoke one-epoch path  
6. **Evaluation** — metrics + results schema  
7. **Reproducibility** — identical seeded forward passes  
8. **Artifacts** — required files, no notebooks, config contracts  

Any FAIL stops the workflow. Fix, then re-run the same workflow.

---

## How contributors use the system

```bash
# 1. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Before coding
python scripts/workflow.py pre-change

# 3. Before training
python scripts/workflow.py pre-training

# 4. Train (only after PASS)
python scripts/run_experiments.py --config config/config.yaml \
  --protocol speaker_independent --models M0 M1 M2 M3 --seeds 42

# 5. After training
python scripts/workflow.py post-training

# 6. Before PR / merge
python scripts/workflow.py ci-local
```

Read next:

1. [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) — start here to find where anything lives
2. [`docs/SER_PROJECT_SPEC.md`](docs/SER_PROJECT_SPEC.md)
3. [`docs/WORKFLOW_GUIDE.md`](docs/WORKFLOW_GUIDE.md)
4. [`docs/VALIDATION_PROTOCOL.md`](docs/VALIDATION_PROTOCOL.md)
5. [`docs/CI_SPEC.md`](docs/CI_SPEC.md)

---

## Explicit non-goals of this install

- No new CNN / CNN-BiLSTM implementations were added in this task.
- Training logic was not redesigned; only automation hooks and audits were added.
- Existing scripts were retained (`validate_pipeline.py`, `smoke_test.py`, etc.).

## 2026-09-28 reorganization pass

A follow-up cleanup removed structural duplication introduced across multiple
installation prompts:

- Deleted root-level `SER_PROJECT_SPEC.md` / `VALIDATION_PROTOCOL.md` (byte-identical
  forks of the `docs/` originals). `docs/` is now the single canonical location.
- Added `check_no_duplicate_canonical_docs()` to `scripts/check_artifacts.py` so CI fails
  if these files are ever re-forked to the root.
- Removed the empty, unused root `reports/` directory.
- Removed `artifacts/results/` — a stale duplicate of `artifacts/runs/*/metrics.json`
  left over from an earlier config that pointed `outputs_dir` there.
- Stopped `scripts/gap_analysis.py` from double-writing RQ tables and `gap_analysis.md`
  into both `results/` and `artifacts/reports/`. Research-facing tables now live only in
  `results/` (config `paths.results_dir`); `artifacts/reports/` is reserved exclusively
  for Research-OS validation/workflow audit reports.
- Added [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) as the single map of
  "what lives where and why," including which files are hand-authored vs. auto-generated.

See [`IMPLEMENTATION_AUDIT.md`](IMPLEMENTATION_AUDIT.md) for the audit trail of this pass.
