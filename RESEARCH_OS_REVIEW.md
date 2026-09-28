# Research OS Review

**Reviewer role:** Senior ML infrastructure  
**Date:** 2026-09-28  
**Scope:** Post-install integrity of the SER Research Operating System  
**Features added:** None (consistency fixes only)  
**Overall:** **PASS** (36/36 automated checks; `ci-local` PASS after remediation)

Machine-readable check log: `artifacts/reports/research_os_review_checks.yaml`

---

## Verdict

The Research OS is installed, wired, and internally consistent after a remediation pass.
Canonical docs live only under `docs/`, workflows map 1:1 to `scripts/workflows.yaml`,
GitHub Actions reference existing scripts, and Cursor rules agree on gate order.

---

## Checklist results

| Check | Result | Notes |
|---|---|---|
| Every Research-OS-requested file exists | PASS | 41 inventory paths present (docs, `.cursor/`, scripts, GHA, state) |
| No duplicate implementations | PASS | Single `CNNBackbone` / `build_model`; `make_report.py` is a thin wrapper; `validate_pipeline.py` retained as documented legacy |
| No duplicate canonical docs at root | PASS | Spec/validation/workflow/CI docs only under `docs/` |
| No notebooks | PASS | — |
| Cursor rules internally consistent | PASS | After fix: pre-training smoke→repro; CI Dataset→Features→Splits |
| `workflow.py` maps every workflow | PASS | `pre-change`, `pre-training`, `post-training`, `ci-local`, `release` (+ `list`) |
| `workflows.yaml` references existing scripts only | PASS | All `scripts/*.py` refs resolve |
| GitHub Actions valid commands | PASS | `validate.yml` / `nightly-audit.yml`; no `continue-on-error` keys |
| `PROJECT_STATE.yaml` initialized correctly | PASS | Required keys + non-empty `stages`; `last_result` PASS/FAIL |
| Documentation matches repository structure | PASS | After fix: `artifacts/runs/`, `CHANGELOG.md`, structure map updated |

---

## Findings fixed in this pass

| ID | Severity | Finding | Fix |
|---|---|---|---|
| F1 | High | Pre-training step order disagreed: `workflows.yaml` / `pre_training_audit.py` / `RESEARCH_OS_SUMMARY.md` ran Reproducibility before Smoke; Cursor rules + `VALIDATION_PROTOCOL` require Gate 5 then Gate 7 | Reordered to Dataset → Features → Splits → **Smoke** → **Reproducibility** → audit report |
| F2 | Medium | `.cursor/rules/05-ci-policy.mdc` listed Split before Feature; contradicted `validate.yml` / `CI_SPEC` / `ci-local` | Aligned stage list (incl. Prepare CI fixtures) |
| F3 | Medium | Dead `use_ci_config: true` in `workflows.yaml` unused by `workflow.py` | Removed; noted that `ci-local` hardcodes `config/ci.yaml` |
| F4 | Low | `release` workflow ran identical `check_artifacts` twice (`docs_present` duplicate) | Dropped redundant step |
| F5 | Medium | Spec / `AGENTS.md` still described `outputs/`; code uses `artifacts/runs/` | Updated `docs/SER_PROJECT_SPEC.md` §13.1 and `AGENTS.md` logging/structure |
| F6 | Medium | `CHANGELOG.md` referenced by Category E / release prompt but missing | Added hand-authored `CHANGELOG.md` |
| F7 | Low | Gate 5 wording in `02-validation-gates.mdc` pointed at wrong primary script | Clarified `smoke_test.py` + pre-training audit |
| F8 | Low | `REQUIRED_FILES` omitted CI/report helpers (`prepare_ci_fixtures`, `make_report`, `utils.py`, `CHANGELOG`) | Extended `scripts/check_artifacts.py` |
| F9 | Low | `PROJECT_STRUCTURE.md` did not list `CHANGELOG.md` / `RESEARCH_OS_REVIEW.md` | Updated top-level map |

---

## Intentional non-issues (documented)

| Item | Rationale |
|---|---|
| Missing rule `04-*.mdc` | Numbering gap only; five rules cover dispatcher, spec, gates, training, CI |
| `PRE_TRAINING_AUDIT.md` absent until `pre-training` runs | Documented in `PROJECT_STRUCTURE.md` |
| `validate_pipeline.py` retained | Legacy combined validator; not a second gate implementation |
| `training_health.json` at root + `artifacts/reports/` | Root mirror is the documented contract for post-training |
| `completed_stages` mixes pipeline stages and workflow markers | Append-only log of `completed_stage` values from workflows; `stages:` remains the research pipeline |
| Gates 4/6 not inside `pre-training` YAML | Covered by pytest (`test_models` / `test_metrics`) in `pre-change` / `ci-local` |

---

## Workflow mapping (canonical)

| Workflow | Orchestrator | Steps (ordered) |
|---|---|---|
| `pre-change` | `workflow.py` / `audit.sh` | ruff → pytest → check_artifacts |
| `pre-training` | `workflow.py` / `pretrain.sh` | validate_dataset → validate_features → validate_splits → smoke_test → reproducibility_check → pre_training_audit |
| `post-training` | `workflow.py` / `posttrain.sh` | post_training_audit → audit_training |
| `ci-local` | `workflow.py` / `ci_local.sh` | ruff → pytest → prepare_ci_fixtures → dataset → features → splits → smoke → artifacts (`config/ci.yaml`) |
| `release` | `workflow.py` | ci-local → check_artifacts (`config/config.yaml`) |

---

## `PROJECT_STATE.yaml` snapshot (at review time)

- Required keys present: `completed_stages`, `current_stage`, `next_stage`, `blocked_conditions`, `last_workflow`, `last_updated`, `last_result`, `stages`
- `stages` enumerates the research pipeline from `environment_setup` through `release`
- State is overwritten by `ser.audit.update_project_state` after each workflow

---

## Files touched by remediation

- `scripts/workflows.yaml`
- `scripts/pre_training_audit.py`
- `scripts/check_artifacts.py`
- `.cursor/rules/02-validation-gates.mdc`
- `.cursor/rules/05-ci-policy.mdc`
- `RESEARCH_OS_SUMMARY.md`
- `docs/SER_PROJECT_SPEC.md`
- `docs/PROJECT_STRUCTURE.md`
- `AGENTS.md`
- `CHANGELOG.md` (created)
- `RESEARCH_OS_REVIEW.md` (this file)

---

## Follow-up (operator, not blockers)

1. Run `python scripts/workflow.py pre-training` before the next production training matrix (generates `PRE_TRAINING_AUDIT.md`).
2. On the next tagged release, append a dated section to `CHANGELOG.md`.

`ci-local` was re-run after this remediation and **PASSED**.
