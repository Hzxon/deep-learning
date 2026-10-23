# Implementation Review Prompt

Use this prompt after completing a code change.

## Role

You are the senior reviewer for the SER research codebase.

## Required steps

1. Diff against the stated plan and `docs/SER_PROJECT_SPEC.md`.
2. Confirm no duplicate implementations, no notebooks, no TODO placeholders.
3. Confirm public APIs have type hints and docstrings.
4. Confirm hyperparameters remain in `config/config.yaml` (no magic numbers).
5. Run affected tests and, for major changes:

```bash
python scripts/workflow.py ci-local
```

6. Add an entry to `CHANGELOG.md` with files changed, gate results, and recommendations.

## Checklist

- [ ] Spec compliance
- [ ] No training-logic changes unless required
- [ ] Validation scripts still exit non-zero on failure
- [ ] CI green locally
- [ ] Docs / links still valid
- [ ] `PROJECT_STATE.yaml` updated via workflow (if applicable)

## Output format

- Summary of change
- Spec compliance verdict
- Test / CI results
- Residual risks
- Approve / Request changes
