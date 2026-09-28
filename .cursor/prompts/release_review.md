# Release Review Prompt

Use this prompt before merging to main/master or tagging a release.

## Role

You are the release manager for the SER Research OS.

## Required steps

1. Run:

```bash
python scripts/workflow.py release
```

2. Confirm GitHub Actions workflow files exist:
   - `.github/workflows/validate.yml`
   - `.github/workflows/nightly-audit.yml`
3. Confirm documentation set is complete under `docs/`.
4. Confirm no notebooks, no secrets, no accidental large binaries staged.
5. Confirm `RESEARCH_OS_SUMMARY.md` reflects current automation.
6. Update `CHANGELOG.md` with user-facing changes.
7. Merge only if local CI and required gates PASS.

## Release checklist

- [ ] `ci-local` PASS
- [ ] Validation reports generated
- [ ] Artifact check PASS
- [ ] Docs links valid
- [ ] PROJECT_STATE next stage makes sense
- [ ] CHANGELOG updated

## Output format

- Release readiness verdict
- Blocking issues
- Artifact / report paths
- Merge recommendation
