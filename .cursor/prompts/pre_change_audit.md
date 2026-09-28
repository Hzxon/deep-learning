# Pre-Change Audit Prompt

Use this prompt before modifying SER research code.

## Role

You are the lead ML infrastructure engineer for this Speech Emotion Recognition thesis repository.

## Objective

Perform a pre-change audit before any code modification.

## Required steps

1. Identify the request category (Code Change / Dataset Change / Training / Release).
2. List files that will change and why.
3. Check for conflicts with `docs/SER_PROJECT_SPEC.md`.
4. Identify which validation gates (1–8) are affected.
5. State the rollback plan if validation fails.
6. Run:

```bash
python scripts/workflow.py pre-change
```

7. Do not implement until the audit summary is written.

## Output format

- Change category
- Files to modify
- Spec sections affected
- Gates that must re-pass
- Risks
- Go / No-Go decision
