# Post-Training Audit Prompt

Use this prompt immediately after training completes.

## Role

You are the experiment quality auditor for this SER thesis repository.

## Required steps

1. Locate run directories under `artifacts/runs/` (or configured `outputs_dir`).
2. Run:

```bash
python scripts/workflow.py post-training
```

3. Inspect:
   - `POST_TRAINING_AUDIT.md`
   - `artifacts/reports/training_health.json`
   - annotated training-curve PNGs
4. Detect and report:
   - overfitting (train ≫ val)
   - underfitting (both low / flat)
   - exploding / vanishing gradients (loss NaN/Inf or near-zero updates)
   - learning plateau
   - checkpoint correctness (best epoch matches monitor metric)
5. Do not treat results as thesis-ready until the audit PASSes or issues are documented with severity.

## Output format

- Runs audited
- Health flags per run
- Best epoch vs claimed checkpoint
- Metric sanity checks
- Recommendations (retrain / tune / accept)
