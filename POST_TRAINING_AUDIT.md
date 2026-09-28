# Post-Training Audit

- **Timestamp:** 2026-09-28T15:11:59Z
- **Repository:** `/Users/hazron/Downloads/ser_research_full`
- **Git HEAD:** `not-a-git-repo`
- **Python:** 3.14.7
- **Workflow:** `post_training_audit`
- **Overall:** PASS

## Gate summary

| Gate | Result | Message |
|---|---|---|
| post_training | PASS | runs=6 |

## Checks

### post_training

- **random_M0_seed42:** PASS — healthy
- **random_M1_seed42:** PASS — healthy
- **random_M2_seed42:** PASS — train/val UAR gap=0.187
- **smoke_M0:** PASS — healthy
- **speaker_independent_M0_seed42:** PASS — healthy
- **trainer_framework_check:** PASS — healthy

## Flag definitions

- **overfitting:** final train UAR exceeds val UAR by > 0.15
- **underfitting:** train and val UAR both < 0.25
- **exploding_gradients:** non-finite or extremely large loss
- **vanishing_gradients:** train loss barely decreases while high
- **learning_plateau:** recent val UAR flat and low
- **checkpoint_incorrect:** missing checkpoint or best epoch mismatch

## Recommendations

- random_M2_seed42: flags=['overfitting']
