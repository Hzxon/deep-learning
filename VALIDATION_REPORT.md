# Validation Report — Dataset Integrity

- **Timestamp:** 2026-09-28T15:31:31Z
- **Repository:** `/Users/hazron/Downloads/ser_research_full`
- **Git HEAD:** `not-a-git-repo`
- **Python:** 3.14.7
- **Workflow:** `validate_dataset`
- **Overall:** PASS

## Gate summary

| Gate | Result | Message |
|---|---|---|
| dataset_integrity | PASS | n=216 errors=0 missing_audio=200 |

## Checks

### dataset_integrity

- **metadata_columns:** PASS — ok
- **non_empty:** PASS — n=216
- **no_duplicate_audio:** PASS — duplicates=0
- **no_discarded_emotions:** PASS — discarded=[]
- **emotion_labels:** PASS — found=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad'] expected=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
- **dataset_count:** PASS — found=['CREMA-D', 'RAVDESS', 'SAVEE', 'TESS']
- **speaker_counts:** PASS — CREMA-D=6, RAVDESS=6, TESS=2, SAVEE=4
- **duration_positive:** PASS — bad=0 mean=3.000s
- **audio_readable:** PASS — checked=0 missing=200 errors=0

## Recommendations

- All gates passed. Safe to proceed.
