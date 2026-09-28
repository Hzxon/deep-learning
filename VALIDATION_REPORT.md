# Validation Report — Dataset Integrity

- **Timestamp:** 2026-09-28T15:59:55Z
- **Repository:** `/Users/hazron/Downloads/ser_research_full`
- **Git HEAD:** `d3b7bc0`
- **Python:** 3.14.7
- **Workflow:** `validate_dataset`
- **Overall:** PASS

## Gate summary

| Gate | Result | Message |
|---|---|---|
| dataset_integrity | PASS | n=11318 errors=0 missing_audio=0 |

## Checks

### dataset_integrity

- **metadata_columns:** PASS — ok
- **non_empty:** PASS — n=11318
- **no_duplicate_audio:** PASS — duplicates=0
- **no_discarded_emotions:** PASS — discarded=[]
- **emotion_labels:** PASS — found=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad'] expected=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
- **dataset_count:** PASS — found=['CREMA-D', 'RAVDESS', 'SAVEE', 'TESS']
- **speaker_counts:** PASS — CREMA-D=91/91, RAVDESS=24/24, TESS=2/2, SAVEE=4/4
- **duration_positive:** PASS — bad=0 mean=2.599s
- **audio_present:** PASS — missing=0 of sampled=200
- **audio_readable:** PASS — checked=200 errors=[] target_sr=16000

## Recommendations

- All gates passed. Safe to proceed.
