# Reproducible Speech Emotion Recognition (SER) Research Pipeline

This repository implements an end-to-end SER experiment designed around four public English emotional-speech corpora:

- RAVDESS
- CREMA-D
- TESS
- SAVEE

The common label set is **angry, disgust, fear, happy, neutral, sad**. Samples with emotions outside this six-class intersection are discarded from the unified manifest.

## Research design

See `RESEARCH_DESIGN.md` for the literature-grounded gap framing and how each RQ maps to an experiment.

The implementation turns the workflow into three evaluation protocols and four controlled models.

### Protocols

- **A: random**: sample-level stratified train/validation/test split. This reproduces the common random-split setting but allows speaker overlap.
- **B: speaker_independent**: speaker-disjoint train/validation/test split using grouped assignment.
- **C: leave_one_corpus_out**: one corpus is completely held out for testing; the remaining corpora are used for training/validation with speaker-disjoint assignment.

### Models / ablation

- **M0**: CNN, no augmentation
- **M1**: CNN + log-mel augmentation
- **M2**: CNN-BiLSTM, no augmentation
- **M3**: CNN-BiLSTM + log-mel augmentation

This is a 2x2 ablation: augmentation x temporal modeling.

### Research questions operationalized

**RQ1.** How much does performance change when moving from a sample-random split to a speaker-independent split?

**RQ2.** How much does performance change under leave-one-corpus-out cross-corpus evaluation, and which emotion classes contribute to the degradation?

**RQ3.** What are the separate and interaction effects of augmentation and the BiLSTM temporal module under within-corpus and cross-corpus protocols?

These are testable research questions, not claims that the implementation has already established a novel result.

## Expected log-mel shape

With 16 kHz audio, a fixed 3 s window, `n_fft=512`, `win_length=512`, and `hop_length=256`, the feature tensor is exactly **1 x 64 x 188** with centered STFT framing.

## Project layout

```text
ser_research_full/
├── docs/                 # Canonical specs — single source of truth, see docs/PROJECT_STRUCTURE.md
├── .cursor/              # Research OS rules, prompts, command wrappers
├── config/               # config.yaml (production), ci.yaml (CI fixtures only)
├── src/ser/              # Library: data, features, splits, models, train, metrics, analysis, plotting, audit
├── scripts/              # One CLI per pipeline stage + workflow.py orchestrator
├── tests/                # pytest, mirrors scripts/ and src/ser/
├── data/                 # raw/ + processed/ (gitignored)
├── artifacts/            # Per-run engineering outputs + Research-OS audit reports (gitignored)
├── results/              # Research-facing tables/figures: results.csv, RQ tables (gitignored)
└── .github/workflows/    # validate.yml (PR gate), nightly-audit.yml
```

Full map with ownership per file: **[`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)**.

## Research Operating System

This repository is validation-first. Do not start training until pre-training gates pass.

```bash
python scripts/workflow.py pre-change      # before code changes
python scripts/workflow.py pre-training    # mandatory before training
python scripts/workflow.py post-training   # after training
python scripts/workflow.py ci-local        # local CI / PR readiness
```

Docs: [`docs/SER_PROJECT_SPEC.md`](docs/SER_PROJECT_SPEC.md) · [`docs/WORKFLOW_GUIDE.md`](docs/WORKFLOW_GUIDE.md) · [`docs/VALIDATION_PROTOCOL.md`](docs/VALIDATION_PROTOCOL.md) · [`RESEARCH_OS_SUMMARY.md`](RESEARCH_OS_SUMMARY.md)

## Collaborators: run from zero (macOS / Linux / Windows + CUDA)

**Canonical full playbook:** [`docs/WORKFLOW_GUIDE.md`](docs/WORKFLOW_GUIDE.md)

That guide covers clone → venv → CUDA/MPS install → dataset download layout →
validation gates → training → gap analysis → report, including Windows PowerShell
commands and NVIDIA CUDA wheel notes. Do **not** start full training until
`python scripts/workflow.py pre-training` passes.

## Install (short)

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For NVIDIA GPUs, install a **CUDA** PyTorch build from
[pytorch.org](https://pytorch.org/get-started/locally/) (see Workflow Guide Stage 0).
Device priority in code: **MPS → CUDA → CPU**.

## Dataset layout

Datasets are **not** in git. Download and extract yourself:

```text
data/raw/
├── ravdess/
├── cremad/
├── tess/
└── savee/
```

The scanner recursively searches each root for `.wav` files and parses the standard filenames/folder names used by these corpora. Sources and checks: [`docs/WORKFLOW_GUIDE.md`](docs/WORKFLOW_GUIDE.md) Stage 1.

## Run from zero (summary)

Prefer the staged checklist in [`docs/WORKFLOW_GUIDE.md`](docs/WORKFLOW_GUIDE.md). Condensed:

```bash
python scripts/prepare_manifest.py --config config/config.yaml
python scripts/preprocess.py --config config/config.yaml
python scripts/make_splits.py --config config/config.yaml --seed 42
python scripts/workflow.py pre-training
# only if PASS — smoke first, then full matrix:
python scripts/run_experiments.py --config config/config.yaml \
  --protocol speaker_independent --models M0 --seeds 42 --max-epochs 2
python scripts/run_experiments.py --config config/config.yaml \
  --protocol all --models M0 M1 M2 M3 --seeds 42 43 44
python scripts/workflow.py post-training
python scripts/gap_analysis.py --config config/config.yaml
python scripts/make_report.py --config config/config.yaml
```

Outputs: per-run folders under `artifacts/runs/`; research tables under `results/`.

## Important experimental rules

1. **Augmentation is applied only to training samples.** Validation and test data remain untouched.
2. **Feature normalization statistics are computed from the training split only.** This avoids using validation/test information for normalization.
3. **Early stopping monitors validation UAR.** The best checkpoint is the epoch with highest validation UAR.
4. **Test data is never used for model selection.**
5. **Protocol C holds out the entire corpus.** No file from the held-out corpus is used in training or validation.
6. **Results should be reported with per-class recall/F1 and confusion matrices, not only accuracy.**

## Data caveat: TESS

TESS has only two speakers. A strict three-way speaker-disjoint split *inside TESS alone* is therefore not meaningful. The implementation therefore uses pooled speaker grouping for protocol B and the held-out-corpus design for protocol C. This limitation should be stated explicitly in a paper.

## Literature motivation

A 2023 systematic review identifies speaker-independent experiments as an important challenge in SER and discusses evaluation protocols and metrics such as UAR. A 2026 study using the same four corpora explicitly emphasizes speaker-independent partitioning to avoid speaker identity leakage. Recent multi-dataset work also continues to investigate cross-corpus generalization, showing that high within-dataset results do not by themselves establish robust transfer.

Useful starting references:

- Akçay & Oğuz, *Speech emotion recognition using machine learning — A systematic review* (2023), DOI: 10.1016/j.iswa.2023.200266.
- Recent multi-corpus CNN work using RAVDESS, TESS, SAVEE and CREMA-D (Scientific Reports, 2026).
- Recent speaker-independent multi-dataset CNN/LSTM work (2026).

Replace/extend these references with the exact papers from the literature review used in your thesis so the final “research gap” is tied directly to prior methods rather than inferred from a generic survey.
