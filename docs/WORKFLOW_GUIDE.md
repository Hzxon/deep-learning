# Workflow Guide

Operational playbook from a **fresh clone** to the final thesis report.
This is the **canonical collaborator onboarding doc** (macOS, Linux, and Windows + CUDA).

Canonical orchestrator (cross-platform — prefer this over shell wrappers):

```bash
python scripts/workflow.py <workflow>
```

Optional Unix wrappers (macOS / Linux / Git Bash / WSL only):

```bash
.cursor/commands/audit.sh
.cursor/commands/pretrain.sh
.cursor/commands/posttrain.sh
.cursor/commands/ci_local.sh
```

State file: [`../PROJECT_STATE.yaml`](../PROJECT_STATE.yaml)

---

## Collaborator quickstart (0 → done)

Use this checklist on a new machine. Details for each stage follow below.

| Step | What | Command / action |
| --- | --- | --- |
| 0 | Clone + Python 3.12 venv + deps | See **Stage 0** (OS-specific) |
| 1 | Download 4 corpora into `data/raw/` | See **Stage 1** |
| 2 | Manifest | `python scripts/prepare_manifest.py --config config/config.yaml` |
| 3 | Dataset gate | `python scripts/validate_dataset.py --config config/config.yaml` |
| 4 | Features | `python scripts/preprocess.py --config config/config.yaml` |
| 5 | Feature gate | `python scripts/validate_features.py --config config/config.yaml` |
| 6 | Splits | `python scripts/make_splits.py --config config/config.yaml --seed 42` |
| 7 | Split gate | `python scripts/validate_splits.py --config config/config.yaml --seed 42` |
| 8 | Smoke | `python scripts/smoke_test.py` |
| 9 | Pre-training audit (**required**) | `python scripts/workflow.py pre-training` |
| 10 | Train | See **Stage 10** (start small, then full matrix) |
| 11 | Post-training audit | `python scripts/workflow.py post-training` |
| 12 | Gap analysis (RQ1–RQ3) | `python scripts/gap_analysis.py --config config/config.yaml` |
| 13 | Report + release check | `python scripts/make_report.py --config config/config.yaml` |

**Hard rule:** do not start real training until `python scripts/workflow.py pre-training` prints `PASS`.

**Disk / time ballpark (production):**

| Resource | Approx. |
| --- | --- |
| Raw audio (`data/raw/`) | ~2 GB |
| Log-mel features (`data/processed/`) | ~0.5 GB |
| Full matrix `3 protocols × M0–M3 × 3 seeds` (72 jobs, 40 epochs) | many hours → ~1 day depending on GPU |
| Apple MPS / CUDA GPU | strongly preferred over CPU |

---

## Stage 0 — Environment Setup

**Purpose.** Create a Python **3.12** environment with the scientific stack and a working PyTorch device (`cuda` / `mps` / `cpu`).

### 0.1 Clone

```bash
git clone https://github.com/Hzxon/deep-learning.git
cd deep-learning
```

Raw datasets and experiment outputs are **not** in git (see `.gitignore`). Every machine rebuilds them locally.

### 0.2 Create and activate a venv

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Windows (CMD)**

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

**Windows alternative:** WSL2 (Ubuntu) — follow the Linux commands inside WSL. Often smoother for audio libs (`librosa` / `soundfile`).

### 0.3 Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pins a generic `torch>=2.4`. That often installs a **CPU** wheel on Windows. For GPU training, install a **CUDA** build that matches your driver.

#### Windows / Linux + NVIDIA CUDA

1. Check driver / CUDA capability: [NVIDIA drivers](https://www.nvidia.com/drivers) and `nvidia-smi`.
2. Install the matching PyTorch wheel from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) (select OS, Pip, CUDA version).

Example (CUDA 12.4 — **verify the exact index URL on pytorch.org** before copying):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

If `requirements.txt` re-resolves `torch` to CPU, install the CUDA torch wheel **after** `requirements.txt`, or install requirements excluding torch then add CUDA torch:

```bash
pip install -r requirements.txt
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
```

#### macOS (Apple Silicon)

```bash
pip install -r requirements.txt
```

PyTorch uses **MPS** automatically when available. Device priority in code: **MPS → CUDA → CPU**.

#### CPU-only (any OS)

```bash
pip install -r requirements.txt
```

Fine for smoke tests; full 72-job matrix will be very slow.

### 0.4 Verify imports and device

```bash
python -c "import torch, librosa, sklearn, yaml, pandas, numpy; print('deps ok', torch.__version__); print('cuda', torch.cuda.is_available()); print('mps', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())"
python -c "import sys; sys.path.insert(0, 'src'); from ser.utils import get_device; print('device', get_device())"
```

All `scripts/*.py` entry points add `src/` to `sys.path` automatically — run them from the **repo root** with the venv active. No editable install is required.

**Expected.** `deps ok ...` and `device` one of `cuda`, `mps`, or `cpu`.

**Checklist.**

- [ ] Python ≥ 3.12
- [ ] `torch` imports
- [ ] On Windows GPU machines: `torch.cuda.is_available()` is `True`
- [ ] On Apple Silicon: MPS available (or CPU fallback understood)

**Troubleshooting.**

| Symptom | Fix |
| --- | --- |
| `torch.cuda.is_available() == False` on NVIDIA PC | Install CUDA-enabled PyTorch from pytorch.org; update GPU driver |
| `soundfile` / `libsndfile` errors on Windows | Install [Microsoft VC++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist); or use WSL2 |
| `python` not found | Use `py -3.12` (Windows) or `python3.12` (Unix) |
| Bash wrappers fail on Windows | Use `python scripts/workflow.py ...` instead of `.cursor/commands/*.sh` |

---

## Stage 1 — Dataset Preparation

**Purpose.** Place the four public corpora under `data/raw/` so paths match `config/config.yaml`.

**Required layout**

```text
data/raw/
├── cremad/     # CREMA-D WAVs (recursive .wav OK)
├── ravdess/    # RAVDESS Actor_* trees
├── tess/       # TESS OAF_ / YAF_ (or equivalent)
└── savee/      # SAVEE AudioData / WAVs
```

Official sources (download + extract yourself; not shipped in git):

| Corpus | Typical source |
| --- | --- |
| RAVDESS | [Zenodo record 1188976](https://zenodo.org/records/1188976) (speech actors) |
| CREMA-D | [CREMA-D GitHub / project page](https://github.com/CheyneyComputerScience/CREMA-D) |
| TESS | University of Toronto TSpace / commonly mirrored datasets |
| SAVEE | [SAVEE project page](http://kahlan.eps.surrey.ac.uk/savee/) (registration may be required) |

Folder names must match config keys (`cremad`, `ravdess`, `tess`, `savee`). Internal layout can vary; scanners find `.wav` recursively and parse standard filenames.

**Commands.**

```bash
# macOS / Linux
ls data/raw/cremad data/raw/ravdess data/raw/tess data/raw/savee

# Windows PowerShell
dir data\raw\cremad, data\raw\ravdess, data\raw\tess, data\raw\savee
```

**Checklist.**

- [ ] Four corpora present with many `.wav` files
- [ ] Paths match `config/config.yaml` → `paths.cremad` / `ravdess` / `tess` / `savee`

**Troubleshooting.** Restore downloads; do not rearrange corpus-internal naming casually (parsers depend on filenames).

---

## Stage 2 — Manifest Generation

**Purpose.** Build the unified six-class manifest (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`). Discarded labels (`calm`, `surprise`, `pleasant_surprise`) are dropped automatically.

**Commands.**

```bash
python scripts/prepare_manifest.py --config config/config.yaml
python scripts/inspect_manifest.py --config config/config.yaml
```

**Expected files.** `data/processed/manifest.csv`

**Expected.** ≈ **11 318** rows, **121** speakers, six emotion classes only.

**Troubleshooting.** Fix parsers in `src/ser/data.py` if counts diverge.

---

## Stage 3 — Dataset Audit

**Purpose.** Gate 1.

```bash
python scripts/validate_dataset.py --config config/config.yaml
```

**Expected.** `DATASET VALIDATION PASSED` · `VALIDATION_REPORT.md` · `artifacts/reports/validation_dataset.json`

**Troubleshooting.** See [`VALIDATION_PROTOCOL.md`](VALIDATION_PROTOCOL.md) Gate 1.

---

## Stage 4 — Feature Generation

**Purpose.** Extract log-mel features shaped **`(64, 188)`** (input tensor `1 × 64 × 188`).

```bash
python scripts/preprocess.py --config config/config.yaml
```

**Expected files.** `data/processed/features/{dataset}/*.npy`

Use `--overwrite` only when intentionally recomputing.

---

## Stage 5 — Feature Audit

**Purpose.** Gate 2.

```bash
python scripts/validate_features.py --config config/config.yaml
```

**Expected.** `FEATURE VALIDATION PASSED`

---

## Stage 6 — Split Generation

**Purpose.** Export Protocols A / B / C.

```bash
python scripts/make_splits.py --config config/config.yaml --seed 42
```

**Expected files.** `data/splits/*_seed42/` (random, speaker_independent, four LOCO folds).

---

## Stage 7 — Split Audit

**Purpose.** Gate 3 (no speaker leakage on B; no corpus leakage on C).

```bash
python scripts/validate_splits.py --config config/config.yaml --seed 42
```

**Expected.** `SPLIT VALIDATION PASSED`

---

## Stage 8 — Smoke Test

**Purpose.** Gate 5 on synthetic / minimal path.

```bash
python scripts/smoke_test.py
```

**Expected.** `SMOKE TEST PASSED`

---

## Stage 9 — Pre-Training Audit

**Purpose.** Aggregate Gates 1–3, 5, 7. **Mandatory before real training.**

```bash
python scripts/workflow.py pre-training
```

**Expected.** `WORKFLOW pre-training: PASS` · `PRE_TRAINING_AUDIT.md`

If any gate fails: **STOP**, fix, re-run. Do not train.

---

## Stage 10 — Training

**Purpose.** Run experiments. Device is selected automatically (`mps` → `cuda` → `cpu`).

### Recommended path for a new machine

**A. Tiny smoke experiment first** (minutes):

```bash
python scripts/run_experiments.py \
  --config config/config.yaml \
  --protocol speaker_independent \
  --models M0 \
  --seeds 42 \
  --max-epochs 2
```

**B. One protocol, one seed** (sanity):

```bash
python scripts/run_experiments.py \
  --config config/config.yaml \
  --protocol speaker_independent \
  --models M0 M1 M2 M3 \
  --seeds 42
```

**C. Full research matrix** (long — often overnight+):

```bash
python scripts/run_experiments.py \
  --config config/config.yaml \
  --protocol all \
  --models M0 M1 M2 M3 \
  --seeds 42 43 44
```

Optional: append `--gap-analysis` to run gap analysis after the matrix finishes.

**Expected files.** `artifacts/runs/<run_id>/{config.yaml,history.csv,best_model.pt,metrics.json,report.txt,...}` and rows under `results/`.

**Checklist.**

- [ ] Pre-training audit already PASS
- [ ] Confirmed `Device: cuda` (Windows GPU) or `Device: mps` / `cpu` in log
- [ ] Seeds logged

**Troubleshooting.** If interrupted, re-run only missing `(protocol, model, seed[, fold])` combinations — do not delete healthy completed runs unless intentionally restarting.

**Windows tip.** Keep the laptop plugged in; avoid sleep. In PowerShell you can use:

```powershell
python scripts/run_experiments.py --config config/config.yaml --protocol all --models M0 M1 M2 M3 --seeds 42 43 44
```

---

## Stage 11 — Post-Training Audit

```bash
python scripts/workflow.py post-training
```

**Expected.** `POST_TRAINING_AUDIT.md`, `training_health.json`, annotated curves under `artifacts/plots/`.

---

## Stage 12 — Gap Analysis

```bash
python scripts/gap_analysis.py --config config/config.yaml
```

**Expected.** `results/rq1_*.csv`, `results/rq2_*.csv`, `results/rq3_*.csv`, `results/gap_analysis.md` (mean ± std across seeds).

---

## Stage 13 — Report Generation

```bash
python scripts/make_report.py --config config/config.yaml
python scripts/workflow.py release
```

**Expected.** Human-readable report artifacts; local CI / release gates PASS; `PROJECT_STATE.yaml` updated.

---

## End-to-end quick reference

```bash
# Developer loop (before code changes)
python scripts/workflow.py pre-change

# Before training
python scripts/workflow.py pre-training

# After training
python scripts/workflow.py post-training

# PR / merge readiness
python scripts/workflow.py ci-local
python scripts/workflow.py release
```

### One-shot sequence (after datasets are in `data/raw/`)

```bash
python scripts/prepare_manifest.py --config config/config.yaml
python scripts/validate_dataset.py --config config/config.yaml
python scripts/preprocess.py --config config/config.yaml
python scripts/validate_features.py --config config/config.yaml
python scripts/make_splits.py --config config/config.yaml --seed 42
python scripts/validate_splits.py --config config/config.yaml --seed 42
python scripts/smoke_test.py
python scripts/workflow.py pre-training
# only if PASS:
python scripts/run_experiments.py --config config/config.yaml --protocol all --models M0 M1 M2 M3 --seeds 42 43 44
python scripts/workflow.py post-training
python scripts/gap_analysis.py --config config/config.yaml
python scripts/make_report.py --config config/config.yaml
```

---

## Related docs

| Doc | Role |
| --- | --- |
| [`SER_PROJECT_SPEC.md`](SER_PROJECT_SPEC.md) | Research / model / metric contract |
| [`VALIDATION_PROTOCOL.md`](VALIDATION_PROTOCOL.md) | Gate recovery |
| [`CI_SPEC.md`](CI_SPEC.md) | CI architecture |
| [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) | Where every file lives |
| [`../README.md`](../README.md) | Human entry point |
