AGENTS.md

# ROLE

You are the lead ML engineer for an academic Speech Emotion Recognition (SER) research project. This is NOT a tutorial project, notebook, or Kaggle competition. This repository must be production-quality, reproducible, modular, and suitable for a university thesis and paper.

Always modify the existing project instead of creating parallel files unless explicitly instructed.

---

# PROJECT GOAL

Build an end-to-end Speech Emotion Recognition research framework using PyTorch.

Pipeline:

Raw datasets
→ metadata manifest
→ preprocessing
→ feature extraction (log-mel spectrogram)
→ dataset splits
→ augmentation
→ model training
→ evaluation
→ ablation study
→ gap analysis
→ report generation.

The repository must allow running every experiment through command-line scripts.

---

# RESEARCH OBJECTIVE

This research investigates generalization in Speech Emotion Recognition across multiple datasets.

Research Questions:

RQ1.
How much performance changes between random split and speaker-independent split.

RQ2.
How much performance drops under Leave-One-Corpus-Out evaluation.

RQ3.
What is the contribution of data augmentation and temporal modeling (BiLSTM) individually and jointly.

The framework must support experiments answering all RQs.

---

# DATASETS

Use four public datasets.

1. CREMA-D
2. RAVDESS
3. TESS
4. SAVEE

The datasets are already downloaded locally.

Folder structure:

data/raw/
cremad/
ravdess/
tess/
savee/

Manifest already generated.

Current manifest statistics:

CREMA-D

- speakers: 91

RAVDESS

- speakers: 24

TESS

- speakers: 2

SAVEE

- speakers: 4

Total samples ≈ 12k.

---

# FINAL EMOTION CLASSES

Normalize every dataset into exactly six classes.

angry
disgust
fear
happy
neutral
sad

Discard:

- calm
- surprise
- pleasant_surprise

The preprocessing code must ignore discarded emotions.

Class mapping must be centralized in one file.

---

# PREPROCESSING SPECIFICATION

Every audio must be converted into identical input.

Audio preprocessing:

- mono
- sample rate = 16000 Hz
- duration = exactly 3 seconds
- trim silence
- pad with zeros if shorter
- normalize waveform

Feature extraction:

Log-Mel Spectrogram

Parameters:

sample_rate = 16000
n_fft = 512
hop_length = 256
win_length = 512
n_mels = 64
f_min = 20
f_max = 8000

Output shape:

64 x 188

Save each feature as float32 NumPy array.

Directory:

data/processed/features/{dataset}/...

Do NOT recompute existing features unless overwrite flag is enabled.

---

# METADATA

manifest.csv already exists.

Columns must include:

filepath
dataset
speaker
emotion
label
duration
split
feature_path

Never duplicate metadata logic across files.

Create reusable metadata utilities.

---

# DATASET SPLITS

Support three protocols.

Protocol A

Random Stratified Split

70 / 10 / 20

Stratify by emotion.

Protocol B

Speaker Independent

No speaker overlap.

70 / 10 / 20 speakers.

Special handling:

TESS has only two speakers.

One speaker train.

One speaker test.

Validation merged into training for TESS only.

Protocol C

Leave-One-Corpus-Out

Train:
three datasets.

Test:
remaining dataset.

Need four folds.

Fold examples:

test = CREMA-D
train = RAVDESS + TESS + SAVEE

Repeat for every corpus.

Store split CSVs.

data/splits/

Every split must be reproducible using seed.

---

# DATA AUGMENTATION

Augment only training samples.

Validation/Test must never be augmented.

Implement modular transforms.

Transforms:

Time Shift

Random Gain

Gaussian Noise

Pitch Shift

Time Stretch

SpecAugment

Each transform has configurable probability.

Compose transforms from config.

Must be deterministic under seed.

---

# MODEL VARIANTS

Implement four experiment variants.

M0

CNN

No augmentation.

M1

CNN

Augmentation enabled.

M2

CNN + BiLSTM

No augmentation.

M3

CNN + BiLSTM

Augmentation enabled.

---

# CNN ARCHITECTURE

Input:

1 x 64 x 188

Three convolution blocks.

Block:

Conv2D 3x3
BatchNorm
ReLU
MaxPool2D 2x2

Channels:

32
64
128

Global Average Pooling.

128-dimensional embedding.

Dropout = 0.3

Linear classifier.

Output:

6 logits.

Architecture should be configurable.

---

# CNN-BiLSTM ARCHITECTURE

CNN backbone identical.

Before pooling:

Interpret frequency-compressed CNN feature map as temporal sequence.

Feed sequence into:

BiLSTM

hidden_size = 128

bidirectional = True

Output embedding:

256

Dropout.

Fully Connected.

6 logits.

CNN backbone should be reusable.

No duplicated CNN implementation.

---

# TRAINING SPECIFICATION

Framework:

PyTorch.

Device priority:

MPS
CUDA
CPU

Mixed precision when supported.

Training features:

Early stopping.

Checkpoint best model.

Gradient clipping.

Learning rate scheduler.

Configurable optimizer.

Supported optimizers:

Adam

AdamW

Loss:

CrossEntropy.

Optional class weights computed automatically.

---

# METRICS

Compute:

Accuracy

UAR

Macro Precision

Macro Recall

Macro F1

Confusion Matrix

Classification Report

ROC is NOT required.

Metrics computed after every epoch.

Early stopping monitored using Validation UAR.

---

# LOGGING

Every experiment creates a unique folder under `artifacts/runs/`
(`paths.outputs_dir` in `config/config.yaml`).

Example:

artifacts/runs/
speaker_independent_M3_seed42/
config.yaml
history.csv
best_model.pt
metrics.json
confusion_matrix.png
report.txt

History CSV contains:

epoch
train_loss
val_loss
train_accuracy
val_accuracy
val_uar
val_macro_f1
learning_rate

---

# ABLATION SYSTEM

Need automatic experiment runner.

scripts/run_experiments.py

CLI examples:

python scripts/run_experiments.py --protocol speaker_independent --models M0 M1 M2 M3 --seeds 42 43 44

Runner executes every combination.

Collect results into one CSV.

results/results.csv

Columns:

protocol
model
seed
dataset_fold
accuracy
uar
macro_f1
precision
recall

---

# GAP ANALYSIS

scripts/gap_analysis.py

Automatically generate tables for RQ1 RQ2 RQ3.

RQ1

Compare:

Random vs Speaker Independent.

Metrics:

Accuracy
UAR
Macro-F1

Mean ± Std across seeds.

RQ2

Leave-One-Corpus-Out performance.

Table:

Train Corpora

Test Corpus

Accuracy

UAR

Macro-F1

RQ3

Ablation table.

Compare:

M0
M1
M2
M3

Include delta improvements.

Generate CSV and Markdown table.

---

# VISUALIZATION

Automatically generate plots.

Training curves.

Loss.

Accuracy.

UAR.

Macro-F1.

Confusion Matrix.

Bar chart comparing protocols.

Bar chart comparing models.

Save PNG only.

No notebook plotting.

---

# CONFIGURATION

All hyperparameters live in config/config.yaml.

No magic numbers inside Python code.

Use dataclass or configuration loader.

Support overriding through CLI.

Example:

python scripts/run_experiments.py --config config/config.yaml --batch_size 64 --epochs 100

---

# REPRODUCIBILITY

Create utility that sets seeds for:

Python

NumPy

PyTorch

Torch CUDA

Torch MPS

Every experiment logs seed.

Use deterministic behavior whenever possible.

---

# PROJECT STRUCTURE

Keep this structure (canonical map: `docs/PROJECT_STRUCTURE.md`).

ser_research_full/

docs/

.cursor/

config/

scripts/

src/ser/

data/

artifacts/

results/

Do not create notebooks.

Do not create duplicate implementations.

---

# CODING STYLE

Requirements:

- Python 3.12.
- Full type hints.
- Docstrings for public classes/functions.
- Modular classes.
- No hardcoded dataset paths.
- Use pathlib everywhere.
- Avoid global variables.
- Separate model/data/training/evaluation responsibilities.

Whenever modifying code:

1. Explain what files will change.
2. Implement complete code.
3. Keep imports clean.
4. Ensure scripts remain runnable from repository root.

Never provide placeholder implementations.
Never use TODO unless explicitly requested.
All code must be immediately executable.
