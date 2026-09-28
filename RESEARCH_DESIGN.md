# Research-gap framing for the SER experiment

## What the literature currently gives us

Recent SER work shows three recurring situations:

1. Multi-corpus studies commonly merge RAVDESS, CREMA-D, TESS and SAVEE and use augmentation, but a high score on a merged random split does not establish cross-corpus transfer.
2. Speaker-independent evaluation is important because sample-level random splits can put the same speaker in both training and testing, making it difficult to know whether the model learned emotion cues or speaker/recording cues.
3. CNN, LSTM/BiLSTM, log-mel features and augmentation are established components. Therefore, a thesis based only on “CNN + BiLSTM is new” would be a weak novelty claim.

## More defensible research gap

The gap should be framed as a **controlled generalization analysis**, not as invention of a new backbone:

> Existing SER studies use different split protocols and often report strong results under conditions that are not directly comparable. It remains useful to quantify, under one fixed preprocessing and metric pipeline, (a) how much the apparent performance changes when speaker overlap is removed, (b) how much it degrades when an entire corpus is unseen, and (c) whether augmentation and explicit temporal modeling still provide consistent gains under these stricter protocols.

This is intentionally narrower and experimentally falsifiable.

## RQ mapping to experiments

### RQ1: speaker leakage / protocol sensitivity

Compare exactly the same M0–M3 under:

- Random split A
- Speaker-independent split B

Report UAR and Macro-F1, plus the difference between A and B. Keep the model, preprocessing, labels, optimizer and test metric unchanged.

### RQ2: cross-corpus robustness

For each held-out corpus:

- train on the other corpora;
- keep validation inside the training pool;
- never use the held-out corpus for model selection;
- test once on the held-out corpus.

This produces four leave-one-corpus-out evaluations and per-emotion recall/F1.

### RQ3: mechanism-level ablation

Use the 2x2 design:

| Model | Augmentation | BiLSTM |
|---|---:|---:|
| M0 | no | no |
| M1 | yes | no |
| M2 | no | yes |
| M3 | yes | yes |

The code calculates the four simple contrasts plus the augmentation x temporal-modeling interaction. This lets the discussion answer *why* performance changes instead of merely reporting that one model scored higher.

## What would count as evidence

Do not decide the conclusion in advance. The data should determine it.

Examples:

- If random UAR is substantially higher than speaker-independent UAR across seeds, report a protocol sensitivity effect.
- If leave-one-corpus-out UAR drops further, report a corpus-shift effect and inspect which emotions are affected.
- If augmentation improves M1 over M0 and/or M3 over M2 consistently across seeds, report an augmentation effect.
- If M2/M3 improve over M0/M1, report evidence for temporal modeling.
- If the interaction term is near zero, the two components may be mostly additive; if it changes substantially, their combination behaves differently from their isolated effects.

The implementation does not assume any of these outcomes.

## One important dataset constraint

TESS has only two speakers. A strict three-way speaker-disjoint split inside TESS alone is impossible. The implementation therefore mirrors a practical pooled speaker-independent strategy: TESS contributes one speaker to training and one to testing, with no separate TESS validation speaker. This limitation must be reported in the methods section rather than hidden.
