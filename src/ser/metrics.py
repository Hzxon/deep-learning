from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)


def classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    labels: list[int],
    target_names: list[str] | None = None,
) -> dict:
    """Compute Accuracy, UAR, macro precision/recall/F1, confusion matrix, and report."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    accuracy = float(accuracy_score(y_true, y_pred))
    uar = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_precision = float(np.mean(precision)) if len(precision) else 0.0
    macro_recall = float(np.mean(recall)) if len(recall) else 0.0
    names = target_names if target_names is not None else [str(x) for x in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        zero_division=0,
        output_dict=True,
    )
    return {
        "accuracy": accuracy,
        "uar": uar,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": {
            str(label): {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for label, p, r, f, s in zip(labels, precision, recall, f1, support, strict=True)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": report,
    }
