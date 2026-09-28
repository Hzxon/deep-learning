"""Speech Emotion Recognition research package."""

from .data import LABEL_TO_ID, LABELS, SERDataset, attach_feature_paths, build_manifest
from .features import FEATURE_SHAPE, build_augmentations, preprocess_file
from .models import (
    CNNBackbone,
    CNNBiLSTMClassifier,
    CNNClassifier,
    build_model,
    count_parameters,
    verify_model_shapes,
)
from .train import Trainer, train_and_evaluate

__all__ = [
    "CNNBackbone",
    "CNNBiLSTMClassifier",
    "CNNClassifier",
    "FEATURE_SHAPE",
    "LABELS",
    "LABEL_TO_ID",
    "SERDataset",
    "Trainer",
    "attach_feature_paths",
    "build_augmentations",
    "build_manifest",
    "build_model",
    "count_parameters",
    "preprocess_file",
    "train_and_evaluate",
    "verify_model_shapes",
]
