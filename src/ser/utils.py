from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU / CUDA / MPS) for reproducibility.

    Also makes cuDNN deterministic. Ops without a deterministic kernel only warn.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    """Device priority: MPS (Apple Silicon) → CUDA → CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dirs(*paths: str | Path) -> None:
    """Create each directory (and parents) if missing."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: str | Path) -> Path:
    """Write ``data`` as indented UTF-8 JSON, creating parent dirs. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def save_yaml(data: Any, path: str | Path) -> None:
    """Write ``data`` as YAML, keeping key order and creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def amp_device_type(device: torch.device) -> str | None:
    """Return autocast device type when mixed precision is supported."""
    if device.type == "cuda":
        return "cuda"
    if device.type == "mps":
        return "mps"
    return None
