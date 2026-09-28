from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)

# Research feature geometry: channel x n_mels x time.
EXPECTED_INPUT_HW: tuple[int, int, int] = (1, 64, 188)
DEFAULT_CHANNELS: tuple[int, ...] = (32, 64, 128)


class ConvBlock(nn.Module):
    """Conv2D 3x3 → BatchNorm → ReLU → MaxPool2D 2x2."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNBackbone(nn.Module):
    """Shared CNN encoder used by both CNN and CNN-BiLSTM classifiers.

    Input:  ``(B, 1, 64, 188)``
    Output: ``(B, C, F', T')`` with ``C`` equal to the last channel width (default 128).
    After three 2x2 pools: ``F'=8``, ``T'=23`` for the default geometry.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: Sequence[int] = DEFAULT_CHANNELS,
    ) -> None:
        super().__init__()
        if len(channels) < 1:
            raise ValueError("channels must contain at least one width")
        layers: list[nn.Module] = []
        prev = int(in_channels)
        for width in channels:
            layers.append(ConvBlock(prev, int(width)))
            prev = int(width)
        self.blocks = nn.Sequential(*layers)
        self.out_channels = prev
        self.in_channels = int(in_channels)
        self.channels = tuple(int(c) for c in channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert_input_shape(x, in_channels=self.in_channels)
        feats = self.blocks(x)
        if feats.ndim != 4:
            raise AssertionError(f"CNNBackbone expected 4D output, got {tuple(feats.shape)}")
        if feats.shape[1] != self.out_channels:
            raise AssertionError(
                f"CNNBackbone channel mismatch: got {feats.shape[1]}, expected {self.out_channels}"
            )
        return feats


class CNNClassifier(nn.Module):
    """CNN SER classifier: shared backbone → GAP → Dropout → Linear."""

    def __init__(
        self,
        num_classes: int = 6,
        dropout: float = 0.3,
        backbone: CNNBackbone | None = None,
        in_channels: int = 1,
        channels: Sequence[int] = DEFAULT_CHANNELS,
    ) -> None:
        super().__init__()
        self.backbone = backbone if backbone is not None else CNNBackbone(in_channels=in_channels, channels=channels)
        embed_dim = self.backbone.out_channels
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.num_classes = int(num_classes)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)                 # (B, C, F', T')
        pooled = self.pool(feats)                # (B, C, 1, 1)
        if pooled.shape[-2:] != (1, 1):
            raise AssertionError(f"GAP output must be (1,1), got {tuple(pooled.shape)}")
        embedding = torch.flatten(pooled, 1)     # (B, C)
        if embedding.shape[1] != self.embed_dim:
            raise AssertionError(
                f"CNN embedding dim {embedding.shape[1]} != expected {self.embed_dim}"
            )
        logits = self.classifier(self.dropout(embedding))
        assert_logits_shape(logits, batch_size=x.shape[0], num_classes=self.num_classes)
        return logits


class CNNBiLSTMClassifier(nn.Module):
    """CNN-BiLSTM SER classifier with a shared CNN backbone.

    Frequency-compresses the CNN feature map into a temporal sequence, then
    applies a bidirectional LSTM. Default ``hidden_size=128`` → embedding 256.
    """

    def __init__(
        self,
        num_classes: int = 6,
        dropout: float = 0.3,
        hidden_size: int = 128,
        num_layers: int = 1,
        backbone: CNNBackbone | None = None,
        in_channels: int = 1,
        channels: Sequence[int] = DEFAULT_CHANNELS,
    ) -> None:
        super().__init__()
        self.backbone = backbone if backbone is not None else CNNBackbone(in_channels=in_channels, channels=channels)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.lstm = nn.LSTM(
            input_size=self.backbone.out_channels,
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            bidirectional=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        self.embed_dim = int(hidden_size) * 2
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        self.num_classes = int(num_classes)
        self.hidden_size = int(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)                              # (B, C, F', T')
        freq_collapsed = self.freq_pool(feats).squeeze(2)     # (B, C, T')
        if freq_collapsed.ndim != 3:
            raise AssertionError(f"Expected 3D sequence map, got {tuple(freq_collapsed.shape)}")
        if freq_collapsed.shape[1] != self.backbone.out_channels:
            raise AssertionError("Frequency-pooled channel width does not match backbone")
        sequence = freq_collapsed.transpose(1, 2)             # (B, T', C)
        lstm_out, _ = self.lstm(sequence)                     # (B, T', 2H)
        if lstm_out.shape[-1] != self.embed_dim:
            raise AssertionError(
                f"BiLSTM output dim {lstm_out.shape[-1]} != expected {self.embed_dim}"
            )
        embedding = lstm_out.mean(dim=1)                      # (B, 2H)
        logits = self.classifier(self.dropout(embedding))
        assert_logits_shape(logits, batch_size=x.shape[0], num_classes=self.num_classes)
        return logits


# Backward-compatible aliases used by older imports / docs.
CNNEncoder = CNNBackbone
CNN_SER = CNNClassifier
CNN_BiLSTM_SER = CNNBiLSTMClassifier


def assert_input_shape(
    x: torch.Tensor,
    *,
    in_channels: int = EXPECTED_INPUT_HW[0],
    n_mels: int = EXPECTED_INPUT_HW[1],
    n_frames: int = EXPECTED_INPUT_HW[2],
) -> None:
    """Assert spectrogram batch has shape ``(B, C, n_mels, n_frames)``."""
    if x.ndim != 4:
        raise AssertionError(f"Expected 4D input (B,C,F,T), got shape {tuple(x.shape)}")
    _, c, f, t = x.shape
    if c != in_channels or f != n_mels or t != n_frames:
        raise AssertionError(
            f"Expected input (*, {in_channels}, {n_mels}, {n_frames}), got {tuple(x.shape)}"
        )


def assert_logits_shape(logits: torch.Tensor, *, batch_size: int, num_classes: int) -> None:
    """Assert classifier output is ``(B, num_classes)``."""
    if logits.ndim != 2 or logits.shape[0] != batch_size or logits.shape[1] != num_classes:
        raise AssertionError(
            f"Expected logits ({batch_size}, {num_classes}), got {tuple(logits.shape)}"
        )


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total / trainable / frozen parameter counts."""
    total = 0
    trainable = 0
    for param in model.parameters():
        n = int(param.numel())
        total += n
        if param.requires_grad:
            trainable += n
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def format_parameter_count(model: nn.Module) -> str:
    """Human-readable parameter summary."""
    counts = count_parameters(model)
    return (
        f"params total={counts['total']:,} "
        f"trainable={counts['trainable']:,} "
        f"frozen={counts['frozen']:,}"
    )


def _model_cfg_channels(cfg: dict) -> tuple[int, ...]:
    model_cfg = cfg.get("model", {})
    channels = model_cfg.get("channels", list(DEFAULT_CHANNELS))
    return tuple(int(c) for c in channels)


def build_model(model_id: str, num_classes: int, cfg: dict) -> nn.Module:
    """Build M0/M1 (CNN) or M2/M3 (CNN-BiLSTM) from config."""
    model_cfg = cfg.get("model", {})
    dropout = float(model_cfg.get("dropout", 0.3))
    in_channels = int(model_cfg.get("in_channels", EXPECTED_INPUT_HW[0]))
    channels = _model_cfg_channels(cfg)
    backbone = CNNBackbone(in_channels=in_channels, channels=channels)

    if model_id in {"M0", "M1"}:
        model: nn.Module = CNNClassifier(
            num_classes=num_classes,
            dropout=dropout,
            backbone=backbone,
        )
    elif model_id in {"M2", "M3"}:
        model = CNNBiLSTMClassifier(
            num_classes=num_classes,
            dropout=dropout,
            hidden_size=int(model_cfg.get("lstm_hidden", 128)),
            num_layers=int(model_cfg.get("lstm_layers", 1)),
            backbone=backbone,
        )
    else:
        raise ValueError(f"Unknown model id: {model_id}")

    LOGGER.info("%s -> %s", model_id, format_parameter_count(model))
    return model


def uses_augmentation(model_id: str) -> bool:
    """M1 and M3 train with augmentation enabled."""
    return model_id in {"M1", "M3"}


def verify_model_shapes(
    model: nn.Module,
    *,
    batch_size: int = 2,
    input_shape: tuple[int, int, int] = EXPECTED_INPUT_HW,
    num_classes: int | None = None,
    device: torch.device | None = None,
) -> dict[str, tuple[int, ...]]:
    """Run a dry-forward pass and return observed tensor shapes.

    Raises ``AssertionError`` if input/output shapes violate the research contract.
    """
    device = device or torch.device("cpu")
    c, f, t = input_shape
    x = torch.randn(batch_size, c, f, t, device=device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        if isinstance(model, (CNNClassifier, CNNBiLSTMClassifier)):
            feats = model.backbone(x)
            logits = model(x)
            shapes = {
                "input": tuple(x.shape),
                "backbone": tuple(feats.shape),
                "logits": tuple(logits.shape),
            }
        else:
            logits = model(x)
            shapes = {"input": tuple(x.shape), "logits": tuple(logits.shape)}

    expected_classes = num_classes
    if expected_classes is None and hasattr(model, "num_classes"):
        expected_classes = int(model.num_classes)
    if expected_classes is not None:
        assert_logits_shape(logits, batch_size=batch_size, num_classes=expected_classes)
    return shapes
