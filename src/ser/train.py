from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .data import SERDataset
from .metrics import classification_metrics
from .models import build_model, uses_augmentation
from .utils import amp_device_type, ensure_dirs, save_json, save_yaml

LOGGER = logging.getLogger(__name__)

# Backward-compatible alias for older imports.
FeatureDataset = SERDataset


def feature_stats(df: pd.DataFrame) -> tuple[float, float]:
    """Compute global mean/std over training log-mel features."""
    total = 0
    sum_x = 0.0
    sum_x2 = 0.0
    for path in tqdm(df["feature_path"].tolist(), desc="Computing train normalization", leave=False):
        x = np.load(path).astype(np.float64)
        total += x.size
        sum_x += float(x.sum())
        sum_x2 += float(np.square(x).sum())
    mean = sum_x / max(total, 1)
    var = sum_x2 / max(total, 1) - mean * mean
    return mean, float(np.sqrt(max(var, 1e-12)))


def class_weights(df: pd.DataFrame, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = df["label"].value_counts().reindex(range(num_classes), fill_value=0).to_numpy(dtype=np.float64)
    total = counts.sum()
    weights = total / np.maximum(counts * num_classes, 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def build_optimizer(model: nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    """Create Adam or AdamW from config."""
    name = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(train_cfg["learning_rate"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer '{name}'. Expected adam or adamw.")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: dict,
    max_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    """Create LR scheduler from config (reduce_on_plateau | cosine | step | none)."""
    raw = train_cfg.get("scheduler", {"name": "reduce_on_plateau"})
    if raw is None:
        return None
    if isinstance(raw, str):
        sched_cfg: dict[str, Any] = {"name": raw}
    else:
        sched_cfg = dict(raw)

    name = str(sched_cfg.get("name", "reduce_on_plateau")).lower()
    if name in {"none", "null", "off", ""}:
        return None
    if name in {"reduce_on_plateau", "plateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(sched_cfg.get("mode", "max")),
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 2)),
            min_lr=float(sched_cfg.get("min_lr", 1e-6)),
        )
    if name in {"cosine", "cosineannealing"}:
        t_max = int(sched_cfg["t_max"]) if sched_cfg.get("t_max") is not None else max_epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(t_max, 1),
            eta_min=float(sched_cfg.get("min_lr", 0.0)),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sched_cfg.get("step_size", 10)),
            gamma=float(sched_cfg.get("gamma", 0.1)),
        )
    raise ValueError(f"Unsupported scheduler '{name}'.")


def _step_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    monitored_value: float,
) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(monitored_value)
    else:
        scheduler.step()


@dataclass
class EarlyStopping:
    """Stop training when the monitored metric stops improving."""

    patience: int
    mode: str = "max"
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"max", "min"}:
            raise ValueError("EarlyStopping mode must be 'max' or 'min'")
        self.best: float = float("-inf") if self.mode == "max" else float("inf")
        self.stale: int = 0
        self.should_stop: bool = False
        self.improved: bool = False

    def step(self, value: float) -> bool:
        """Update state with a new metric value. Returns True if this step improved."""
        if self.mode == "max":
            better = value > self.best + self.min_delta
        else:
            better = value < self.best - self.min_delta
        if better:
            self.best = float(value)
            self.stale = 0
            self.improved = True
        else:
            self.stale += 1
            self.improved = False
            if self.stale >= self.patience:
                self.should_stop = True
        return self.improved


class Trainer:
    """Config-driven SER trainer with AMP, TensorBoard, early stopping, and checkpoints."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        *,
        scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
        grad_clip_norm: float = 5.0,
        patience: int = 8,
        monitor_metric: str = "val_uar",
        mixed_precision: bool = True,
        run_dir: Path,
        checkpoint_path: Path,
        tensorboard: bool = True,
        tensorboard_dir: Path | None = None,
        label_names: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.grad_clip_norm = float(grad_clip_norm)
        self.monitor_metric = monitor_metric
        self.label_names = label_names or []
        self.metadata = metadata or {}
        self.run_dir = Path(run_dir)
        self.checkpoint_path = Path(checkpoint_path)
        ensure_dirs(self.run_dir, self.checkpoint_path.parent)

        self.early_stopping = EarlyStopping(patience=int(patience), mode="max")
        self.history: list[dict[str, float | int]] = []
        self.best_epoch = 0
        self.best_val_uar = -1.0

        amp_type = amp_device_type(device) if mixed_precision else None
        self.amp_type = amp_type
        self.use_amp = amp_type is not None
        # GradScaler is reliable on CUDA; skip on MPS/CPU.
        self.scaler = torch.amp.GradScaler("cuda") if amp_type == "cuda" else None

        self.writer: SummaryWriter | None = None
        if tensorboard:
            tb_dir = Path(tensorboard_dir) if tensorboard_dir is not None else self.run_dir / "tb"
            ensure_dirs(tb_dir)
            self.writer = SummaryWriter(log_dir=str(tb_dir))
            LOGGER.info("TensorBoard logging -> %s", tb_dir)

    def _autocast(self):
        if self.use_amp and self.amp_type is not None:
            return torch.amp.autocast(device_type=self.amp_type, dtype=torch.float16)
        # No-op context manager.
        from contextlib import nullcontext

        return nullcontext()

    def _run_epoch(
        self,
        loader: DataLoader,
        *,
        train: bool,
    ) -> tuple[float, list[int], list[int]]:
        self.model.train(train)
        running_loss = 0.0
        y_true: list[int] = []
        y_pred: list[int] = []

        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train), self._autocast():
                logits = self.model(x)
                loss = self.criterion(logits, y)

            if train:
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()

            running_loss += float(loss.detach().item()) * len(y)
            pred = torch.argmax(logits.detach(), dim=1)
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(pred.detach().cpu().tolist())

        return running_loss / max(len(loader.dataset), 1), y_true, y_pred

    def _metrics(self, y_true: list[int], y_pred: list[int]) -> dict:
        labels = list(range(len(self.label_names))) if self.label_names else sorted(set(y_true) | set(y_pred))
        return classification_metrics(y_true, y_pred, labels=labels, target_names=self.label_names or None)

    def _save_checkpoint(self, extra: dict[str, Any] | None = None) -> None:
        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_epoch": self.best_epoch,
            "best_val_uar": self.best_val_uar,
            "history": self.history,
            **self.metadata,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, self.checkpoint_path)
        # Also keep a copy inside the run folder for AGENTS-style layout.
        torch.save(payload, self.run_dir / "best_model.pt")

    def _log_tensorboard(self, row: dict[str, float | int], train_m: dict, val_m: dict) -> None:
        if self.writer is None:
            return
        epoch = int(row["epoch"])
        self.writer.add_scalar("loss/train", float(row["train_loss"]), epoch)
        self.writer.add_scalar("loss/val", float(row["val_loss"]), epoch)
        self.writer.add_scalar("accuracy/train", float(row["train_accuracy"]), epoch)
        self.writer.add_scalar("accuracy/val", float(row["val_accuracy"]), epoch)
        self.writer.add_scalar("uar/train", float(train_m["uar"]), epoch)
        self.writer.add_scalar("uar/val", float(row["val_uar"]), epoch)
        self.writer.add_scalar("macro_f1/train", float(train_m["macro_f1"]), epoch)
        self.writer.add_scalar("macro_f1/val", float(row["val_macro_f1"]), epoch)
        self.writer.add_scalar("lr", float(row["learning_rate"]), epoch)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int,
    ) -> list[dict[str, float | int]]:
        """Train with early stopping on validation UAR."""
        for epoch in range(1, max_epochs + 1):
            train_loss, train_true, train_pred = self._run_epoch(train_loader, train=True)
            val_loss, val_true, val_pred = self._run_epoch(val_loader, train=False)
            train_m = self._metrics(train_true, train_pred)
            val_m = self._metrics(val_true, val_pred)

            monitored = float(val_m["uar"])
            _step_scheduler(self.scheduler, monitored)

            row: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_accuracy": float(train_m["accuracy"]),
                "val_accuracy": float(val_m["accuracy"]),
                "train_uar": float(train_m["uar"]),
                "val_uar": float(val_m["uar"]),
                "train_macro_f1": float(train_m["macro_f1"]),
                "val_macro_f1": float(val_m["macro_f1"]),
                "val_macro_precision": float(val_m["macro_precision"]),
                "val_macro_recall": float(val_m["macro_recall"]),
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            }
            self.history.append(row)
            self._log_tensorboard(row, train_m, val_m)

            print(
                f"[{self.metadata.get('run_id', 'run')}] epoch={epoch:02d} "
                f"train_loss={train_loss:.4f} train_UAR={train_m['uar']:.4f} "
                f"val_loss={val_loss:.4f} val_UAR={val_m['uar']:.4f} "
                f"val_F1={val_m['macro_f1']:.4f} lr={row['learning_rate']:.2e}"
            )

            if self.early_stopping.step(monitored):
                self.best_epoch = epoch
                self.best_val_uar = monitored
                self._save_checkpoint({"val_metrics": val_m})

            if self.early_stopping.should_stop:
                LOGGER.info(
                    "Early stopping at epoch %d (best val_uar=%.4f @ epoch %d)",
                    epoch,
                    self.best_val_uar,
                    self.best_epoch,
                )
                break

        if self.best_epoch == 0:
            # No improvement recorded (should be rare); still save last weights.
            self.best_epoch = int(self.history[-1]["epoch"]) if self.history else 0
            self.best_val_uar = float(self.history[-1]["val_uar"]) if self.history else -1.0
            self._save_checkpoint()

        self._write_history()
        return self.history

    def _write_history(self) -> None:
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.run_dir / "history.csv", index=False)
        save_json(self.history, self.run_dir / "history.json")

    def load_best(self) -> dict[str, Any]:
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        return checkpoint

    def evaluate(self, loader: DataLoader) -> tuple[float, dict]:
        """Evaluate the current model weights."""
        loss, y_true, y_pred = self._run_epoch(loader, train=False)
        metrics = self._metrics(y_true, y_pred)
        return loss, metrics

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None


def train_and_evaluate(
    cfg: dict,
    model_id: str,
    split_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    labels: list[str],
    seed: int,
    run_id: str,
    device: torch.device,
    max_epochs_override: int | None = None,
) -> dict:
    """High-level training entry used by experiment runners (CLI-compatible)."""
    num_classes = len(labels)
    train_cfg = cfg["train"]
    mean, std = feature_stats(train_df)

    train_ds = SERDataset(
        train_df,
        augment=uses_augmentation(model_id),
        aug_cfg=cfg["augmentation"],
        mean=mean,
        std=std,
        seed=seed,
    )
    val_ds = SERDataset(
        val_df,
        augment=False,
        aug_cfg=cfg["augmentation"],
        mean=mean,
        std=std,
        seed=seed + 1,
    )
    test_ds = SERDataset(
        test_df,
        augment=False,
        aug_cfg=cfg["augmentation"],
        mean=mean,
        std=std,
        seed=seed + 2,
    )

    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(model_id, num_classes=num_classes, cfg=cfg)
    if bool(train_cfg.get("use_class_weights", True)):
        weights = class_weights(train_df, num_classes).to(device)
        criterion: nn.Module = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    max_epochs = int(max_epochs_override or train_cfg["max_epochs"])
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg, max_epochs=max_epochs)

    outputs_dir = Path(cfg["paths"].get("outputs_dir", cfg["paths"]["results_dir"]))
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    tb_root = Path(cfg["paths"].get("tensorboard_dir", outputs_dir / "tensorboard"))
    run_dir = outputs_dir / run_id
    ensure_dirs(outputs_dir, ckpt_dir, results_dir, tb_root, run_dir)

    save_yaml(cfg, run_dir / "config.yaml")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        grad_clip_norm=float(train_cfg.get("grad_clip_norm", 5.0)),
        patience=int(train_cfg.get("patience", 8)),
        monitor_metric=str(train_cfg.get("monitor_metric", "val_uar")),
        mixed_precision=bool(train_cfg.get("mixed_precision", True)),
        run_dir=run_dir,
        checkpoint_path=ckpt_dir / f"{run_id}.pt",
        tensorboard=bool(train_cfg.get("tensorboard", True)),
        tensorboard_dir=tb_root / run_id,
        label_names=labels,
        metadata={
            "run_id": run_id,
            "model_id": model_id,
            "split": split_name,
            "seed": seed,
            "mean": mean,
            "std": std,
            "labels": labels,
            "cfg": cfg,
        },
    )

    start_time = time.time()
    try:
        history = trainer.fit(train_loader, val_loader, max_epochs=max_epochs)
        trainer.load_best()
        test_loss, test_metrics = trainer.evaluate(test_loader)
    finally:
        trainer.close()
    elapsed = time.time() - start_time

    artifact = {
        "run_id": run_id,
        "model_id": model_id,
        "split": split_name,
        "seed": seed,
        "device": str(device),
        "best_epoch": trainer.best_epoch,
        "best_val_uar": trainer.best_val_uar,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "train_speakers": int(train_df["speaker"].nunique()),
        "val_speakers": int(val_df["speaker"].nunique()),
        "test_speakers": int(test_df["speaker"].nunique()),
        "train_datasets": sorted(train_df["dataset"].unique().tolist()),
        "test_datasets": sorted(test_df["dataset"].unique().tolist()),
        "normalization": {"mean": mean, "std": std},
        "elapsed_seconds": elapsed,
        "optimizer": str(train_cfg.get("optimizer", "adamw")),
        "scheduler": train_cfg.get("scheduler"),
        "mixed_precision": bool(train_cfg.get("mixed_precision", True)) and amp_device_type(device) is not None,
        "history_epochs": len(history),
    }
    save_json(artifact, results_dir / f"{run_id}.json")
    save_json(history, results_dir / f"{run_id}_history.json")
    save_json(artifact, run_dir / "metrics.json")
    save_json(history, run_dir / "history.json")

    # Human-readable report.
    report_lines = [
        f"run_id: {run_id}",
        f"model: {model_id}",
        f"split: {split_name}",
        f"seed: {seed}",
        f"device: {device}",
        f"best_epoch: {trainer.best_epoch}",
        f"best_val_uar: {trainer.best_val_uar:.4f}",
        f"test_accuracy: {test_metrics['accuracy']:.4f}",
        f"test_uar: {test_metrics['uar']:.4f}",
        f"test_macro_precision: {test_metrics['macro_precision']:.4f}",
        f"test_macro_recall: {test_metrics['macro_recall']:.4f}",
        f"test_macro_f1: {test_metrics['macro_f1']:.4f}",
        f"elapsed_seconds: {elapsed:.1f}",
    ]
    (run_dir / "report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return artifact
