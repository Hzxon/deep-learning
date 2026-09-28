#!/usr/bin/env python3
"""Gate 7 — Reproducibility check (deterministic forward pass)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser.audit import CheckResult, GateResult, exit_code, render_gate_report, save_json, save_markdown  # noqa: E402
from ser.models import build_model  # noqa: E402
from ser.utils import get_device, seed_everything  # noqa: E402


def _forward_once(cfg: dict, seed: int, device: torch.device) -> np.ndarray:
    seed_everything(seed)
    model = build_model("M0", num_classes=len(cfg["labels"]), cfg=cfg).to(device)
    model.eval()
    x = torch.randn(2, 1, 64, 188, device=device)
    with torch.no_grad():
        logits = model(x)
    return logits.detach().cpu().numpy()


def reproducibility_check(cfg: dict, *, seed: int) -> GateResult:
    device = get_device()
    # Prefer CPU for bit-stable comparison across MPS/CUDA nondeterminism edge cases.
    compare_device = torch.device("cpu")
    a = _forward_once(cfg, seed, compare_device)
    b = _forward_once(cfg, seed, compare_device)
    max_abs = float(np.max(np.abs(a - b)))
    identical = bool(np.allclose(a, b, atol=0.0, rtol=0.0))

    checks = [
        CheckResult("seed_logged", True, f"seed={seed} device={device} compare_device={compare_device}"),
        CheckResult("deterministic_forward", identical, f"max_abs_diff={max_abs}"),
        CheckResult("finite_logits", bool(np.isfinite(a).all()), f"shape={list(a.shape)}"),
    ]
    passed = all(c.passed for c in checks)
    return GateResult(
        gate="reproducibility",
        passed=passed,
        checks=checks,
        message=f"identical={identical} max_abs_diff={max_abs}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check deterministic model behavior under a fixed seed.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default="artifacts/reports/REPRODUCIBILITY.md")
    parser.add_argument("--json", default="artifacts/reports/reproducibility.json")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    gate = reproducibility_check(cfg, seed=int(args.seed))
    save_markdown(
        render_gate_report(
            title="Validation Report — Reproducibility",
            workflow="reproducibility_check",
            gates=[gate],
        ),
        ROOT / args.report,
    )
    save_json({"gate": gate.to_dict()}, ROOT / args.json)

    for check in gate.checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    print("REPRODUCIBILITY " + ("PASSED" if gate.passed else "FAILED"))
    raise SystemExit(exit_code(gate.passed))


if __name__ == "__main__":
    main()
