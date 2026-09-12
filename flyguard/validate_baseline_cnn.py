"""Trains and validates the CNN baseline (CLAUDE.md task 4, second half).

Generates train/val/test MuJoCo datasets (independent seeds -- no trial
overlap), trains `TinyLoomNet`, and checks three things numerically:

1. Accuracy on held-out trials (not held-out frames -- the split is by
   trial, so no frame from a test trial was seen during training).
2. The static-shortcut check: does the model actually use motion, or could
   it be reading off absolute object size (looming and translation trials
   are drawn from different depth ranges, so a size-only shortcut is a real
   risk -- see `baseline_cnn.static_shortcut_accuracy`)?
3. Inference latency on CPU and GPU, for comparison against the LIF
   detector's per-step cost and Lee's tau's near-zero cost.

Usage:
    MUJOCO_GL=egl python -m flyguard.validate_baseline_cnn
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import torch  # noqa: E402

from flyguard.baseline_cnn import (  # noqa: E402
    TinyLoomNet,
    evaluate,
    frame_pairs_from_dataset,
    inference_latency_ms,
    static_shortcut_accuracy,
    train_model,
)
from flyguard.mujoco_world import generate_dataset  # noqa: E402


def ensure_datasets(train_npz: Path, val_npz: Path, test_npz: Path,
                     n_train: int, n_val: int, n_test: int, n_frames: int, resolution: int):
    if not train_npz.exists():
        generate_dataset(train_npz, n_trials_per_condition=n_train, n_frames=n_frames,
                          resolution=resolution, seed=1)
    if not val_npz.exists():
        generate_dataset(val_npz, n_trials_per_condition=n_val, n_frames=n_frames,
                          resolution=resolution, seed=2)
    if not test_npz.exists():
        generate_dataset(test_npz, n_trials_per_condition=n_test, n_frames=n_frames,
                          resolution=resolution, seed=3)


def main(train_npz: Path, val_npz: Path, test_npz: Path, epochs: int, device: str,
         n_train: int, n_val: int, n_test: int, n_frames: int, resolution: int):
    ensure_datasets(train_npz, val_npz, test_npz, n_train, n_val, n_test, n_frames, resolution)

    print(f"device: {device}")
    X_train, y_train, _ = frame_pairs_from_dataset(train_npz)
    X_val, y_val, _ = frame_pairs_from_dataset(val_npz)
    X_test, y_test, _ = frame_pairs_from_dataset(test_npz)
    print(f"train: {len(X_train)} frame-pairs | val: {len(X_val)} | test: {len(X_test)}")

    t0 = time.perf_counter()
    model = None
    for epoch, loss, model in train_model(X_train, y_train, epochs=epochs, device=device):
        val_acc = evaluate(model, X_val, y_val, device=device)
        print(f"epoch {epoch:2d}: train loss {loss:.4f} | val acc {val_acc:.3f}")
    train_time = time.perf_counter() - t0

    test_acc = evaluate(model, X_test, y_test, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\ntest accuracy (held-out trials): {test_acc:.3f}  "
          f"({n_params} parameters, {train_time:.1f}s training)")

    shortcut_acc = static_shortcut_accuracy(model, X_test, y_test, device=device)
    verdict = "OK (near chance, using real motion)" if shortcut_acc < 0.6 else \
              "WARNING: looks like a size/depth shortcut, not motion"
    print(f"static-shortcut accuracy (frame duplicated, zero real motion): "
          f"{shortcut_acc:.3f} -- {verdict}")

    lat_gpu = inference_latency_ms(model, X_test, device=device) if device == "cuda" else None
    model_cpu = TinyLoomNet()
    model_cpu.load_state_dict(model.state_dict())
    lat_cpu = inference_latency_ms(model_cpu, X_test, device="cpu")
    print(f"inference latency: {lat_cpu:.3f} ms/frame-pair (CPU)"
          + (f", {lat_gpu:.3f} ms/frame-pair (GPU)" if lat_gpu is not None else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", type=Path, default=Path("data/cnn_train.npz"))
    ap.add_argument("--val-npz", type=Path, default=Path("data/cnn_val.npz"))
    ap.add_argument("--test-npz", type=Path, default=Path("data/cnn_test.npz"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-train", type=int, default=150, help="trials per condition, train set")
    ap.add_argument("--n-val", type=int, default=30, help="trials per condition, val set")
    ap.add_argument("--n-test", type=int, default=30, help="trials per condition, test set")
    ap.add_argument("--n-frames", type=int, default=50)
    ap.add_argument("--resolution", type=int, default=64)
    a = ap.parse_args()
    main(a.train_npz, a.val_npz, a.test_npz, a.epochs, a.device,
         a.n_train, a.n_val, a.n_test, a.n_frames, a.resolution)
