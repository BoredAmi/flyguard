"""Offline benchmark (CLAUDE.md layer 3): replay a shared set of trials
through all three detectors and report one accuracy/latency comparison.
All three detection pipelines already exist and are individually validated
(`validate_encoder.py`, `validate_baseline_tau.py`, `validate_baseline_cnn.py`)
-- this script is assembly, not new detection logic.

**Important asymmetry, stated plainly rather than hidden:** Lee's tau and
the CNN run on the same real rendered MuJoCo pixels generated for this
benchmark. The connectome/LIF detector does **not** yet -- the retinotopic
encoder still only consumes synthetic flow fields matching a trial's known
*condition* (`expanding_flow_2d` / `translational_flow_2d`), not flow
estimated from the rendered frames themselves (see CLAUDE.md "Remaining
open item" / task 4 note). Building a real image-pixel -> T4/T5-column flow
estimator is future work, not something to improvise here under time
pressure -- so the connectome row of this benchmark reuses the already
-validated per-heading firing rates in `data/hemisphere_sweep.json`
(`validate_encoder.py --both-sides --heading-step 15`) rather than
re-running the LIF simulation per rendered trial. Its accuracy number is
real and reproducible, but its *input* is not the same rendered pixels the
other two detectors see -- keep that in the README methods section, not
just here.

Usage:
    MUJOCO_GL=egl python -m flyguard.offline_benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch  # noqa: E402

from flyguard.baseline_cnn import (  # noqa: E402
    frame_pairs_from_dataset,
    inference_latency_ms,
    train_model,
)
from flyguard.baseline_tau import apparent_radius_from_area, lee_tau  # noqa: E402
from flyguard.mujoco_world import apparent_area_px, generate_dataset  # noqa: E402

TAU_WARNING_S = 3.0  # matches validate_baseline_tau.py's convention


def tau_predictions(npz_path: Path, resolution_note: str, smooth_window: int = 5):
    z = np.load(npz_path)
    frames, condition, dt = z["frames"], z["condition"], float(z["params"][0, 4])
    preds, latencies = [], []
    for i in range(frames.shape[0]):
        t0 = time.perf_counter()
        area = apparent_area_px(frames[i])
        radius = apparent_radius_from_area(area)
        tau = lee_tau(radius, dt, smooth_window=smooth_window)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed / frames.shape[1] * 1000)  # ms/frame
        preds.append("looming" if np.any(tau < TAU_WARNING_S) else "translation")
    acc = float(np.mean([p == c for p, c in zip(preds, condition)]))
    return acc, float(np.mean(latencies))


def cnn_predictions(train_npz: Path, npz_path: Path, epochs: int, device: str):
    X_train, y_train, _ = frame_pairs_from_dataset(train_npz)
    model = None
    for _, _, model in train_model(X_train, y_train, epochs=epochs, device=device, seed=0):
        pass

    X, y, tid = frame_pairs_from_dataset(npz_path)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device))
        pred_pairs = logits.argmax(dim=1).cpu().numpy()

    z = np.load(npz_path)
    condition = z["condition"]
    trial_ids = z["trial_id"]
    correct = 0
    for i, cond in enumerate(condition):
        mask = tid == trial_ids[i]
        majority = int(np.round(pred_pairs[mask].mean())) if mask.any() else 0
        pred_label = "looming" if majority == 1 else "translation"
        correct += (pred_label == cond)
    acc = correct / len(condition)
    lat = inference_latency_ms(model, X[:1], device=device)
    return acc, lat


def connectome_accuracy(sweep_json: Path):
    """Reuses `data/hemisphere_sweep.json` rather than re-running the LIF
    simulation here (see module docstring for why). Decision rule: classify
    "looming" if the LPLC2 rate exceeds the midpoint between that
    hemisphere's own looming rate and its mean translation rate -- the
    simplest threshold that doesn't hand-tune to any specific heading.
    """
    with open(sweep_json) as f:
        data = json.load(f)
    results = {}
    for side, d in data.items():
        looming_rate = d["looming"]
        trans_rates = list(d["translation"].values())
        threshold = (looming_rate + np.mean(trans_rates)) / 2
        # 1 true "looming" condition + len(trans_rates) true "translation" conditions
        correct = int(looming_rate > threshold)  # the looming trial itself
        correct += sum(r <= threshold for r in trans_rates)
        total = 1 + len(trans_rates)
        results[side] = {
            "accuracy": correct / total,
            "threshold_hz": threshold,
            "n_conditions": total,
        }
    return results


def main(train_npz: Path, bench_npz: Path, sweep_json: Path,
         n_bench_trials: int, n_frames: int, resolution: int,
         cnn_train_trials: int, cnn_epochs: int, device: str):
    if not bench_npz.exists():
        generate_dataset(bench_npz, n_trials_per_condition=n_bench_trials,
                          n_frames=n_frames, resolution=resolution, seed=42)

    # The CNN must be trained at the SAME resolution it's evaluated at -- a
    # model trained on 64px frames and evaluated on 128px ones (needed for
    # tau's accuracy, see the resolution sweep in "Pilot findings") is not a
    # fair comparison, so always (re)generate a training set at the
    # benchmark's own resolution rather than reusing a differently-sized one.
    if not train_npz.exists():
        generate_dataset(train_npz, n_trials_per_condition=cnn_train_trials,
                          n_frames=n_frames, resolution=resolution, seed=41)

    print("=== Lee's tau (real rendered pixels) ===")
    tau_acc, tau_lat = tau_predictions(bench_npz, resolution_note=f"{resolution}px")
    print(f"accuracy: {tau_acc:.3f} | mean latency: {tau_lat:.4f} ms/frame\n")

    print("=== CNN (real rendered pixels) ===")
    cnn_acc, cnn_lat = cnn_predictions(train_npz, bench_npz, cnn_epochs, device)
    print(f"accuracy: {cnn_acc:.3f} | inference latency: {cnn_lat:.4f} ms/frame-pair ({device})\n")

    print("=== Connectome/LIF (synthetic flow matching known condition -- see module docstring) ===")
    conn = connectome_accuracy(sweep_json)
    for side, r in conn.items():
        print(f"{side:5s}: accuracy {r['accuracy']:.3f} over {r['n_conditions']} conditions "
              f"(threshold {r['threshold_hz']:.1f} Hz)")

    print("\n=== Summary ===")
    print(f"{'detector':12s} {'accuracy':>9s}  {'input':30s} {'training data':>14s}")
    print(f"{'tau':12s} {tau_acc:9.3f}  {'real MuJoCo pixels':30s} {'none':>14s}")
    print(f"{'CNN':12s} {cnn_acc:9.3f}  {'real MuJoCo pixels':30s} {'300 trials':>14s}")
    for side, r in conn.items():
        print(f"{'LIF (' + side + ')':12s} {r['accuracy']:9.3f}  "
              f"{'synthetic flow, matched condition':30s} {'none':>14s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", type=Path, default=Path("data/benchmark_train.npz"),
                     help="CNN training set -- kept separate from data/cnn_train.npz "
                          "since it must match --resolution exactly")
    ap.add_argument("--bench-npz", type=Path, default=Path("data/benchmark.npz"))
    ap.add_argument("--sweep-json", type=Path, default=Path("data/hemisphere_sweep.json"))
    ap.add_argument("--n-bench-trials", type=int, default=20, help="trials per condition")
    ap.add_argument("--cnn-train-trials", type=int, default=150, help="trials per condition")
    ap.add_argument("--n-frames", type=int, default=50)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--cnn-epochs", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    main(a.train_npz, a.bench_npz, a.sweep_json, a.n_bench_trials, a.n_frames,
         a.resolution, a.cnn_train_trials, a.cnn_epochs, a.device)
