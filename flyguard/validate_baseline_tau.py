"""Validates the Lee's-tau baseline against ground truth (the project's task 4).

Two things get checked numerically, not by eye:

1. Accuracy: for a looming trial, `flyguard.baseline_tau`'s estimate should
   track the trial's true (exactly known, constant-velocity) time-to-contact.
   Swept across render resolution and smoothing window, because the first
   unsmoothed 64x64 attempt was unusable (mean abs error ~1.1s against a
   true tau range of ~1.5-2.8s) -- pixel-count quantization noise on a
   ~4-7px-radius disc gets amplified by finite differencing. This sweep is
   what justifies the resolution/smoothing defaults, rather than picking
   them by feel.
2. Specificity: for a translation trial (no real collision course), tau
   should stay `inf` almost everywhere -- the classical cue should not
   raise a false collision alarm just because an object is moving.

Usage:
    MUJOCO_GL=egl python -m flyguard.validate_baseline_tau
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from flyguard.baseline_tau import apparent_radius_from_area, lee_tau, true_time_to_contact
from flyguard.mujoco_world import SCENE_XML, TrialParams, apparent_area_px, render_trial

import mujoco  # noqa: E402


def accuracy_sweep():
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    p = TrialParams(condition="looming", n_frames=40, dt=1 / 30, depth0=6.0, radius=0.3, speed=2.0)
    true_tau = true_time_to_contact(p.depth0, p.speed, p.radius, p.dt, p.n_frames)

    print("resolution/smoothing sweep (looming trial, accuracy vs. ground truth):")
    print(f"{'res':>5s} {'window':>7s} {'finite':>8s} {'mean_err':>9s} {'median_err':>11s}")
    for res, window in [(64, 1), (64, 5), (64, 9), (128, 1), (128, 5), (256, 1)]:
        renderer = mujoco.Renderer(model, height=res, width=res)
        try:
            frames = render_trial(model, renderer, data, p)
            area = apparent_area_px(frames)
            radius = apparent_radius_from_area(area)
            est_tau = lee_tau(radius, p.dt, smooth_window=window)
        finally:
            renderer.close()
        finite = np.isfinite(est_tau)
        err = np.abs(est_tau[finite] - true_tau[finite])
        print(f"{res:5d} {window:7d} {finite.sum():5d}/{len(true_tau)} "
              f"{err.mean():9.3f} {np.median(err):11.3f}")


def specificity_check(warning_threshold_s: float = 3.0):
    """"Finite" is the wrong bar for a false alarm: quantization noise on an
    essentially-constant apparent size produces plenty of technically-finite
    tau values, but they're huge (tens to tens-of-thousands of seconds --
    see the raw trace in this module's docstring history / the project notes).
    A real system acts on tau dropping *below* some warning threshold, not
    on mere finiteness -- so that's what gets checked here. Threshold
    default (3s) matches the true time-to-contact range of the looming
    trials used in `accuracy_sweep` (~1.5-2.8s), i.e. "would this have
    looked as urgent as a real approach during this trial's duration."
    """
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=128, width=128)
    try:
        results = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            p = TrialParams(condition="translation", n_frames=40, dt=1 / 30,
                             depth0=float(rng.uniform(3.0, 5.0)), radius=0.3,
                             speed=float(rng.uniform(1.0, 2.0)))
            frames = render_trial(model, renderer, data, p)
            area = apparent_area_px(frames)
            radius = apparent_radius_from_area(area)
            est_tau = lee_tau(radius, p.dt, smooth_window=5)
            finite = est_tau[np.isfinite(est_tau)]
            n_alarm = int((est_tau < warning_threshold_s).sum())
            results.append(n_alarm)
            min_finite = finite.min() if len(finite) else float("nan")
            print(f"translation trial {seed}: {len(finite)}/{len(est_tau)} frames finite "
                  f"(min finite tau = {min_finite:.1f}s) | "
                  f"{n_alarm} frames below the {warning_threshold_s:.0f}s warning threshold")
    finally:
        renderer.close()
    print(f"\ntotal false alarms (tau < {warning_threshold_s:.0f}s) across "
          f"{len(results)} translation trials: {sum(results)}")


if __name__ == "__main__":
    accuracy_sweep()
    print()
    specificity_check()
