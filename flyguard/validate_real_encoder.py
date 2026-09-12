"""Validates the real image-based encoder (`flyguard.encoder.flow_from_frame_pair`
+ `optical_flow`) against the real connectome, end to end: render a MuJoCo
trial, estimate optical flow between every consecutive frame pair, drive
the real extracted subnetwork's T4/T5 columns with that flow (not a
synthetic ground-truth flow field), and measure whether LPLC2 still
discriminates looming from translation.

This is the step the project notes flagged and deferred at every prior opportunity
(the offline benchmark's LIF row, the ROS2 node's `lplc2_drive` input) --
finally closing the loop from rendered pixels to the real connectome,
rather than from a trial's known ground-truth condition.

Architecture: for each consecutive rendered frame pair, estimate flow,
build a `poisson_stim_fn` for that pair, and run the LIF network forward
for the number of steps corresponding to one rendered frame's duration
(`dt_frame / dt_lif`) -- a real frame-by-frame replay, not a single static
flow field held for the whole trial (that's what the synthetic-flow
methodology in `validate_encoder.py` does; this script is deliberately
different because now the "stimulus" genuinely changes every frame).

Uses `mujoco_world.matched_depth_trial_params`, not `make_trial_params` --
the first version of this script used the latter and got a *reversed*
result (translation beating looming). Root cause: `make_trial_params`
draws looming and translation trials from different depth ranges (5-8m vs
3-5m), so translation discs render ~1.5-2x larger by area on average --
more boundary/edge pixels with detectable optical flow, hence more driven
T4/T5 columns, independent of the actual motion pattern. Lee's tau and the
CNN aren't affected by this (tau's formula divides out absolute size; the
CNN's own shortcut check already confirmed it isn't using size), but this
encoder's raw-pixel-evidence drive scheme is. See the project notes "Pilot
findings" for the full diagnosis and the controlled result once depth is
matched.

Usage:
    MUJOCO_GL=egl python -m flyguard.validate_real_encoder \
        --npz data/looming.npz --columns ~/flywire/column_assignment.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from flyguard.encoder import column_pixel_bounds, flow_from_frame_pair, load_column_assignment, poisson_stim_fn
from flyguard.extract import load_subnetwork
from flyguard.lif import LIFNetwork, LIFParams, rates
from flyguard.mujoco_world import SCENE_XML, matched_depth_trial_params, render_trial

import mujoco  # noqa: E402


def replay_trial_through_connectome(W, lplc2_idx, sub_ids, columns, x_bounds, y_bounds,
                                     frames, dt_frame, params: LIFParams,
                                     burn_frames: int, base_rate_hz: float, peak_weight: float,
                                     window: int, n_iters: int, seed: int) -> float:
    """Streams one rendered trial's frame pairs through the real network,
    frame by frame, and returns the LPLC2 mean firing rate over the
    post-burn-in portion."""
    net = LIFNetwork(W, params, seed=seed)
    n_steps_per_frame = max(1, round(dt_frame / params.dt))

    all_spikes = []
    for i in range(len(frames) - 1):
        flow_vx, flow_vy = flow_from_frame_pair(
            frames[i], frames[i + 1], columns, x_bounds, y_bounds, window=window, n_iters=n_iters
        )
        stim_fn = poisson_stim_fn(columns, flow_vx, flow_vy, sub_ids, dt=params.dt,
                                   base_rate_hz=base_rate_hz, peak_weight=peak_weight, seed=seed + i)
        spikes = net.run(n_steps_per_frame, stim_fn=stim_fn, record=lplc2_idx)
        all_spikes.append(spikes)

    spikes = np.concatenate(all_spikes, axis=0)
    post_burn = spikes[burn_frames * n_steps_per_frame:]
    return float(rates(post_burn, params.dt).mean()) if len(post_burn) else 0.0


def main(npz_path: Path, columns_path: Path, side: str, n_trials: int, n_frames: int,
         resolution: int, burn_frames: int, base_rate_hz: float, peak_weight: float,
         window: int, n_iters: int, seed: int, depth0: float):
    W, meta, groups = load_subnetwork(npz_path)
    lplc2_idx = np.flatnonzero(((meta.cell_type == "LPLC2") & (meta.side == side)).to_numpy())
    sub_ids = meta.root_id.to_numpy()
    print(f"LPLC2 ({side}): {len(lplc2_idx)} neurons")

    columns = load_column_assignment(columns_path, side=side)
    x_bounds, y_bounds = column_pixel_bounds(columns)
    print(f"T4/T5 ({side}) columns: {len(columns)}, x_bounds={x_bounds}, y_bounds={y_bounds}")

    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=resolution, width=resolution)

    params = LIFParams(dt=1e-4)
    dt_frame = 1.0 / 30.0

    try:
        results = {"looming": [], "translation": []}
        for condition in ("looming", "translation"):
            for trial in range(n_trials):
                p = matched_depth_trial_params(condition, seed=seed * 100 + trial, n_frames=n_frames,
                                                dt=dt_frame, depth0=depth0)
                frames = render_trial(model, renderer, data, p)
                rate = replay_trial_through_connectome(
                    W, lplc2_idx, sub_ids, columns, x_bounds, y_bounds, frames, dt_frame, params,
                    burn_frames, base_rate_hz, peak_weight, window, n_iters, seed=seed + trial
                )
                results[condition].append(rate)
                print(f"{condition:12s} trial {trial}: LPLC2 mean rate = {rate:.1f} Hz")
    finally:
        renderer.close()

    loom_mean = np.mean(results["looming"])
    trans_mean = np.mean(results["translation"])
    threshold = (loom_mean + trans_mean) / 2
    print(f"\nlooming mean: {loom_mean:.1f} Hz over {n_trials} trials")
    print(f"translation mean: {trans_mean:.1f} Hz over {n_trials} trials")
    print(f"ratio: {loom_mean / max(trans_mean, 1e-9):.2f}x")
    n_correct = sum(r > threshold for r in results["looming"]) + \
        sum(r <= threshold for r in results["translation"])
    print(f"threshold-based accuracy (midpoint rule, threshold={threshold:.1f} Hz): "
          f"{n_correct} / {2 * n_trials} = {n_correct / (2 * n_trials):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=Path("data/looming.npz"))
    ap.add_argument("--columns", type=Path, required=True)
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--n-trials", type=int, default=5, help="trials per condition")
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--burn-frames", type=int, default=5)
    ap.add_argument("--base-rate-hz", type=float, default=250.0)
    ap.add_argument("--peak-weight", type=float, default=30.0)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--n-iters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--depth0", type=float, default=5.0,
                     help="shared starting depth (m) for both conditions -- see "
                          "matched_depth_trial_params, avoids the size/depth confound "
                          "found in the first version of this script")
    a = ap.parse_args()
    main(a.npz, a.columns, a.side, a.n_trials, a.n_frames, a.resolution,
         a.burn_frames, a.base_rate_hz, a.peak_weight, a.window, a.n_iters, a.seed, a.depth0)
