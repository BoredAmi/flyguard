"""Layer 2 of the architecture (the project notes): MuJoCo headless rendering of the
looming-vs-translation stimulus pair as real images, offline, into `.npz`
datasets -- the step between the synthetic flow fields in `flyguard.stimuli`
and the offline benchmark that will replay these frames through the LIF
detector, a flow-divergence baseline, and a CNN baseline.

Stimulus design follows the classic paradigm this project is modeling
(Klapoetke et al. 2017): a dark disc against a textured backdrop either
**expands** (looming -- an object approaching the eye) or **translates**
laterally at constant depth (self-motion-analog control -- same object,
same starting angular size, no net expansion). The disc's position is
scripted via a MuJoCo mocap body (kinematic, not simulated) rather than
free-body physics: there is no collision or dynamics to integrate here,
only a stimulus trajectory, so scripting position directly as a function of
frame index keeps rendering exactly reproducible and independent of CPU
speed (no physics timestep to drift), per the project notes' hardware-constraints
note. Rendering itself is GPU-accelerated via EGL; nothing downstream of
the rendered frames needs the GPU.

Usage:
    MUJOCO_GL=egl python -m flyguard.mujoco_world --out data/mujoco_dataset.npz
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Headless rendering requires MUJOCO_GL=egl. Set before `mujoco` is imported
# if the caller's environment hasn't already chosen a backend.
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

# Camera at the world origin, fixed, looking down world +x with world +z up.
# xyaxes gives the camera's local x/y axes in world coordinates; MuJoCo
# cameras look along their local -z, so local x=(0,-1,0), local y=(0,0,1)
# gives local z = x-cross-y = (-1,0,0) -> looks along world +x. Verified by
# `test_camera_looks_down_world_x` (checked against an object placed on-axis).
SCENE_XML = """
<mujoco>
  <visual>
    <headlight ambient="1 1 1" diffuse="1 1 1" specular="0 0 0"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".6 .6 .7" rgb2="1 1 1"
             width="300" height="300"/>
    <material name="grid_mat" texture="grid" texrepeat="10 10" reflectance="0"/>
  </asset>
  <worldbody>
    <light pos="0 0 0" dir="1 0 0" diffuse="1 1 1" directional="true"/>
    <camera name="eye" pos="0 0 0" xyaxes="0 -1 0 0 0 1"/>
    <geom name="backdrop" type="plane" pos="12 0 0" euler="0 90 0" size="25 25 0.1" material="grid_mat"/>
    <body name="stimulus" mocap="true" pos="6 0 0">
      <geom name="disc" type="sphere" size="0.3" rgba="0 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""


@dataclass(frozen=True)
class TrialParams:
    condition: str          # "looming" or "translation"
    n_frames: int
    dt: float                # seconds between frames
    depth0: float             # starting distance along the camera axis (m)
    radius: float              # physical sphere radius (m)
    speed: float                # approach speed (looming, m/s) or lateral speed (translation, m/s)
    y0: float = 0.0             # starting lateral offset (m), translation only


def trajectory(p: TrialParams) -> np.ndarray:
    """Per-frame (x, y) mocap position, world coordinates, shape [n_frames, 2]."""
    t = np.arange(p.n_frames) * p.dt
    if p.condition == "looming":
        x = p.depth0 - p.speed * t
        x = np.clip(x, p.radius * 1.5, None)  # never let the disc reach the camera
        y = np.full(p.n_frames, p.y0)
    elif p.condition == "translation":
        x = np.full(p.n_frames, p.depth0)
        y = p.y0 + p.speed * t - p.speed * (p.n_frames * p.dt) / 2.0  # sweep centered on y0
    else:
        raise ValueError(f"unknown condition {p.condition!r}")
    return np.stack([x, y], axis=1)


def render_trial(model, renderer, data, p: TrialParams) -> np.ndarray:
    """Render one trial. Returns frames [n_frames, H, W, 3] uint8."""
    positions = trajectory(p)
    frames = np.empty((p.n_frames, renderer.height, renderer.width, 3), dtype=np.uint8)
    for i, (x, y) in enumerate(positions):
        data.mocap_pos[0] = [x, y, 0.0]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="eye")
        frames[i] = renderer.render()
    return frames


def apparent_area_px(frames: np.ndarray, dark_thresh: int = 70) -> np.ndarray:
    """Per-frame count of dark (stimulus) pixels -- proportional to the
    disc's apparent *area*, not its radius (area ~ pixel count; radius ~
    sqrt(pixel count)). Used to numerically verify looming vs. translation
    behave as intended (see `validate_mujoco_world.py`) and as the raw
    signal for the Lee's-tau baseline (`flyguard.baseline_tau`), without
    requiring visual inspection of the rendered images."""
    dark = (frames.astype(np.int32).sum(axis=-1) / 3) < dark_thresh
    return dark.reshape(frames.shape[0], -1).sum(axis=1).astype(np.float64)


CAMERA_HALF_FOV_TAN = 0.4142  # tan(22.5 deg): default MuJoCo fovy=45, square image -> fovx=fovy


def make_trial_params(condition: str, seed: int, n_frames: int = 60, dt: float = 1.0 / 30.0) -> TrialParams:
    """Mildly randomised (but seeded, reproducible) trial parameters, so a
    dataset has more than one identical trial per condition.

    Translation's lateral speed is derived from depth (rather than drawn
    independently of it) so the disc's sweep stays within a safe fraction
    of the visible frame width for the whole trial regardless of depth or
    trial duration -- an object that exits and re-enters frame is not the
    "uniform translation at constant depth" control stimulus this is meant
    to be, and pollutes the size-constancy check (see `validate_mujoco_world.py`).
    """
    rng = np.random.default_rng(seed)
    radius = 0.3
    if condition == "looming":
        depth0 = float(rng.uniform(5.0, 8.0))
        speed = float(rng.uniform(1.5, 3.0))
        y0 = float(rng.uniform(-0.3, 0.3))
    else:
        depth0 = float(rng.uniform(3.0, 5.0))
        duration = n_frames * dt
        half_width = depth0 * CAMERA_HALF_FOV_TAN
        max_excursion = 0.6 * half_width  # stay within 60% of the half-frame width
        sweep_fraction = float(rng.uniform(0.4, 1.0))  # how much of that budget to use
        speed = sweep_fraction * (2 * max_excursion) / duration
        y0 = 0.0
    return TrialParams(condition=condition, n_frames=n_frames, dt=dt,
                        depth0=depth0, radius=radius, speed=speed, y0=y0)


def matched_depth_trial_params(condition: str, seed: int, n_frames: int = 60,
                                dt: float = 1.0 / 30.0, depth0: float = 5.0) -> TrialParams:
    """Same idea as `make_trial_params`, but both conditions start at the
    *same* depth (default 5.0m, same physical radius) rather than
    `make_trial_params`'s independently-drawn ranges (looming 5-8m,
    translation 3-5m).

    Exists because of a real confound found while building the real
    image-based encoder (`validate_real_encoder.py`, see the project notes "Pilot
    findings"): a raw-pixel-evidence drive scheme (more boundary/edge
    pixels with detectable optical flow -> more driven T4/T5 columns ->
    more aggregate LIF excitation) is not size-invariant the way Lee's tau
    (which divides by size, `r/(dr/dt)`) or the CNN (verified via its own
    shortcut check) are. `make_trial_params`'s mismatched depth ranges mean
    translation discs are reliably ~1.5-2x larger by rendered area than
    looming discs, which was enough to reverse the real-encoder's
    discrimination (translation reading as more "looming" than looming)
    until depth was controlled for. Existing baselines (tau, CNN, and
    `mujoco_world`'s own size-constancy checks) are unaffected by this and
    don't need to switch to this function -- only a size-sensitive raw
    -evidence drive scheme like the real-image encoder does.
    """
    rng = np.random.default_rng(seed)
    radius = 0.3
    if condition == "looming":
        speed = float(rng.uniform(1.5, 3.0))
        y0 = float(rng.uniform(-0.3, 0.3))
    elif condition == "translation":
        duration = n_frames * dt
        half_width = depth0 * CAMERA_HALF_FOV_TAN
        max_excursion = 0.6 * half_width
        sweep_fraction = float(rng.uniform(0.4, 1.0))
        speed = sweep_fraction * (2 * max_excursion) / duration
        y0 = 0.0
    else:
        raise ValueError(f"unknown condition {condition!r}")
    return TrialParams(condition=condition, n_frames=n_frames, dt=dt,
                        depth0=depth0, radius=radius, speed=speed, y0=y0)


def generate_dataset(out: Path, n_trials_per_condition: int = 5, n_frames: int = 60,
                      resolution: int = 64, seed: int = 0):
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=resolution, width=resolution)
    try:
        all_frames, conditions, trial_ids, params_out = [], [], [], []
        trial_id = 0
        for condition in ("looming", "translation"):
            for k in range(n_trials_per_condition):
                p = make_trial_params(condition, seed=seed * 1000 + trial_id, n_frames=n_frames)
                frames = render_trial(model, renderer, data, p)
                all_frames.append(frames)
                conditions.append(condition)
                trial_ids.append(trial_id)
                params_out.append((p.depth0, p.radius, p.speed, p.y0, p.dt))
                trial_id += 1
        frames_arr = np.stack(all_frames)  # [n_trials, n_frames, H, W, 3]
        params_arr = np.array(params_out, dtype=np.float64)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            frames=frames_arr,
            condition=np.array(conditions, dtype="U16"),
            trial_id=np.array(trial_ids, dtype=np.int64),
            params=params_arr,  # columns: depth0, radius, speed, y0, dt
            seed=seed,
        )
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): "
              f"{frames_arr.shape[0]} trials x {frames_arr.shape[1]} frames "
              f"@ {resolution}x{resolution}")
    finally:
        renderer.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/mujoco_dataset.npz"),
                     help="Rendered image sequences are regenerable from this "
                          "script and are not meant to be committed.")
    ap.add_argument("--n-trials", type=int, default=5, help="trials per condition")
    ap.add_argument("--n-frames", type=int, default=60)
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    generate_dataset(a.out, a.n_trials, a.n_frames, a.resolution, a.seed)
