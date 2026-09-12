"""Lee's tau baseline (CLAUDE.md task 4): classical time-to-contact from
optic flow divergence, as a non-connectome comparison point for the LIF
detector's discrimination behaviour.

Lee (1976) showed that for an object approaching at constant velocity, the
time remaining to contact equals the instantaneous apparent size divided by
its rate of expansion:

    tau(t) = r(t) / (dr/dt)

where r is the object's linear angular size (NOT its area -- area grows as
r^2, so using area directly instead of its square root silently scales tau
by a factor of 2, see `test_baseline_tau.py`). This requires no depth
sensor and no learned parameters -- exactly the kind of "cheap, geometric,
explainable" baseline a connectome-derived detector should be measured
against. Unlike the LIF/LPLC2 circuit, tau has no opinion about looming
vs. translation as *categories*: it only knows about expansion rate, so a
translating (non-expanding) object correctly reads as "not on a collision
course" (tau undefined / very large), which is a different failure mode
than the LIF circuit's opponency mechanism, worth comparing.

Uses `flyguard.mujoco_world.apparent_area_px` as the raw per-frame size
signal, so it runs on the same rendered datasets as the connectome
detector -- an apples-to-apples input, not a separate synthetic stimulus.
"""

from __future__ import annotations

import numpy as np

from flyguard.mujoco_world import apparent_area_px


def apparent_radius_from_area(area_px: np.ndarray) -> np.ndarray:
    """Convert a pixel-count area signal to a radius-like linear size,
    r ~ sqrt(area / pi). Assumes a roughly circular silhouette (true for
    this project's disc stimulus); a different shape would need its own
    area-to-linear-size conversion."""
    return np.sqrt(np.maximum(area_px, 0.0) / np.pi)


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average, edge-padded so the output stays the same
    length. `window=1` is a no-op. Real vision-based TTC estimators always
    filter the size signal before differentiating it -- see module
    docstring and `validate_baseline_tau.py` for why this matters here."""
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    pad = window // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, kernel, mode="valid")[:len(x)]


def lee_tau(radius_px: np.ndarray, dt: float, smooth_window: int = 1) -> np.ndarray:
    """Time-to-contact estimate per frame, in seconds.

    Central differences (via `np.gradient`) for the interior, one-sided at
    the edges. Where the rate of expansion is <= 0 (shrinking or constant
    apparent size -- no imminent collision by this cue), returns `np.inf`:
    there is no finite time-to-contact to report, and treating that as
    "very large" rather than a divide-by-zero error is the semantically
    correct behaviour.

    `smooth_window` (odd, e.g. 5) pre-smooths `radius_px` before
    differentiating -- single-pixel quantization noise in a small rendered
    disc gets amplified by finite differencing into large tau errors (see
    `validate_baseline_tau.py`'s resolution/smoothing sweep); smoothing is
    the standard fix, along with rendering at higher resolution.
    """
    radius_px = smooth(radius_px, smooth_window)
    n = len(radius_px)
    dr_dt = np.gradient(radius_px, dt)
    tau = np.full(n, np.inf)
    positive = dr_dt > 1e-9
    tau[positive] = radius_px[positive] / dr_dt[positive]
    return tau


def tau_series(frames: np.ndarray, dt: float, dark_thresh: int = 70, smooth_window: int = 1) -> np.ndarray:
    """End-to-end: rendered frames -> per-frame Lee's tau estimate (seconds)."""
    area = apparent_area_px(frames, dark_thresh=dark_thresh)
    radius = apparent_radius_from_area(area)
    return lee_tau(radius, dt, smooth_window=smooth_window)


def true_time_to_contact(depth0: float, speed: float, radius: float, dt: float, n_frames: int) -> np.ndarray:
    """Ground-truth time-to-contact for a `flyguard.mujoco_world` looming
    trial: the disc approaches at constant speed along the camera axis and
    "contact" is defined the same way `mujoco_world.trajectory` clips
    it -- depth reaching 1.5x the physical radius. Exact and linear in t by
    construction (constant-velocity approach), used to validate `lee_tau`'s
    estimate against ground truth rather than only checking its qualitative
    shape."""
    t = np.arange(n_frames) * dt
    contact_depth = radius * 1.5
    return (depth0 - speed * t - contact_depth) / speed
