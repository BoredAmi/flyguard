"""Camera frames in, per-hemisphere T4/T5 drive out.

This is the only part of the runtime that touches pixels, and it is
deliberately free of any simulator, robot or connectome import -- give it two
consecutive grayscale or RGB frames from any camera and it returns the two
numbers the circuit consumes. `flyguard.avoid` uses it against MuJoCo renders
and the ROS2 camera node uses it against `sensor_msgs/Image`; neither is
special to it.

## Why the drive is magnitude-weighted by default

`encode_drive`'s original form divides each column's flow vector by its
magnitude, which is correct for the unit-magnitude synthetic fields it was
built for and close to fatal on real flow, where it reduces the pooled drive
to roughly "how many columns have any measurable flow at all". Measured on
rendered frames: obstacle-versus-empty contrast **1.01x**, and a turn signal
of -0.011/+0.012 for an obstacle 1.4 m off to one side -- right in sign, but
a factor of ~40 smaller than the magnitude-weighted encoder's and
indistinguishable from drift in closed loop. With magnitude weighting the
same obstacle raises absolute drive **2.02x** and the left/right difference
reaches **-0.464/+0.461**.

`flyguard.encoder.encode_drive` still defaults to the unweighted form so that
every earlier result in this repo reproduces untouched; the runtime turns it
on.

## Why the flow estimator is pyramidal

Lucas-Kanade fails by *collapsing toward zero* on displacements larger than
its window, not by saturating -- so surfaces close to a moving camera, which
sweep past fastest, read as **emptier than open space**. Measured pre-fix:
pooled drive 0.0138 hard against a wall versus 0.0259 mid-corridor. That is a
sign inversion, and the steering law turned into the wall. No gain tuning
fixes a sign error. The coarse-to-fine pyramid recovers an 11 px shift to
0.03 px where warp-and-resolve alone returns ~0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from flyguard.encoder import (
    column_pixel_bounds,
    encode_drive,
    load_column_assignment,
    sample_flow_rect,
)
from flyguard.optical_flow import lucas_kanade_flow_pyramidal, to_grayscale
from flyguard.runtime.types import Percept


class BilateralEncoder:
    """Splits the eye view down the middle and drives each optic lobe from
    its own hemifield.

    Verified against the renderer rather than assumed (see
    `tests/test_arena.py::test_image_side_matches_world_side`): with the
    camera rig in `flyguard.arena`, world +y (the agent's left) projects to
    the *left* half of the image, so image-left is the left eye.

    `mirror` selects the horizontal convention for mapping each hemisphere's
    retinotopic columns onto its half-image:

    * ``"none"`` -- both hemispheres use the same mapping as every other
      experiment in this repo (column +x, which `flyguard.encoder` fixes as
      rostral/front, maps to the right edge of the region).
    * ``"anatomical"`` -- the right hemifield is mirrored, because for the
      right eye the rostral direction points toward the image *centre*,
      while for the left eye it points toward the image edge. This is the
      geometrically motivated choice, and it is exactly the per-hemisphere
      ambiguity CLAUDE.md flags as unresolved.

    Both are provided because neither is confirmed. Running the closed loop
    under both is a *behavioural* test of the convention, which is more than
    the open-loop sweeps could offer.

    `row_band` keeps only a horizontal band of the image, as (top, bottom)
    fractions, so the ground plane can be excluded. Driving forward over a
    textured floor generates strong optic flow, and the natural expectation
    is that it is pure nuisance. `validate_arena --sections bands` measures
    the turn signal for an obstacle 1-4 m away offset 1.4 m to one side,
    against an empty corridor:

        band          obst left   obst right   signal   empty sd    SNR
        full image      -0.214       +0.241     0.227      0.038    6.0
        upper 60%       -0.464       +0.461     0.463      0.115    4.0
        upper 40%       -0.543       +0.523     0.533      0.150    3.6
        lower 40%       -0.003       +0.018     0.010      0.014    0.7

    Half of that expectation holds and half does not, which is why the table
    is here rather than a one-line justification. The ground genuinely
    carries **no obstacle signal** (0.010, SNR 0.7). But excluding it
    amplifies the noise more than the signal and *lowers* SNR, because the
    floor is densely textured and flows consistently, which stabilises the
    denominator of the contrast-normalised ratio; the sky and horizon that
    remain are sparser and noisier.

    An earlier version of this measurement divided by the *mean* empty
    -corridor turn rather than its standard deviation. A mean can sit near
    zero through cancellation while the tick-to-tick spread is large, and it
    reported the opposite ranking. The lesson is the one this repo keeps
    relearning: check that the statistic means what the decision needs it to
    mean.

    **Neither advantage survives into behaviour.** Running the whole
    closed-loop benchmark under both bands (`--row-band 0 1`, 10 arenas):
    the flow controller is identical (60% collisions, 30% goals either way),
    and the connectome controller differs by a single arena, trading 80% ->
    70% collisions for 15.7 -> 14.3 m mean distance. At this sample size the
    parameter is not load-bearing in either direction, so the default is kept
    only for continuity with the recorded demo -- it is not a tuned choice,
    and nothing in the results rests on it.

    Note also that masking the ground at all is a concession the *fly* does
    not need: it is forced by this encoder pooling uniformly over the whole
    hemifield, rather than by LPLC2's real localised receptive fields.
    """

    def __init__(self, columns_csv: Path, mirror: str = "anatomical",
                 magnitude_weighted: bool = True, sat_px: float = 2.0,
                 window: int = 13, n_iters: int = 3, n_levels: int = 3,
                 row_band: tuple[float, float] = (0.0, 0.6)):
        if mirror not in ("none", "anatomical"):
            raise ValueError(f"unknown mirror convention {mirror!r}")
        self.mirror = mirror
        self.row_band = row_band
        self.magnitude_weighted = magnitude_weighted
        self.sat_px = sat_px
        self.window = window
        self.n_iters = n_iters
        self.n_levels = n_levels
        self.columns = {
            "left": load_column_assignment(columns_csv, side="left"),
            "right": load_column_assignment(columns_csv, side="right"),
        }
        self.bounds = {s: column_pixel_bounds(c) for s, c in self.columns.items()}
        # Only the right hemifield flips, and only under "anatomical".
        self.flip = {"left": False, "right": mirror == "anatomical"}

    def n_columns(self, side: str) -> int:
        return len(self.columns[side])

    def settings(self) -> dict:
        """The configuration a `Calibration` has to match to be valid here."""
        return {
            "mirror": self.mirror,
            "row_band": list(self.row_band),
            "magnitude_weighted": self.magnitude_weighted,
            "sat_px": self.sat_px,
            "window": self.window,
            "n_iters": self.n_iters,
            "n_levels": self.n_levels,
        }

    def __call__(self, prev_frame: np.ndarray, frame: np.ndarray) -> Percept:
        """Returns per-side mean T4/T5 drive plus the raw flow speed, which
        the diagnostics use to separate "the signal changed" from "the flow
        estimate changed".

        The result is a `Percept`, which is also subscriptable and iterable
        like the plain dict this used to return."""
        g1, g2 = to_grayscale(prev_frame), to_grayscale(frame)
        vx, vy = lucas_kanade_flow_pyramidal(g1, g2, window=self.window,
                                             n_iters=self.n_iters, n_levels=self.n_levels)
        h, w = vx.shape
        r0, r1 = int(h * self.row_band[0]), max(int(h * self.row_band[1]), int(h * self.row_band[0]) + 1)
        vx, vy = vx[r0:r1], vy[r0:r1]
        mid = w // 2
        halves = {"left": (vx[:, :mid], vy[:, :mid]), "right": (vx[:, mid:], vy[:, mid:])}

        out = {}
        for side, (hx, hy) in halves.items():
            xb, yb = self.bounds[side]
            fvx, fvy = sample_flow_rect(hx, hy, self.columns[side], xb, yb,
                                        flip_x=self.flip[side])
            drive = encode_drive(self.columns[side], fvx, fvy,
                                 magnitude_weighted=self.magnitude_weighted,
                                 sat_px=self.sat_px)
            out[f"drive_{side}"] = float(drive.mean())
            out[f"speed_{side}"] = float(np.hypot(hx, hy).mean())
        return Percept(**out)
