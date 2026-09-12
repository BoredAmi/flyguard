"""The smallest useful integration: frames in, commands out, no ROS2.

    python examples/minimal_loop.py --columns column_assignment.csv

Runs the whole pipeline against a synthetic camera so it works with nothing
plugged in. Swap `FakeCamera` for your own `read()` and the rest is unchanged
-- that is the entire integration surface.

To adapt this to a real vehicle you need three things, in this order:

1. **A calibration measured on your camera** (`python -m flyguard.calibrate`).
   Without one the pilot cruises straight and never brakes. This example
   measures one inline from its own synthetic footage, which is exactly what
   you should *not* do on hardware: calibrate once, save the JSON, load it.

2. **A speed the numbers were measured at.** The calibration encodes how much
   optic flow ordinary travel produces, so it is only valid near the speed it
   was recorded at. Change cruise speed, re-calibrate.

3. **The `flow` baseline, run at least once** (`backend="flow"`). It is this
   same pipeline with the 530 neurons removed and everything else identical.
   On this project's corridor benchmark it beat the circuit. Measure yours.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from flyguard import Calibration, FlyGuardPilot
from flyguard.calibrate import measure
from flyguard.runtime import BilateralEncoder


class FakeCamera:
    """A drifting texture, optionally zooming to imitate an approach.

    Replace this with your camera. The pilot needs a uint8 array of shape
    (H, W) or (H, W, 3) and nothing else -- no timestamps, no intrinsics, no
    calibration matrix.
    """

    def __init__(self, resolution=96, approaching=False, seed=0):
        rng = np.random.default_rng(seed)
        self.field = (rng.random((resolution * 3, resolution * 3)) * 255).astype(np.uint8)
        self.resolution = resolution
        self.approaching = approaching
        self.t = 0

    def read(self) -> np.ndarray:
        from scipy import ndimage

        r = self.resolution
        patch = self.field[r : 2 * r, r + self.t : 2 * r + self.t]
        if self.approaching:
            zoomed = ndimage.zoom(patch, 1.0 + 0.05 * self.t, order=1)
            top = (zoomed.shape[0] - r) // 2
            left = (zoomed.shape[1] - r) // 2
            patch = zoomed[top : top + r, left : left + r]
        self.t += 1
        return patch.astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--columns", type=Path, required=True,
                    help="FlyWire column_assignment.csv")
    ap.add_argument("--npz", type=Path, default=None)
    ap.add_argument("--backend", choices=("connectome", "flow"), default="connectome")
    ap.add_argument("--calibration", type=Path, default=None,
                    help="a saved calibration; without one, this example measures "
                         "a throwaway one from its own synthetic footage")
    ap.add_argument("--ticks", type=int, default=20)
    a = ap.parse_args()

    encoder = BilateralEncoder(a.columns)
    tick_hz = 10.0

    if a.calibration:
        calibration = Calibration.load(a.calibration)
    else:
        print("measuring a throwaway calibration (on hardware: do this once, save it)")
        # One camera per recording, read repeatedly -- a fresh camera per frame
        # would replay frame zero every time and yield two identical clips.
        clear_cam, obstacle_cam = FakeCamera(approaching=False), FakeCamera(approaching=True)
        clear = np.stack([clear_cam.read() for _ in range(12)])
        obstacle = np.stack([obstacle_cam.read() for _ in range(12)])
        calibration, _ = measure(clear, obstacle, encoder, backend=a.backend,
                                 npz_path=a.npz, tick_hz=tick_hz)

    pilot = FlyGuardPilot(a.columns, calibration, backend=a.backend,
                          npz_path=a.npz, tick_hz=tick_hz,
                          v_cruise=0.3, v_escape=0.09, logger=print)

    camera = FakeCamera(approaching=True)
    print(f"\n{'t':>5} {'v':>6} {'omega':>7} {'estop':>6}  telemetry")
    for tick in range(a.ticks):
        t = tick / tick_hz
        cmd = pilot.step(camera.read(), t=t)

        # ---- this is where you would command the vehicle ----
        # base.drive(linear=cmd.v, angular=cmd.omega)
        # if cmd.estop: base.brake()

        extra = ""
        if "dnp01_spikes" in cmd.telemetry:
            extra = (f"LPLC2 {cmd.telemetry['lplc2_left_hz']:5.0f}/"
                     f"{cmd.telemetry['lplc2_right_hz']:<5.0f} Hz  "
                     f"DNp01 {cmd.telemetry['dnp01_spikes']:3d}")
        print(f"{t:5.1f} {cmd.v:6.2f} {cmd.omega:+7.2f} {str(cmd.estop):>6}  {extra}")

    print("\nReminder: `estop` follows a real pathway (LPLC2/LC4 -> DNp01, the Giant\n"
          "Fiber). `omega` does not -- the connectome has no edges from here to the\n"
          "steering descending neurons, so the turn is ours. Run --backend flow to\n"
          "see what the same pipeline does without the 530 neurons.")


if __name__ == "__main__":
    main()
