"""The whole loop as one object: push frames, get commands.

    from flyguard.runtime import FlyGuardPilot, Calibration

    pilot = FlyGuardPilot(
        columns_csv="column_assignment.csv",
        calibration=Calibration.load("my_camera.json"),
    )
    while camera.is_open():
        cmd = pilot.step(camera.read())
        base.drive(cmd.v, cmd.omega)

No ROS2, no MuJoCo, no arena. `flyguard[mujoco]` and the ROS2 package are
both optional on top of this; the pilot itself needs only numpy, scipy and
pandas, which is what lets it run on a companion computer that could never
host a simulator.

## What runs each tick

    frame pair -> pyramidal Lucas-Kanade -> per-hemisphere T4/T5 columns
               -> normalise against the calibration
               -> 530 real neurons (LIF, anatomical weights)
               -> LPLC2 population rates + DNp01 spike count
               -> saccadic steering -> Command(v, omega, estop)

The `flow` backend is the same path with the circuit removed, and it exists
because it is the only honest way to answer "what do 530 real neurons add
over the signal you fed them". On this project's corridor benchmark the
answer so far is: less than the raw flow, and almost all of the circuit's
advantage came from the braking channel rather than the steering one. Keep
the baseline available in any deployment -- a robot is a much harsher test
than the benchmark, and the comparison is the point.

## Timing

`step()` takes an optional `t` in seconds. Omit it and the pilot reads a
monotonic clock, which is what you want on a robot; pass it explicitly for
deterministic replay. The saccade state machine is driven entirely by these
timestamps, so a stalled camera cannot leave it mid-saccade forever.
"""

from __future__ import annotations

import time
from pathlib import Path

from flyguard.runtime.calibration import Calibration
from flyguard.runtime.circuit import CoreCircuit
from flyguard.runtime.detector import BilateralEncoder
from flyguard.runtime.steering import (
    SaccadicSteering,
    as_sides,
    bilateral_turn,
    normalize_drive,
)
from flyguard.runtime.types import SIDES, Command, Percept

BACKENDS = ("connectome", "flow")


class FlyGuardPilot:
    """Frames in, `Command` out.

    Parameters
    ----------
    columns_csv
        FlyWire `column_assignment.csv`, the retinotopic map that says where
        each real T4/T5 neuron looks.
    calibration
        A `Calibration` measured on *this* camera. A default-constructed one
        has infinite thresholds, meaning "never turn, never brake" -- the
        pilot warns rather than pretending.
    backend
        "connectome" runs the real circuit; "flow" is the no-circuit baseline
        with identical gains, normalisation and calibration.
    """

    def __init__(
        self,
        columns_csv: Path | str,
        calibration: Calibration | None = None,
        *,
        backend: str = "connectome",
        npz_path: Path | str | None = None,
        tick_hz: float = 10.0,
        encoder: BilateralEncoder | None = None,
        seed: int = 1,
        base_rate_hz: float = 250.0,
        peak_weight: float = 30.0,
        logger=None,
        **steering_kwargs,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.backend = backend
        self.calibration = calibration or Calibration()
        self._log = logger or (lambda msg: None)

        self.encoder = encoder or BilateralEncoder(Path(columns_csv))
        for warning in self.calibration.check_compatible(self.encoder.settings()):
            self._log(f"calibration mismatch: {warning}")
        if not self.calibration.is_measured():
            self._log(
                "no measured calibration: thresholds are infinite, so the pilot "
                "will cruise straight and never brake. Measure one with "
                "`python -m flyguard.calibrate`."
            )

        self.drive_center = as_sides(self.calibration.drive_center)
        self.drive_scale = float(self.calibration.drive_scale)

        self.steering = SaccadicSteering(
            turn_offset=self.calibration.turn_offset,
            saccade_threshold=self.calibration.saccade_threshold,
            estop_threshold=self.calibration.estop_threshold,
            **steering_kwargs,
        )

        self.circuit = None
        if backend == "connectome":
            self.circuit = CoreCircuit(
                npz_path,
                tick_hz=tick_hz,
                base_rate_hz=base_rate_hz,
                peak_weight=peak_weight,
                seed=seed,
                record=("LPLC2", "DNp01"),
                bilateral=True,
            )
            self._log(self.circuit.describe())

        self._prev_frame = None
        self._t0 = None

    # -- lifecycle --------------------------------------------------------

    def reset(self) -> None:
        """Clear the frame history and all timing state. Call this whenever
        the video stream is interrupted -- a frame pair straddling a gap
        produces a huge spurious flow field."""
        self._prev_frame = None
        self._t0 = None
        self.steering.reset()
        if self.circuit is not None:
            self.circuit.reset()

    def _clock(self, t: float | None) -> float:
        if t is not None:
            return float(t)
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        return now - self._t0

    # -- the loop ---------------------------------------------------------

    def perceive(self, frame) -> Percept | None:
        """Encode one frame against its predecessor. Returns None for the
        first frame of a stream, which has no pair."""
        prev, self._prev_frame = self._prev_frame, frame
        if prev is None:
            return None
        return self.encoder(prev, frame)

    def step(self, frame, t: float | None = None) -> Command:
        """One control tick: frame in, command out."""
        t = self._clock(t)
        percept = self.perceive(frame)
        if percept is None:
            return self.steering.cruise()
        return self.act(percept, t)

    def act(self, percept, t: float) -> Command:
        """The half of `step` after perception, exposed separately so a
        recorded percept stream can be replayed without re-running flow."""
        percept = Percept.from_dict(percept)
        norm = {
            side: normalize_drive(
                percept[f"drive_{side}"], self.drive_center[side], self.drive_scale
            )
            for side in SIDES
        }
        telemetry = {f"norm_{s}": norm[s] for s in SIDES}

        if self.circuit is None:
            # Mean, not max: the normalised drive clips at 1.0, so a max-based
            # statistic pins at the ceiling whenever either side clips.
            turn = bilateral_turn(norm["left"], norm["right"])
            estop_stat = 0.5 * (norm["left"] + norm["right"])
        else:
            response = self.circuit.step(norm)
            rate_l = response.rate_hz("LPLC2", "left")
            rate_r = response.rate_hz("LPLC2", "right")
            estop_stat = float(response.spike_count("DNp01"))
            turn = bilateral_turn(rate_l, rate_r)
            telemetry.update({
                "lplc2_left_hz": rate_l,
                "lplc2_right_hz": rate_r,
                "dnp01_spikes": int(estop_stat),
            })

        return self.steering.step(turn, estop_stat, t, telemetry)
