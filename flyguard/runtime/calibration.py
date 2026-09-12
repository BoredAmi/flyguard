"""Measured operating points for one camera and one scene.

Four numbers stand between a working detector and a useless one, and none of
them can be guessed -- they depend on the camera's field of view, the
resolution, the texture of the world and how fast the vehicle moves. They are
measured, never hand-picked, and getting that wrong was instructive twice
over (both failures are on record in the project notes):

* `drive_center` comes from an **unobstructed** view -- "what open space
  looks like" -- so that ordinary cruising sits at drive ~0.5 and LPi's
  `(1 - drive)` channel keeps its range in both directions. Centring on
  obstacle-rich views instead put cruising near 0, where both LPLC2 pools
  fall nearly silent and the steering ratio becomes two small noisy numbers
  divided by each other (measured: p99 turn noise 0.232 against a median of
  0.037).

* `drive_scale` comes from the **obstacle** views -- the range real obstacles
  actually produce.

* `turn_offset` and `saccade_threshold` are measured on an **unobstructed**
  view, where by construction any signal is noise. Measuring them on obstacle
  views counted genuine obstacle responses as noise and put the saccade
  threshold at 0.30-0.43 when the true floor is ~0.02, so the vehicle almost
  never turned.

* `estop_threshold` is the deliberate exception and uses the **obstacle**
  views, because escape should mean "closer than almost any moment of an
  ordinary traversal", not "something is visible at all". Referenced to an
  empty view it fired on 87% of ticks and left the vehicle permanently
  braking.

A calibration is tied to the rig that produced it. The stored `encoder` and
`source` blocks are there so a mismatch is visible rather than silent: a
calibration measured at 96x96 with the ground masked out does not transfer to
a 480p forward camera, and `Calibration.check_compatible` says so out loud.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1


@dataclass
class Calibration:
    """Everything the runtime needs that is not anatomical.

    Note what is *not* in here: no synaptic weight, no cell-type selection, no
    circuit parameter. Those come from the connectome and stay fixed. This
    file holds only the sensor-conditioning constants, which is exactly the
    boundary the project's working agreement draws -- learned or fitted
    quantities never touch the connectome path.
    """

    drive_center: dict = field(default_factory=lambda: {"left": 0.0, "right": 0.0})
    drive_scale: float = 1.0
    turn_offset: float = 0.0
    saccade_threshold: float = float("inf")
    estop_threshold: float = float("inf")

    #: Encoder settings this calibration was measured under. A calibration is
    #: only valid for an encoder configured the same way.
    encoder: dict = field(default_factory=dict)
    #: Free-form provenance: what was measured, how many samples, when.
    source: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        # inf is not valid JSON; round-trip it as a string sentinel.
        for key in ("saccade_threshold", "estop_threshold"):
            if d[key] == float("inf"):
                d[key] = "inf"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        d = dict(d)
        version = d.pop("schema_version", SCHEMA_VERSION)
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"calibration schema version {version} is newer than this "
                f"flyguard understands ({SCHEMA_VERSION}); upgrade flyguard"
            )
        for key in ("saccade_threshold", "estop_threshold"):
            if isinstance(d.get(key), str):
                d[key] = float(d[key])
        known = {f for f in cls.__dataclass_fields__ if f != "schema_version"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown calibration fields: {sorted(unknown)}")
        return cls(**d)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "Calibration":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Measure one for your camera with:\n"
                f"    python -m flyguard.calibrate --help"
            )
        return cls.from_dict(json.loads(path.read_text()))

    # -- validation -------------------------------------------------------

    def check_compatible(self, encoder_settings: dict) -> list[str]:
        """Return human-readable warnings where the live encoder differs from
        the one this calibration was measured on. Empty list means a match.

        Returned rather than raised: a mismatch degrades the numbers, it does
        not make them meaningless, and a robot mid-flight should not crash on
        one. Callers are expected to log every line.
        """
        warnings = []
        for key, measured in sorted(self.encoder.items()):
            live = encoder_settings.get(key)
            if live is None:
                continue
            if isinstance(measured, (list, tuple)) and isinstance(live, (list, tuple)):
                same = list(measured) == list(live)
            else:
                same = measured == live
            if not same:
                warnings.append(
                    f"calibration was measured with {key}={measured!r} but the "
                    f"live encoder uses {key}={live!r}"
                )
        return warnings

    def is_measured(self) -> bool:
        """False for a default-constructed calibration, whose infinite
        thresholds mean "never turn, never brake"."""
        import math

        return math.isfinite(self.saccade_threshold) and self.drive_scale > 0
