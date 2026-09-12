"""Small value types passed across the runtime boundary.

Kept free of numpy-heavy machinery and of any simulator import so that a
robot integration can depend on these without pulling in MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SIDES = ("left", "right")


@dataclass
class Command:
    """What the pilot asks the vehicle to do this tick.

    `v` and `omega` are in whatever units the caller's kinematics use -- the
    runtime treats `v_cruise` as the unit and scales from there, so a
    differential-drive base can read them as m/s and rad/s directly.

    `estop` is the one field that maps onto a measured anatomical pathway
    (DNp01, the Giant Fiber). `omega` does not; see `bilateral_turn`.
    """

    v: float
    omega: float
    estop: bool = False
    telemetry: dict = field(default_factory=dict)


@dataclass
class Percept:
    """Per-hemisphere pooled T4/T5 drive for one frame pair.

    Subscriptable so that it can be passed anywhere the older dict-shaped
    percept was accepted: `percept["drive_left"]` and `percept.drive_left`
    are the same value.
    """

    drive_left: float
    drive_right: float
    speed_left: float = 0.0
    speed_right: float = 0.0

    def __getitem__(self, key: str) -> float:
        try:
            return getattr(self, key)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise KeyError(key) from exc

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def keys(self):
        return self.as_dict().keys()

    def values(self):
        return self.as_dict().values()

    def items(self):
        return self.as_dict().items()

    def as_dict(self) -> dict:
        return {
            "drive_left": self.drive_left,
            "drive_right": self.drive_right,
            "speed_left": self.speed_left,
            "speed_right": self.speed_right,
        }

    @classmethod
    def from_dict(cls, d) -> "Percept":
        if isinstance(d, cls):
            return d
        return cls(
            drive_left=float(d["drive_left"]),
            drive_right=float(d["drive_right"]),
            speed_left=float(d.get("speed_left", 0.0)),
            speed_right=float(d.get("speed_right", 0.0)),
        )
