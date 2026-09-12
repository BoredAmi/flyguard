"""The robot-facing half of FlyGuard.

Everything here is stable, camera-agnostic and free of simulator imports.
The research layer (`flyguard.validate_*`, `flyguard.arena`,
`flyguard.baseline_*`) sits on top of it and is free to churn; this does not.

    from flyguard.runtime import FlyGuardPilot, Calibration

    pilot = FlyGuardPilot("column_assignment.csv", Calibration.load("cam.json"))
    cmd = pilot.step(frame)          # -> Command(v, omega, estop, telemetry)

What is anatomical and what is not, stated once:

* the 530 neurons, their synaptic weights and their signs come from FlyWire
  FAFB v783 and are never fitted;
* `Command.estop` maps onto DNp01, the Giant Fiber -- a real escape pathway;
* `Command.omega` does **not**. LPLC2/LC4 have zero direct edges onto the
  steering descending neurons, so `bilateral_turn` is an engineering
  addition. See `flyguard.runtime.steering`.
"""

from flyguard.runtime.calibration import Calibration
from flyguard.runtime.circuit import (
    BASE_CORE_TYPES,
    POPULATIONS,
    CircuitResponse,
    CoreCircuit,
    default_npz_path,
)
from flyguard.runtime.detector import BilateralEncoder
from flyguard.runtime.pilot import BACKENDS, FlyGuardPilot
from flyguard.runtime.ros_image import SUPPORTED_ENCODINGS, image_to_array
from flyguard.runtime.steering import (
    SaccadicSteering,
    as_sides,
    bilateral_turn,
    ema_trace,
    normalize_drive,
)
from flyguard.runtime.types import SIDES, Command, Percept

__all__ = [
    "BACKENDS",
    "BASE_CORE_TYPES",
    "POPULATIONS",
    "SIDES",
    "BilateralEncoder",
    "Calibration",
    "CircuitResponse",
    "Command",
    "CoreCircuit",
    "FlyGuardPilot",
    "Percept",
    "SUPPORTED_ENCODINGS",
    "SaccadicSteering",
    "as_sides",
    "image_to_array",
    "bilateral_turn",
    "default_npz_path",
    "ema_trace",
    "normalize_drive",
]
