"""FlyGuard -- collision avoidance from the *Drosophila* connectome.

A looming detector whose synaptic weights are read straight out of the
FlyWire FAFB v783 connectome. No training, no fitted parameters, no learned
weights anywhere in the detection path.

Robot integration starts here::

    from flyguard import FlyGuardPilot, Calibration

    pilot = FlyGuardPilot("column_assignment.csv", Calibration.load("cam.json"))
    cmd = pilot.step(frame)     # Command(v, omega, estop, telemetry)

`flyguard.runtime` holds that robot-facing API and depends on nothing heavier
than numpy, scipy and pandas. Everything else in the package is the research
layer that produced and validates it -- connectome extraction
(`flyguard.extract`), the LIF engine (`flyguard.lif`), the retinotopic encoder
(`flyguard.encoder`), baselines, simulated worlds and the `validate_*`
scripts behind every number in the README.

Two things to know before putting this on a vehicle, both measured rather
than assumed:

1. `Command.estop` corresponds to a real pathway -- LPLC2/LC4 onto DNp01, the
   Giant Fiber. `Command.omega` does not: the extracted subnetwork has zero
   direct edges onto the steering descending neurons, so the turn signal is
   an engineering addition layered on top. See `flyguard.runtime.steering`.
2. Run the `flow` backend as a control. It is the identical pipeline with the
   530 neurons removed, and on this project's own corridor benchmark the
   circuit did not beat it.
"""

from flyguard.runtime import (
    BilateralEncoder,
    Calibration,
    Command,
    CoreCircuit,
    FlyGuardPilot,
    Percept,
    SaccadicSteering,
    bilateral_turn,
)

try:  # keep one source of truth for the version: the installed metadata
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("flyguard")
except (ImportError, PackageNotFoundError):  # pragma: no cover - source checkout
    __version__ = "0.2.0"

__all__ = [
    "BilateralEncoder",
    "Calibration",
    "Command",
    "CoreCircuit",
    "FlyGuardPilot",
    "Percept",
    "SaccadicSteering",
    "__version__",
    "bilateral_turn",
]
