"""The 530-neuron core escape circuit, as one reusable object.

Before this module the same circuit was constructed three separate times --
in `flyguard.avoid.ConnectomeController`, in the ROS2 `looming_node`, and in
`flyguard.record_live_demo` -- each repeating the cell-type selection, the
index bookkeeping, the dual-channel Poisson drive and the population-rate
readout. the project notes carried a standing warning to keep them in sync by hand.
They now all call this.

## What the circuit is

`data/looming.npz` holds an 18k-neuron hops=1 subnetwork around the looming
seeds, which is too large to run inside a control loop (0.8x realtime).
`CoreCircuit` restricts it to the cell types that actually carry the escape
computation:

    LPLC2   looming detectors, driven externally from optic flow
    LC4     the parallel looming pathway (excitatory partner of LPLC2)
    LPi     all 11 subtypes -- the inhibition radial motion opponency needs
    DNp01   the Giant Fiber, the escape command neuron

giving ~530 neurons at several times realtime on one CPU core. Every weight
is anatomical; nothing here is fitted.

**LPi is not optional.** Omitting it (an earlier version of the ROS2 node
did) leaves LPLC2/LC4/DNp01's real mutual excitation with nothing to damp
it, and the circuit latches into runaway activity that never releases -- the
same failure mode `flyguard.validate_ablation` measures deliberately as the
"LPi ablated" condition. `CoreCircuit` refuses to build without it.

## The dual-channel drive

Each tick the caller supplies a normalised drive in [0, 1]. LPLC2 is driven
at `drive * base_rate_hz` and LPi at `(1 - drive) * base_rate_hz`: LPi should
be *most* active exactly when the view looks least like a loom, mirroring
radial motion opponency's own logic rather than only ever exciting. This is
the same scheme `flyguard.stimuli.simulate_condition` validated on the
abstract ring circuit.

Drive is delivered as discrete Poisson impulses, never as a constant `i_ext`
held across steps. See the project notes' "Known gotcha": the 20-40 peak-weight
range is calibrated for an impulse, and re-injecting a static current every
step silently saturates the network at its refractory ceiling.

## Monocular vs bilateral

`step(0.7)` drives both hemispheres' pools together as one population -- what
the ROS2 node and the recorder do. `step({"left": 0.2, "right": 0.9})` drives
each hemisphere from its own hemifield, which is what closed-loop steering
needs. The two paths consume the random stream differently (one Poisson draw
over the union versus one per side), so results are reproducible within a
mode but not across them; this is deliberate, because it preserves the exact
stream each existing caller was already measured on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flyguard.extract import load_subnetwork, subset_subnetwork
from flyguard.lif import LIFNetwork, LIFParams
from flyguard.runtime.types import SIDES

#: Cell types pulled into the core circuit, besides the LPi subtypes which
#: are discovered by prefix (there are 11 and their names are data, not ours).
BASE_CORE_TYPES = ("LPLC2", "LC4", "DNp01")

#: Populations addressable by name in `indices()` and `CircuitResponse`.
POPULATIONS = ("LPLC2", "LC4", "LPi", "DNp01")


def default_npz_path() -> Path:
    """Locate `looming.npz`, searching the places it plausibly lives.

    Delegates to `flyguard.paths`, which checks $FLYGUARD_SUBNETWORK, the
    working directory and the package directory. Resolving it by position
    alone breaks a `pip install`: there is no `data/` beside the package in
    site-packages, so the naive guess points at a path that never exists.
    """
    from flyguard.paths import default_subnetwork_path

    return Path(default_subnetwork_path())


@dataclass
class CircuitResponse:
    """One tick of circuit output.

    `spikes` is the raw (n_steps, n_recorded) boolean array, with columns in
    the order given by `record_index`; the recorder uses it to draw a genuine
    per-neuron raster rather than a smoothed rate curve.
    """

    spikes: np.ndarray
    window_s: float
    record_index: np.ndarray
    _slices: dict

    def _slice(self, population: str, side: str | None):
        key = population if side is None else f"{population}:{side}"
        if key not in self._slices:
            raise KeyError(
                f"{key!r} was not recorded; CoreCircuit(record=...) has "
                f"{sorted({k.split(':')[0] for k in self._slices})}"
            )
        return self._slices[key]

    def spike_count(self, population: str, side: str | None = None) -> int:
        """Total spikes this tick. The escape readout uses the *count*, not
        "did it spike at all": a single Giant Fiber spike in a 100 ms window
        is not a commitment to escape, and with both hemispheres driven the
        any-spike test fires on essentially every tick (measured 26/30)."""
        return int(self.spikes[:, self._slice(population, side)].sum())

    def rate_hz(self, population: str, side: str | None = None) -> float:
        """Mean population firing rate over the tick, in Hz."""
        block = self.spikes[:, self._slice(population, side)]
        n = block.shape[1]
        return float(block.sum() / (n * self.window_s)) if n else 0.0

    def any_spike(self, population: str, side: str | None = None) -> bool:
        return bool(self.spikes[:, self._slice(population, side)].any())

    def size(self, population: str, side: str | None = None) -> int:
        sl = self._slice(population, side)
        return sl.stop - sl.start


class CoreCircuit:
    """The real core circuit, loaded once and stepped per control tick.

    Parameters
    ----------
    npz_path
        Subnetwork produced by `flyguard.extract`. Defaults to the packaged
        `data/looming.npz`.
    tick_hz
        Control-loop rate. Each `step()` advances the LIF simulation by one
        tick's worth of `lif_dt` steps.
    base_rate_hz, peak_weight
        Input scaling. `peak_weight` is the "Known gotcha" constant: unit
        weight is 0.275 mV per event, far below the ~26-synapse threshold,
        so external drive needs a weight in the 20-40 range to be felt.
    record
        Which populations to record. Recording costs memory per step, so the
        default is the two the controllers actually read.
    bilateral
        If True, pools are split by hemisphere and `step()` accepts a
        per-side drive mapping.
    """

    def __init__(
        self,
        npz_path: Path | str | None = None,
        *,
        tick_hz: float = 10.0,
        lif_dt: float = 1e-4,
        base_rate_hz: float = 250.0,
        peak_weight: float = 30.0,
        seed: int = 0,
        record: tuple[str, ...] = ("LPLC2", "DNp01"),
        bilateral: bool = False,
    ):
        from flyguard.paths import require_subnetwork_path

        npz_path = require_subnetwork_path(npz_path)
        self.npz_path = npz_path
        self.bilateral = bilateral
        self.base_rate_hz = float(base_rate_hz)
        self.peak_weight = float(peak_weight)
        self.seed = int(seed)

        W, meta, _ = load_subnetwork(npz_path)
        self.lpi_types = sorted(t for t in meta.cell_type.unique() if str(t).startswith("LPi"))
        if not self.lpi_types:
            raise RuntimeError(
                f"{npz_path} contains no LPi cell types. The core circuit needs "
                "them: without inhibition LPLC2/LC4/DNp01's real mutual excitation "
                "latches into runaway activity (see validate_ablation.py, the "
                "'LPi ablated' condition)."
            )
        Wc, meta_c, _ = subset_subnetwork(W, meta, list(BASE_CORE_TYPES) + self.lpi_types)
        self.W = Wc
        self.meta = meta_c
        self.n_neurons = int(Wc.shape[0])

        cell_type = meta_c.cell_type
        masks = {
            "LPLC2": (cell_type == "LPLC2").to_numpy(),
            "LC4": (cell_type == "LC4").to_numpy(),
            "DNp01": (cell_type == "DNp01").to_numpy(),
            "LPi": cell_type.str.startswith("LPi").to_numpy(),
        }
        side_col = meta_c.side.to_numpy()
        self._idx = {}
        for pop, mask in masks.items():
            self._idx[pop] = np.flatnonzero(mask)
            for side in SIDES:
                self._idx[f"{pop}:{side}"] = np.flatnonzero(mask & (side_col == side))

        for required in ("LPLC2", "DNp01"):
            if len(self._idx[required]) == 0:
                raise RuntimeError(f"core circuit has no {required} neurons -- check {npz_path}")

        self.params = LIFParams(dt=float(lif_dt))
        self.tick_hz = float(tick_hz)
        self.n_steps_per_tick = max(1, round((1.0 / self.tick_hz) / self.params.dt))

        # Recorded columns, laid out population-major so a response can slice
        # them by name without searching.
        self.record = tuple(record)
        keys = []
        for pop in self.record:
            keys.extend([f"{pop}:{s}" for s in SIDES] if bilateral else [pop])
        self._record_keys = keys
        blocks, self._slices, cursor = [], {}, 0
        for key in keys:
            idx = self._idx[key]
            blocks.append(idx)
            self._slices[key] = slice(cursor, cursor + len(idx))
            cursor += len(idx)
        if bilateral:
            # Each population's two hemisphere blocks are laid out adjacently,
            # so a whole-population query is just the span across both. The
            # escape readout needs this: DNp01 fires as one command channel,
            # not as a left and a right one.
            for pop in self.record:
                lo = self._slices[f"{pop}:{SIDES[0]}"].start
                hi = self._slices[f"{pop}:{SIDES[-1]}"].stop
                self._slices[pop] = slice(lo, hi)
        self._record_index = (
            np.concatenate(blocks) if blocks else np.array([], dtype=int)
        )
        self.reset()

    # -- introspection ----------------------------------------------------

    @property
    def record_index(self) -> np.ndarray:
        """Global neuron indices of the recorded columns, in column order.
        The demo recorder maps raster rows back to real neurons through this."""
        return self._record_index

    def indices(self, population: str, side: str | None = None) -> np.ndarray:
        key = population if side is None else f"{population}:{side}"
        if key not in self._idx:
            raise KeyError(f"unknown population {key!r}; have {sorted(self._idx)}")
        return self._idx[key]

    def counts(self) -> dict:
        """Neuron count per population, for a startup log line."""
        out = {p: len(self._idx[p]) for p in POPULATIONS}
        out["total"] = self.n_neurons
        return out

    def describe(self) -> str:
        c = self.counts()
        return (
            f"core circuit: {c['total']} neurons (LPLC2={c['LPLC2']}, LC4={c['LC4']}, "
            f"LPi={c['LPi']} across {len(self.lpi_types)} subtypes, DNp01={c['DNp01']})"
        )

    # -- simulation -------------------------------------------------------

    def reset(self) -> None:
        """Rebuild the network state. Deterministic for a given seed."""
        self.net = LIFNetwork(self.W, self.params, seed=self.seed)

    def step(self, drive, *, n_steps: int | None = None) -> CircuitResponse:
        """Advance one control tick under `drive`.

        `drive` is a float in [0, 1] for a monocular circuit, or a mapping
        with "left"/"right" keys for a bilateral one.
        """
        n_steps = self.n_steps_per_tick if n_steps is None else int(n_steps)

        if self.bilateral:
            if not hasattr(drive, "keys"):
                drive = {s: float(drive) for s in SIDES}
            rates = {s: float(np.clip(drive[s], 0.0, 1.0)) for s in SIDES}
        else:
            if hasattr(drive, "keys"):
                raise TypeError(
                    "this CoreCircuit is monocular; pass a float, or build it "
                    "with CoreCircuit(bilateral=True)"
                )
            rates = {"both": float(np.clip(drive, 0.0, 1.0))}

        if self.bilateral:
            # Order matters: left then right, LPLC2 then LPi within each side.
            # This is the exact sequence of Poisson draws the closed-loop
            # benchmark in the project notes was measured on; changing it reshuffles
            # the random stream and silently moves every published number.
            pools = [
                (self._idx[f"LPLC2:{s}"], rates[s], self._idx[f"LPi:{s}"]) for s in SIDES
            ]
        else:
            pools = [(self._idx["LPLC2"], rates["both"], self._idx["LPi"])]

        def stim_fn(_step):
            i_ext = np.zeros(self.n_neurons, dtype=np.float32)
            for exc_idx, d, inh_idx in pools:
                if len(exc_idx):
                    i_ext += self.net.poisson(exc_idx, d * self.base_rate_hz)
                if len(inh_idx):
                    i_ext += self.net.poisson(inh_idx, (1.0 - d) * self.base_rate_hz)
            return i_ext * self.peak_weight

        spikes = self.net.run(n_steps, stim_fn=stim_fn, record=self._record_index)
        slices = dict(self._slices)
        if not self.bilateral:
            # Let callers ask for a side on a monocular circuit and get the
            # combined block, rather than a confusing KeyError.
            for pop in self.record:
                for side in SIDES:
                    slices.setdefault(f"{pop}:{side}", self._slices[pop])
        return CircuitResponse(
            spikes=spikes,
            window_s=n_steps * self.params.dt,
            record_index=self._record_index,
            _slices=slices,
        )
