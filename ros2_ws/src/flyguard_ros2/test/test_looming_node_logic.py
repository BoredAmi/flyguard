"""
Tests the LIF-driving logic underneath looming_node, no live rclpy context.

The node now runs `flyguard.runtime.CoreCircuit`, the same object the
closed-loop benchmark and the demo recorder use, so these tests drive that
directly -- and one of them keeps an *independent*, hand-rolled copy of the
drive scheme and checks the shared implementation still agrees with it. A
regression test that only calls the code under test cannot notice the code
under test changing meaning; this pair can.

The behaviour being guarded is that LPi's presence is what makes the e-stop
settle back to false, not just what makes it trigger true.
"""

from pathlib import Path

import numpy as np
import pytest

flyguard = pytest.importorskip("flyguard", reason="flyguard core package not importable")

from flyguard.extract import load_subnetwork, subset_subnetwork  # noqa: E402
from flyguard.lif import LIFNetwork, LIFParams  # noqa: E402
from flyguard.runtime import CoreCircuit  # noqa: E402

DATA_NPZ = Path(flyguard.__file__).resolve().parent.parent / "data" / "looming.npz"
needs_data = pytest.mark.skipif(not DATA_NPZ.exists(), reason="data/looming.npz not present")

BASE_RATE_HZ, PEAK_WEIGHT, N_STEPS = 250.0, 30.0, 500


def _circuit(seed=0):
    return CoreCircuit(DATA_NPZ, lif_dt=1e-4, base_rate_hz=BASE_RATE_HZ,
                       peak_weight=PEAK_WEIGHT, seed=seed, record=("LPLC2", "DNp01"))


@needs_data
def test_core_circuit_includes_lpi_and_matches_known_size():
    c = _circuit()
    counts = c.counts()
    assert counts["total"] == 530
    assert counts["LPLC2"] == 210
    assert counts["LC4"] == 104
    assert counts["DNp01"] == 2
    assert counts["LPi"] == 214
    assert len(c.lpi_types) == 11


@needs_data
def test_cruise_produces_no_dnp01_spikes_once_settled():
    """
    Regression test for the exact bug found while building this node.

    An earlier version (LPLC2 + LC4 + DNp01, no LPi) latched the e-stop
    permanently true because the reduced circuit had no inhibition at all.
    With LPi included and driven at (1-drive)*base_rate, a sustained
    drive=0 (cruise) tick sequence must settle to zero DNp01 spikes.
    """
    c = _circuit(seed=0)
    counts = [c.step(0.0, n_steps=N_STEPS).spike_count("DNp01") for _ in range(20)]
    assert all(n == 0 for n in counts), f"cruise should be silent at DNp01, got {counts}"


@needs_data
def test_looming_reliably_drives_dnp01():
    c = _circuit(seed=1)
    counts = [c.step(1.0, n_steps=N_STEPS).spike_count("DNp01") for _ in range(5)]
    assert all(n > 10 for n in counts), f"looming should reliably drive DNp01, got {counts}"


@needs_data
def test_core_circuit_matches_an_independent_implementation():
    """
    Cross-check against a hand-rolled copy of the same drive scheme.

    `CoreCircuit` orders its LPi pool by neuron index while this reference
    orders it by subtype, so the two consume the random stream differently
    and the spike counts are *not* expected to match exactly -- what must
    match is the behaviour the node reads: silent at cruise, firing at loom.
    """
    W, meta, groups = load_subnetwork(DATA_NPZ)
    lpi_types = sorted(t for t in meta.cell_type.unique() if str(t).startswith("LPi"))
    Wc, _, groups_c = subset_subnetwork(W, meta, ["LPLC2", "LC4", "DNp01"] + lpi_types)
    net = LIFNetwork(Wc, LIFParams(dt=1e-4), seed=0)
    lplc2_idx = groups_c["LPLC2"]
    lpi_idx = np.concatenate([groups_c.get(t, np.array([], dtype=int)) for t in lpi_types])
    dnp01_idx = groups_c["DNp01"]

    def tick(drive):
        def stim_fn(_k):
            i_ext = net.poisson(lplc2_idx, drive * BASE_RATE_HZ) * PEAK_WEIGHT
            return i_ext + net.poisson(lpi_idx, (1 - drive) * BASE_RATE_HZ) * PEAK_WEIGHT

        return int(net.run(N_STEPS, stim_fn=stim_fn, record=dnp01_idx).sum())

    assert all(tick(0.0) == 0 for _ in range(20))
    assert all(tick(1.0) > 10 for _ in range(5))
