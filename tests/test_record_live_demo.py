import numpy as np

import pytest

# Skip the whole module rather than failing collection: an optional
# dependency missing must not abort the entire test run.
pytest.importorskip("mujoco", reason="the recorder renders its scene with MuJoCo")

from flyguard.record_live_demo import _scene_xy, sample_neuron_ids


def test_scene_xy_translation_holds_constant_depth():
    depths = [_scene_xy(t, 4.0, 2.0, 5.0, 0.3, 2.0)[0] for t in np.linspace(0, 3.9, 20)]
    assert all(d == pytest.approx(5.0) for d in depths)


def test_scene_xy_translation_sweep_is_centered_and_bounded():
    ys = [_scene_xy(t, 4.0, 2.0, 5.0, 0.3, 2.0)[1] for t in np.linspace(0, 4.0, 41)]
    half_width = 5.0 * 0.4142
    max_excursion = 0.6 * half_width
    assert max(abs(y) for y in ys) <= max_excursion + 1e-9


def test_scene_xy_looming_approaches_and_clips():
    xs = [_scene_xy(t, 4.0, 2.0, 5.0, 0.3, 2.0)[0] for t in np.linspace(4.0, 5.99, 40)]
    assert xs[0] > xs[-1]
    assert all(x >= 0.3 * 1.5 - 1e-9 for x in xs)


def test_scene_xy_cycles_deterministically():
    a = _scene_xy(1.5, 4.0, 2.0, 5.0, 0.3, 2.0)
    b = _scene_xy(1.5 + 6.0, 4.0, 2.0, 5.0, 0.3, 2.0)
    assert a == pytest.approx(b)


def test_sample_neuron_ids_reproducible_and_disjoint():
    groups = {
        "LPLC2": np.arange(0, 210),
        "LC4": np.arange(210, 314),
        "LPi01": np.arange(314, 400),
        "LPi02": np.arange(400, 500),
        "DNp01": np.arange(500, 502),
    }
    a = sample_neuron_ids(groups, n_lplc2=24, n_lpi=16)
    b = sample_neuron_ids(groups, n_lplc2=24, n_lpi=16)
    np.testing.assert_array_equal(a["LPLC2"], b["LPLC2"])
    np.testing.assert_array_equal(a["LPi"], b["LPi"])
    assert len(a["LPLC2"]) == 24
    assert len(a["LPi"]) == 16
    assert len(a["DNp01"]) == 2
    assert set(a["LPLC2"]).issubset(set(groups["LPLC2"]))
    assert set(a["LPi"]).issubset(set(groups["LPi01"]) | set(groups["LPi02"]))


def test_sample_neuron_ids_handles_small_groups_gracefully():
    groups = {"LPLC2": np.arange(5), "DNp01": np.arange(5, 7)}
    sampled = sample_neuron_ids(groups, n_lplc2=24, n_lpi=16)
    assert len(sampled["LPLC2"]) == 5  # can't sample more than exist
    assert len(sampled["LPi"]) == 0
