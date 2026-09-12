"""Tests for the single-trial post-mortem.

Only the classifier is tested here -- the replay itself needs MuJoCo, the
connectome and the FlyWire CSVs, and is exercised by running the script. The
classifier is the part that turns a trace into a claim, so it is the part that
can be wrong in a way nobody notices.
"""

import pytest

pytest.importorskip("mujoco", reason="flyguard.avoid needs MuJoCo for headless rendering")

from flyguard.diagnose_trial import _classify  # noqa: E402


def _rows(turns, emas=None):
    emas = emas if emas is not None else turns
    return [{"t": i / 10, "turn": v, "turn_ema": e} for i, (v, e) in enumerate(zip(turns, emas))]


def test_counts_ticks_that_cleared_the_threshold():
    stats = _classify(_rows([0.01, 0.2, -0.3, 0.05]), saccade_threshold=0.1, turn_ema_threshold=1.0)
    assert stats["ticks_over_threshold"] == 2
    assert stats["n_vision_ticks"] == 4


def test_peak_uses_magnitude_not_sign():
    """A hard turn the wrong way is still a large signal; reporting the signed
    max would hide it behind a smaller positive one."""
    stats = _classify(_rows([0.1, -0.9]), saccade_threshold=0.1, turn_ema_threshold=1.0)
    assert stats["peak_abs_turn"] == pytest.approx(0.9)


def test_headroom_says_how_far_below_threshold_a_silent_trace_sat():
    """The distinction that matters: a signal at 0.9x threshold is a
    calibration problem, one at 0.05x is a signal problem."""
    stats = _classify(_rows([0.09, 0.08]), saccade_threshold=0.1, turn_ema_threshold=1.0)
    assert stats["ticks_over_threshold"] == 0
    assert stats["headroom"] == pytest.approx(0.9)


def test_ticks_without_vision_are_skipped():
    """Saccade and refractory ticks carry no turn key; counting them as zeros
    would dilute the peak and understate the signal."""
    rows = [{"t": 0.0, "turn": 0.5, "turn_ema": 0.5}, {"t": 0.1}, {"t": 0.2, "turn": 0.4, "turn_ema": 0.4}]
    stats = _classify(rows, saccade_threshold=0.1, turn_ema_threshold=1.0)
    assert stats["n_vision_ticks"] == 2


def test_an_empty_trace_is_reported_not_crashed():
    assert "verdict" in _classify([], saccade_threshold=0.1, turn_ema_threshold=1.0)


def test_ema_over_threshold_counted_independently_of_instantaneous():
    """The whole point of the EMA channel: a trace that never once clears the
    instantaneous threshold can still clear the smoothed one."""
    turns = [0.05] * 20          # never crosses 0.1 on any single tick
    emas = [0.03 * i for i in range(20)]  # but the smoothed trace ramps past 0.2
    stats = _classify(_rows(turns, emas), saccade_threshold=0.1, turn_ema_threshold=0.2)
    assert stats["ticks_over_threshold"] == 0
    assert stats["ticks_over_ema_threshold"] > 0


def test_ema_peak_uses_magnitude_too():
    stats = _classify(_rows([0.0, 0.0], [0.05, -0.4]), saccade_threshold=0.1, turn_ema_threshold=0.2)
    assert stats["peak_abs_turn_ema"] == pytest.approx(0.4)
