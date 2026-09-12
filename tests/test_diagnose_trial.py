"""Tests for the single-trial post-mortem.

Only the classifier is tested here -- the replay itself needs MuJoCo, the
connectome and the FlyWire CSVs, and is exercised by running the script. The
classifier is the part that turns a trace into a claim, so it is the part that
can be wrong in a way nobody notices.
"""

import pytest

pytest.importorskip("mujoco", reason="flyguard.avoid needs MuJoCo for headless rendering")

from flyguard.diagnose_trial import _classify  # noqa: E402


def _rows(turns):
    return [{"t": i / 10, "turn": v} for i, v in enumerate(turns)]


def test_counts_ticks_that_cleared_the_threshold():
    stats = _classify(_rows([0.01, 0.2, -0.3, 0.05]), saccade_threshold=0.1)
    assert stats["ticks_over_threshold"] == 2
    assert stats["n_vision_ticks"] == 4


def test_peak_uses_magnitude_not_sign():
    """A hard turn the wrong way is still a large signal; reporting the signed
    max would hide it behind a smaller positive one."""
    stats = _classify(_rows([0.1, -0.9]), saccade_threshold=0.1)
    assert stats["peak_abs_turn"] == pytest.approx(0.9)


def test_headroom_says_how_far_below_threshold_a_silent_trace_sat():
    """The distinction that matters: a signal at 0.9x threshold is a
    calibration problem, one at 0.05x is a signal problem."""
    stats = _classify(_rows([0.09, 0.08]), saccade_threshold=0.1)
    assert stats["ticks_over_threshold"] == 0
    assert stats["headroom"] == pytest.approx(0.9)


def test_ticks_without_vision_are_skipped():
    """Saccade and refractory ticks carry no turn key; counting them as zeros
    would dilute the peak and understate the signal."""
    rows = [{"t": 0.0, "turn": 0.5}, {"t": 0.1}, {"t": 0.2, "turn": 0.4}]
    stats = _classify(rows, saccade_threshold=0.1)
    assert stats["n_vision_ticks"] == 2


def test_an_empty_trace_is_reported_not_crashed():
    assert "verdict" in _classify([], saccade_threshold=0.1)
