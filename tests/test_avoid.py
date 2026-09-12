"""Unit tests for the closed-loop controller logic.

None of these need MuJoCo, the connectome, or the retinotopic CSV: the
steering law and the saccade state machine are pure functions of the numbers
handed to them, and every failure mode this module hit during development
(wrong turn sign, saturated normalisation, saccade/escape feedback loop) is
reproducible without rendering a single frame. That is deliberate -- these
are the regression tests for bugs that were originally found live.
"""

import math


import pytest

# Skip the whole module rather than failing collection: an optional
# dependency missing must not abort the entire test run.
pytest.importorskip("mujoco", reason="flyguard.avoid needs MuJoCo for headless rendering")

from flyguard.arena import AgentState, ArenaConfig
from flyguard.avoid import (
    Command,
    FlowController,
    StraightController,
    _as_sides,
    bilateral_turn,
    step_agent_clamped,
)


# --- the steering law ------------------------------------------------------


def test_bilateral_turn_turns_away_from_the_louder_side():
    # right hemisphere louder -> obstacle on the right -> turn LEFT -> positive
    assert bilateral_turn(0.2, 0.8) > 0
    # left louder -> turn RIGHT -> negative
    assert bilateral_turn(0.8, 0.2) < 0


def test_bilateral_turn_is_zero_for_a_symmetric_view():
    """A frontal obstacle drives both hemispheres equally. This is not a
    rounding detail -- it is the blind spot that makes the escape channel
    necessary (see `_BilateralController._command`)."""
    assert bilateral_turn(0.5, 0.5) == pytest.approx(0.0)


def test_bilateral_turn_is_contrast_normalised():
    """Scaling both sides must not change the command. (The divide-by-zero
    guard makes this exact only to ~1e-6, which is why the tolerance is
    absolute rather than the default relative one.)"""
    assert bilateral_turn(0.1, 0.3) == pytest.approx(bilateral_turn(0.2, 0.6), abs=1e-5)


def test_bilateral_turn_handles_a_silent_circuit():
    assert bilateral_turn(0.0, 0.0) == 0.0


def test_bilateral_turn_is_bounded():
    assert -1.0 <= bilateral_turn(0.0, 1.0) <= 1.0
    assert -1.0 <= bilateral_turn(1.0, 0.0) <= 1.0


def test_as_sides_accepts_a_scalar_or_a_mapping():
    assert _as_sides(0.5) == {"left": 0.5, "right": 0.5}
    assert _as_sides({"left": 1.0, "right": 2.0}) == {"left": 1.0, "right": 2.0}


# --- normalisation ---------------------------------------------------------


def _controller(**kw):
    params = dict(drive_center={"left": 0.10, "right": 0.12}, drive_scale=0.05)
    params.update(kw)
    c = FlowController(**params)
    c.reset()
    return c


def test_norm_centres_each_side_on_its_own_resting_level():
    """Per-side offset is the whole point: two hemispheres at different
    resting drive must both map to 0.5, or the steering law inherits a
    constant bias that has nothing to do with obstacles."""
    c = _controller()
    assert c._norm("left", 0.10) == pytest.approx(0.5)
    assert c._norm("right", 0.12) == pytest.approx(0.5)


def test_norm_uses_a_shared_gain_so_the_difference_survives():
    c = _controller()
    # equal *excursions* from each side's centre must give equal normalised
    # values, otherwise a left/right difference is manufactured by scaling
    assert c._norm("left", 0.10 + 0.02) == pytest.approx(c._norm("right", 0.12 + 0.02))


def test_norm_clips_into_the_unit_interval():
    c = _controller()
    assert c._norm("left", 10.0) == 1.0
    assert c._norm("left", -10.0) == 0.0


# --- the saccade state machine --------------------------------------------


def _percept(dl, dr):
    return {"drive_left": dl, "drive_right": dr}


def test_cruises_straight_when_nothing_is_there():
    c = _controller()
    c.saccade_threshold, c.estop_threshold = 0.2, 10.0
    cmd = c(_percept(0.10, 0.12), t=0.0)
    assert cmd.omega == 0.0
    assert cmd.v == c.v_cruise
    assert not cmd.estop


def test_evidence_above_threshold_fires_a_saccade_away_from_the_obstacle():
    c = _controller()
    c.saccade_threshold, c.estop_threshold = 0.1, 10.0
    # drive much higher on the right -> obstacle right -> turn left -> omega > 0
    cmd = c(_percept(0.10, 0.20), t=0.0)
    assert cmd.omega > 0
    cmd = _controller()
    cmd.saccade_threshold, cmd.estop_threshold = 0.1, 10.0
    out = cmd(_percept(0.20, 0.10), t=0.0)
    assert out.omega < 0


def test_saccade_holds_its_direction_for_its_full_duration():
    c = _controller(saccade_duration_s=0.3)
    c.saccade_threshold, c.estop_threshold = 0.1, 10.0
    first = c(_percept(0.10, 0.20), t=0.0)
    assert first.omega > 0
    # even if the view now says the opposite, the saccade completes
    for t in (0.1, 0.2):
        assert c(_percept(0.20, 0.10), t=t).omega == pytest.approx(first.omega)


def test_vision_is_suppressed_during_the_saccade_and_its_settling_period():
    """Regression test for a feedback loop found by tracing a live run:
    rotational flow during a saccade pinned both hemifields at the top of
    their range, which fired the escape, which fired the next saccade. The
    agent oscillated between -37 and +46 degrees while braking."""
    c = _controller(saccade_duration_s=0.3, refractory_s=0.3, estop_cooldown_s=0.2)
    c.saccade_threshold, c.estop_threshold = 0.1, 0.9
    c(_percept(0.10, 0.20), t=0.0)  # start a saccade
    # A saturating percept (both sides pinned high) must not trigger an escape
    # while the saccade is running or settling.
    for t in (0.1, 0.2, 0.35, 0.55):
        cmd = c(_percept(10.0, 10.0), t=t)
        assert not cmd.estop, f"escape fired during suppression at t={t}"
    # once settled, the same percept is allowed to trigger again
    assert c(_percept(10.0, 10.0), t=0.7).estop


def test_refractory_blocks_back_to_back_saccades():
    c = _controller(saccade_duration_s=0.3, refractory_s=0.3)
    c.saccade_threshold, c.estop_threshold = 0.1, 10.0
    c(_percept(0.10, 0.20), t=0.0)
    assert c(_percept(0.10, 0.20), t=0.45).omega == 0.0   # settling
    assert c(_percept(0.10, 0.20), t=0.7).omega != 0.0    # free again


def test_escape_fires_a_saccade_when_the_view_is_symmetric():
    """A frontal obstacle leaves the left/right difference at ~0, so only the
    escape channel can act on it. Without this the agent drives straight into
    a centred obstacle -- measured, at exactly the same place the vision-free
    baseline hit it."""
    c = _controller()
    c.saccade_threshold, c.estop_threshold = 0.5, 0.6
    cmd = c(_percept(0.30, 0.30), t=0.0)  # symmetric but far above threshold
    assert cmd.estop
    assert cmd.omega != 0.0


def test_escape_brakes_rather_than_seizing_the_steering():
    """DNp01 licenses a stop, not a turn. An earlier version overrode omega
    with a full-rate turn and drove into the nearest wall every run."""
    c = _controller(saccade_duration_s=0.0, refractory_s=0.0)
    c.saccade_threshold, c.estop_threshold = 10.0, 0.6   # never enough turn evidence
    cmd = c(_percept(0.30, 0.30), t=0.0)
    assert cmd.estop
    assert cmd.v == c.v_escape
    assert cmd.v < c.v_cruise


def test_estop_stays_latched_for_its_cooldown():
    c = _controller(estop_cooldown_s=0.4)
    c.saccade_threshold, c.estop_threshold = 10.0, 0.6
    assert c(_percept(0.30, 0.30), t=0.0).estop
    assert c(_percept(0.10, 0.12), t=0.2).estop      # still latched
    assert not c(_percept(0.10, 0.12), t=0.5).estop  # expired


def test_reset_clears_saccade_and_estop_state():
    c = _controller()
    c.saccade_threshold, c.estop_threshold = 0.1, 0.6
    c(_percept(0.10, 0.30), t=0.0)
    c.reset()
    assert c(_percept(0.10, 0.12), t=0.0).omega == 0.0


def test_controller_without_a_percept_cruises():
    c = _controller()
    assert c(None, t=0.0).omega == 0.0
    assert c(None, t=0.0).v == c.v_cruise


def test_straight_controller_ignores_vision_entirely():
    s = StraightController()
    s.reset()
    assert s.needs_vision is False
    assert s(None, 0.0).omega == 0.0
    assert s(_percept(0.0, 5.0), 1.0).omega == 0.0


# --- trial mechanics -------------------------------------------------------


def test_step_agent_clamped_bounds_the_heading():
    """Without the clamp an agent that gives up can spin or drive back down
    the corridor and time out, which scores as 'did not collide'."""
    cfg = ArenaConfig()
    s = AgentState(0.0, 0.0, 0.0)
    for _ in range(200):
        s = step_agent_clamped(s, v=1.0, omega=5.0, dt=0.1, cfg=cfg)
    assert abs(s.heading) <= math.radians(80.0) + 1e-9


def test_command_defaults():
    c = Command(v=1.0, omega=0.0)
    assert c.estop is False
    assert c.telemetry == {}
