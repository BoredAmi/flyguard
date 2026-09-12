"""
Tests the pure-logic pieces of vision_node without a live rclpy context.

Covers the scene-position formula (translation sweep stays centered and
bounded, looming approaches and clips) and the drive normalisation --
including a regression test for the calibration bug this node hit live:
a single-ended normaliser (sum / max) put the translation baseline itself
at drive~0.35-0.65, giving looming_node's LPi inhibition no real contrast
and latching /flyguard/estop permanently true. The fix anchors both ends
of the linear map to the measured means instead.
"""

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="mujoco is an optional extra")

from flyguard_ros2.vision_node import DRIVE_SUM_HIGH, DRIVE_SUM_LOW  # noqa: E402


def _scene_xy(t, cruise_s, loom_s, depth0, radius, loom_speed):
    """
    Standalone re-implementation of VisionNode._scene_xy's math.

    Lets this be tested without constructing a live rclpy Node.
    """
    cycle = cruise_s + loom_s
    phase = t % cycle
    if phase < cruise_s:
        half_width = depth0 * 0.4142
        max_excursion = 0.6 * half_width
        speed = (2 * max_excursion) / cruise_s
        y = speed * phase - speed * cruise_s / 2.0
        return depth0, y
    loom_t = phase - cruise_s
    x = max(depth0 - loom_speed * loom_t, radius * 1.5)
    return x, 0.0


def _normalize(drive_sum):
    span = DRIVE_SUM_HIGH - DRIVE_SUM_LOW
    return float(np.clip((drive_sum - DRIVE_SUM_LOW) / span, 0.0, 1.0))


def test_translation_phase_holds_constant_depth():
    depths = [_scene_xy(t, 4.0, 2.0, 5.0, 0.3, 2.0)[0] for t in np.linspace(0, 3.9, 20)]
    assert all(d == pytest.approx(5.0) for d in depths)


def test_translation_phase_sweep_is_centered_and_bounded():
    ys = [_scene_xy(t, 4.0, 2.0, 5.0, 0.3, 2.0)[1] for t in np.linspace(0, 4.0, 41)]
    half_width = 5.0 * 0.4142
    max_excursion = 0.6 * half_width
    assert max(abs(y) for y in ys) <= max_excursion + 1e-9
    # sweep is centered: roughly symmetric around 0 across the phase
    assert ys[0] == pytest.approx(-max_excursion, abs=1e-6)


def test_looming_phase_approaches_and_clips_at_radius():
    xs = [_scene_xy(t, 4.0, 2.0, 5.0, 0.3, 2.0)[0] for t in np.linspace(4.0, 5.99, 40)]
    assert xs[0] > xs[-1], "should be getting closer over the looming phase"
    assert all(x >= 0.3 * 1.5 - 1e-9 for x in xs), "should never clip through the camera"


def test_scene_xy_cycles_deterministically():
    a = _scene_xy(1.5, 4.0, 2.0, 5.0, 0.3, 2.0)
    b = _scene_xy(1.5 + (4.0 + 2.0), 4.0, 2.0, 5.0, 0.3, 2.0)
    assert a == pytest.approx(b)


def test_drive_normalization_translation_mean_is_low():
    """
    Regression test for the live calibration bug.

    The fixed two-point normaliser should put a translation-typical drive
    sum near 0, not the ~0.35-0.65 the old single-ended (sum / max)
    version gave live.
    """
    normalized = _normalize(400.9)  # measured translation mean, see the project notes
    assert normalized < 0.05


def test_drive_normalization_looming_mean_is_high():
    normalized = _normalize(566.3)  # measured looming mean, see the project notes
    assert normalized > 0.95


def test_drive_normalization_clips_outside_measured_range():
    assert _normalize(0.0) == 0.0
    assert _normalize(10000.0) == 1.0


def test_drive_normalization_is_monotonic():
    sums = np.linspace(300, 700, 20)
    normalized = [_normalize(s) for s in sums]
    assert all(a <= b for a, b in zip(normalized, normalized[1:]))
