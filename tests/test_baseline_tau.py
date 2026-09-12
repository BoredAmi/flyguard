import numpy as np

import pytest

# Skip the whole module rather than failing collection: an optional
# dependency missing must not abort the entire test run.
pytest.importorskip("mujoco", reason="the tau baseline runs on MuJoCo-rendered frames")

from flyguard.baseline_tau import (
    apparent_radius_from_area,
    lee_tau,
    smooth,
    true_time_to_contact,
)


def test_apparent_radius_from_area_matches_circle_formula():
    area = np.array([0.0, np.pi, 4 * np.pi, 9 * np.pi])
    radius = apparent_radius_from_area(area)
    np.testing.assert_allclose(radius, [0.0, 1.0, 2.0, 3.0])


def test_apparent_radius_from_area_never_negative_or_nan():
    area = np.array([-5.0, 0.0, 100.0])
    radius = apparent_radius_from_area(area)
    assert np.all(radius >= 0)
    assert np.all(np.isfinite(radius))


def test_lee_tau_matches_analytic_constant_velocity_case():
    """For radius(t) = r0 + v*t (linear growth, as constant-velocity
    approach gives to first order over a short window), tau(t) should equal
    the exact analytic (r0 + v*t) / v."""
    dt = 0.1
    n = 20
    r0, v = 2.0, 0.5
    t = np.arange(n) * dt
    radius = r0 + v * t
    tau = lee_tau(radius, dt)
    expected = radius / v
    # np.gradient's edge handling is one-sided and slightly less accurate there;
    # check the interior tightly and the edges loosely.
    np.testing.assert_allclose(tau[2:-2], expected[2:-2], rtol=0.02)


def test_lee_tau_is_inf_for_constant_radius():
    """No expansion -> no finite time-to-contact by this cue."""
    radius = np.full(20, 5.0)
    tau = lee_tau(radius, dt=0.1)
    assert np.all(np.isinf(tau))


def test_lee_tau_is_inf_while_shrinking():
    radius = np.linspace(10.0, 1.0, 20)  # receding, not approaching
    tau = lee_tau(radius, dt=0.1)
    assert np.all(np.isinf(tau))


def test_lee_tau_decreases_toward_zero_for_realistic_approach():
    """A genuinely accelerating angular-size growth (1/(D0 - v*t) profile,
    matching real constant-velocity 3-D approach where angular size isn't
    linear in t) should give a tau that decreases over the trial, tracking
    time actually remaining."""
    dt = 0.05
    n = 30
    D0, v = 6.0, 2.0
    t = np.arange(n) * dt
    depth = D0 - v * t
    radius = 1.0 / depth  # angular size ~ 1/depth for a fixed physical size
    tau = lee_tau(radius, dt, smooth_window=3)
    finite = np.isfinite(tau)
    assert finite.sum() > n * 0.8
    # tau should be decreasing (looser than strict monotonic, allow small noise)
    assert tau[finite][-1] < tau[finite][0]


def test_smooth_is_noop_for_window_one():
    x = np.array([1.0, 5.0, 2.0, 9.0])
    np.testing.assert_array_equal(smooth(x, 1), x)


def test_smooth_preserves_length_and_reduces_noise():
    rng = np.random.default_rng(0)
    x = np.full(50, 10.0) + rng.normal(0, 1.0, 50)
    smoothed = smooth(x, 5)
    assert len(smoothed) == len(x)
    assert smoothed.std() < x.std()


def test_true_time_to_contact_is_linear_and_matches_trajectory_clip():
    """Ground truth should hit exactly zero when depth reaches the same
    1.5x-radius clip `mujoco_world.trajectory` uses to stop the approach."""
    dt, n = 0.1, 10
    depth0, speed, radius = 5.0, 1.0, 0.3
    tau = true_time_to_contact(depth0, speed, radius, dt, n)
    # linear, slope -1 (tau decreases 1 second of TTC per second elapsed)
    diffs = np.diff(tau)
    np.testing.assert_allclose(diffs, -dt, atol=1e-9)
    contact_time = (depth0 - radius * 1.5) / speed
    np.testing.assert_allclose(tau[0], contact_time)
