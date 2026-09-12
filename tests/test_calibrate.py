"""Tests for the camera calibration procedure.

The statistics are the whole of the risk here. Every constant `measure`
produces is a threshold, every one of them looks plausible when it is wrong,
and this project has already shipped two calibrations that were arithmetically
fine and behaviourally useless (the saccade threshold measured on obstacle
footage, an order of magnitude too high; the escape threshold measured on
clear footage, firing on 87% of ticks). So these tests check *which recording
each constant comes from*, not merely that a number comes out.

A stub encoder stands in for `BilateralEncoder` so the suite needs neither
the FlyWire CSVs nor a real flow computation.
"""

import numpy as np
import pytest

from flyguard.calibrate import _warn_on_weak_separation, load_frames, measure


class _StubEncoder:
    """Maps a frame's mean brightness onto a per-side drive.

    Deterministic and trivially controllable, so a test can say "these frames
    look like open space and those look like an obstacle on the right" without
    rendering anything.
    """

    def __init__(self, bias_right=0.0):
        self.bias_right = bias_right

    def settings(self):
        return {"mirror": "anatomical", "row_band": [0.0, 0.6]}

    def __call__(self, prev, frame):
        level = float(np.mean(frame)) / 255.0
        return {
            "drive_left": level,
            "drive_right": level + self.bias_right * level,
            "speed_left": level,
            "speed_right": level,
        }


def _frames(level, n=12, shape=(8, 8)):
    return np.stack([np.full(shape, level, np.uint8) for _ in range(n)])


def _measure(clear_level=40, obstacle_level=120, **kw):
    encoder = kw.pop("encoder", _StubEncoder())
    return measure(_frames(clear_level), _frames(obstacle_level), encoder,
                   backend="flow", tick_hz=10.0, **kw)


def test_drive_centre_comes_from_the_clear_recording():
    """Centring on obstacle footage puts ordinary cruising near drive 0, where
    both pools fall nearly silent and the steering ratio becomes two small
    noisy numbers divided by each other."""
    cal, _ = _measure(clear_level=40, obstacle_level=200)
    assert cal.drive_center["left"] == pytest.approx(40 / 255.0)


def test_drive_scale_comes_from_the_obstacle_recording():
    cal, _ = _measure(clear_level=40, obstacle_level=200)
    assert cal.drive_scale == pytest.approx((200 - 40) / 255.0, rel=1e-6)


def test_saccade_threshold_is_measured_on_clear_footage_only():
    """With nothing ahead, every turn signal is by construction noise -- so a
    clear recording with no left/right asymmetry must give a threshold at the
    noise floor, regardless of how asymmetric the obstacle footage is."""
    cal, _ = measure(_frames(40), _frames(200), _StubEncoder(bias_right=0.0),
                     backend="flow", tick_hz=10.0)
    assert cal.saccade_threshold == pytest.approx(0.0, abs=1e-6)


def test_a_resting_imbalance_is_absorbed_by_the_per_side_centre():
    """The two optic lobes are not reconstructed equally, so the pooled drives
    sit at measurably different resting levels. Two constants could in
    principle cancel that, and this pins down which one does: the *per-side*
    `drive_center` absorbs it, so a steady imbalance leaves `turn_offset` at
    zero rather than double-correcting it.

    That division of labour is why the gain is deliberately shared while the
    centre is not -- a per-side gain would rescale the hemispheres differently
    and distort the very left/right difference the steering law reads.
    """
    cal, report = measure(_frames(40), _frames(200), _StubEncoder(bias_right=0.5),
                          backend="flow", tick_hz=10.0)
    assert report["side_offset"] > 0.05, "the stub's imbalance should be visible"
    assert cal.drive_center["right"] > cal.drive_center["left"]
    assert cal.turn_offset == pytest.approx(0.0, abs=1e-9)


def test_escape_threshold_sits_above_the_obstacle_median():
    """Escape must mean "closer than almost any moment of ordinary travel",
    so a percentile of the obstacle distribution, never at or below its
    middle."""
    _, report = _measure()
    assert report["estop_threshold"] > report["estop_stat_median"]


def test_identical_recordings_are_refused_rather_than_calibrated():
    """If clear and obstacle footage produce the same drive, there is nothing
    to threshold and a silent zero scale would divide by ~0 downstream."""
    with pytest.raises(ValueError, match="drive scale is zero"):
        _measure(clear_level=80, obstacle_level=80)


def test_calibration_records_the_encoder_it_was_measured_with():
    cal, _ = _measure()
    assert cal.encoder == _StubEncoder().settings()
    assert cal.check_compatible({"row_band": [0.0, 1.0]})


def test_calibration_records_its_provenance():
    cal, _ = _measure()
    assert cal.source["backend"] == "flow"
    assert cal.source["n_clear_pairs"] == 11      # 12 frames -> 11 pairs
    assert cal.source["frame_shape"] == [8, 8]


def test_a_single_frame_cannot_be_calibrated():
    with pytest.raises(ValueError, match="at least 2 frames"):
        measure(_frames(40, n=1), _frames(200), _StubEncoder(), backend="flow")


# --- the warnings, which are the part a user actually acts on --------------


def test_warns_when_the_escape_statistic_does_not_separate():
    """This is the DNp01 saturation, and it is the expected outcome for the
    connectome backend rather than a user error -- so it must be reported
    clearly rather than buried in a threshold that merely looks odd."""
    warnings = _warn_on_weak_separation({
        "estop_threshold": 88.0, "clear_stat_median": 87.0,
        "estop_stat_median": 86.0, "saccade_threshold": 0.07,
    })
    assert any("does not separate" in w for w in warnings)
    assert any("Giant Fiber" in w for w in warnings)


def test_warns_when_escape_would_fire_constantly():
    warnings = _warn_on_weak_separation({
        "estop_threshold": 0.4, "clear_stat_median": 0.5,
        "estop_stat_median": 0.9, "saccade_threshold": 0.07,
    })
    assert any("brake permanently" in w for w in warnings)


def test_warns_when_the_clear_recording_was_not_travelling_straight():
    warnings = _warn_on_weak_separation({
        "estop_threshold": 0.9, "clear_stat_median": 0.4,
        "estop_stat_median": 0.8, "saccade_threshold": 0.95,
    })
    assert any("travelling straight" in w for w in warnings)


def test_a_healthy_calibration_warns_about_nothing():
    assert _warn_on_weak_separation({
        "estop_threshold": 0.9, "clear_stat_median": 0.4,
        "estop_stat_median": 0.7, "saccade_threshold": 0.07,
    }) == []


# --- frame loading ---------------------------------------------------------


def test_load_frames_reads_npy_and_npz(tmp_path):
    frames = _frames(50, n=4)
    np.save(tmp_path / "a.npy", frames)
    np.savez(tmp_path / "b.npz", frames=frames)
    assert np.array_equal(load_frames(tmp_path / "a.npy"), frames)
    assert np.array_equal(load_frames(tmp_path / "b.npz"), frames)


def test_load_frames_reports_an_empty_directory(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no images"):
        load_frames(tmp_path / "empty")
