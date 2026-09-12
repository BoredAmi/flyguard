"""Tests for `flyguard.runtime`, the robot-facing API.

Everything here runs on a synthetic connectome and synthetic frames, so the
whole file executes in about a second and needs neither MuJoCo, nor the
FlyWire CSVs, nor `data/looming.npz`. That matters more for this package than
for the research layer: it is the part someone integrating a robot will read
first and modify most.
"""

import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyguard.runtime import (
    Calibration,
    CoreCircuit,
    Percept,
    SaccadicSteering,
    bilateral_turn,
    ema_trace,
    normalize_drive,
)

# ---------------------------------------------------------------------------
# A synthetic core circuit, shaped like the real one but tiny
# ---------------------------------------------------------------------------


def _write_core_npz(path, *, with_lpi=True, sides=("left", "right")):
    """Write a subnetwork .npz in `flyguard.extract.load_subnetwork`'s format.

    Two LPLC2, two LC4, two LPi and one DNp01 per side. Real signs: LPLC2
    excites DNp01, LPi inhibits LPLC2.
    """
    cell_types, side_col = [], []
    for side in sides:
        spec = [("LPLC2", 2), ("LC4", 2), ("DNp01", 1)]
        if with_lpi:
            spec.append(("LPi01", 2))
        for name, n in spec:
            cell_types += [name] * n
            side_col += [side] * n
    n = len(cell_types)
    meta = pd.DataFrame({"cell_type": cell_types, "side": side_col})
    W = np.zeros((n, n), dtype=np.float32)
    lplc2 = np.flatnonzero(meta.cell_type == "LPLC2")
    dnp01 = np.flatnonzero(meta.cell_type == "DNp01")
    lpi = np.flatnonzero(meta.cell_type.str.startswith("LPi"))
    for i in lplc2:
        for j in dnp01:
            W[i, j] = 40.0
    for i in lpi:
        for j in lplc2:
            W[i, j] = -40.0
    Wsp = sp.csr_matrix(W)
    np.savez(
        path,
        data=Wsp.data, indices=Wsp.indices, indptr=Wsp.indptr, shape=Wsp.shape,
        root_id=np.arange(n, dtype=np.int64),
        cell_type=np.array(cell_types), side=np.array(side_col),
        is_seed=np.ones(n, dtype=bool),
    )
    return path


@pytest.fixture
def core_npz(tmp_path):
    return _write_core_npz(tmp_path / "core.npz")


# ---------------------------------------------------------------------------
# CoreCircuit
# ---------------------------------------------------------------------------


def test_core_circuit_counts_every_population(core_npz):
    c = CoreCircuit(core_npz, tick_hz=100.0)
    counts = c.counts()
    assert counts == {"LPLC2": 4, "LC4": 4, "LPi": 4, "DNp01": 2, "total": 14}


def test_core_circuit_refuses_to_build_without_inhibition(tmp_path):
    """LPi is load-bearing, not an optimisation.

    A circuit of LPLC2 + LC4 + DNp01 alone has no inhibition at all, and its
    real mutual excitation latches into runaway activity that never releases
    -- the same failure mode `validate_ablation.py` measures deliberately as
    the "LPi ablated" condition. The ROS2 node once shipped exactly this by
    omission, so it must fail loudly rather than silently.
    """
    path = _write_core_npz(tmp_path / "no_lpi.npz", with_lpi=False)
    with pytest.raises(RuntimeError, match="no LPi"):
        CoreCircuit(path)


def test_core_circuit_reports_a_missing_snapshot_with_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="flyguard.extract"):
        CoreCircuit(tmp_path / "absent.npz")


def test_monocular_circuit_rejects_a_per_side_drive(core_npz):
    c = CoreCircuit(core_npz, tick_hz=100.0)
    with pytest.raises(TypeError, match="bilateral=True"):
        c.step({"left": 0.1, "right": 0.9})


def test_bilateral_combined_count_equals_the_two_sides(core_npz):
    """The escape readout asks for DNp01 as one channel; the Giant Fiber is
    a single command pathway, not a left one and a right one."""
    c = CoreCircuit(core_npz, tick_hz=100.0, bilateral=True)
    r = c.step({"left": 0.9, "right": 0.9})
    assert r.spike_count("DNp01") == (
        r.spike_count("DNp01", "left") + r.spike_count("DNp01", "right")
    )


def test_unrecorded_population_names_itself_in_the_error(core_npz):
    c = CoreCircuit(core_npz, tick_hz=100.0, record=("LPLC2",))
    with pytest.raises(KeyError, match="DNp01"):
        c.step(0.5).spike_count("DNp01")


def test_reset_makes_the_circuit_reproducible(core_npz):
    c = CoreCircuit(core_npz, tick_hz=100.0, seed=7)
    first = [c.step(0.8).spikes.copy() for _ in range(3)]
    c.reset()
    second = [c.step(0.8).spikes.copy() for _ in range(3)]
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_drive_reaches_the_circuit_monotonically(core_npz):
    """Sanity that the dual-channel drive is wired the right way round: more
    drive must mean more LPLC2 activity, not less."""
    rates = []
    for drive in (0.0, 0.5, 1.0):
        c = CoreCircuit(core_npz, tick_hz=20.0, seed=3)
        rates.append(np.mean([c.step(drive).rate_hz("LPLC2") for _ in range(5)]))
    assert rates[0] < rates[-1], rates


def test_recording_more_populations_widens_the_spike_array(core_npz):
    narrow = CoreCircuit(core_npz, tick_hz=100.0, record=("LPLC2",))
    wide = CoreCircuit(core_npz, tick_hz=100.0, record=("LPLC2", "LPi", "DNp01"))
    assert wide.step(0.5).spikes.shape[1] > narrow.step(0.5).spikes.shape[1]
    assert len(wide.record_index) == wide.step(0.5).spikes.shape[1]


# ---------------------------------------------------------------------------
# Percept
# ---------------------------------------------------------------------------


def test_percept_is_usable_as_the_dict_it_replaced():
    p = Percept(drive_left=0.1, drive_right=0.2, speed_left=1.0, speed_right=2.0)
    assert p["drive_left"] == p.drive_left == 0.1
    assert dict(p.items())["speed_right"] == 2.0
    assert Percept.from_dict(p.as_dict()) == p


def test_percept_from_dict_tolerates_missing_speeds():
    p = Percept.from_dict({"drive_left": 0.3, "drive_right": 0.4})
    assert p.speed_left == 0.0


# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------


def test_bilateral_turn_sign_means_turn_away_from_the_louder_side():
    """Positive omega is left, so a louder *right* hemisphere must give a
    positive turn. Getting this backwards steers into obstacles, and no gain
    tuning fixes a sign error."""
    assert bilateral_turn(0.2, 0.9) > 0
    assert bilateral_turn(0.9, 0.2) < 0
    assert bilateral_turn(0.5, 0.5) == 0.0


def test_bilateral_turn_is_silent_when_both_sides_are():
    assert bilateral_turn(0.0, 0.0) == 0.0


def test_normalize_drive_centres_on_the_resting_level():
    assert normalize_drive(0.1, center=0.1, scale=0.05) == pytest.approx(0.5)
    assert normalize_drive(0.15, center=0.1, scale=0.05) == pytest.approx(1.0)
    assert normalize_drive(0.05, center=0.1, scale=0.05) == pytest.approx(0.0)


def test_normalize_drive_clips_rather_than_running_away():
    assert normalize_drive(99.0, center=0.1, scale=0.05) == 1.0
    assert normalize_drive(-99.0, center=0.1, scale=0.05) == 0.0


def _steering(**kw):
    params = dict(saccade_threshold=0.1, estop_threshold=10.0,
                  saccade_duration_s=0.3, refractory_s=0.3, estop_cooldown_s=0.4)
    params.update(kw)
    return SaccadicSteering(**params)


def test_no_saccade_below_threshold():
    s = _steering()
    assert s.step(0.05, 0.0, 0.0).omega == 0.0


def test_saccade_commits_for_its_full_duration():
    s = _steering()
    first = s.step(0.5, 0.0, 0.0)
    assert first.omega > 0
    # Still turning mid-saccade even though the evidence has vanished.
    assert s.step(0.0, 0.0, 0.1).omega == first.omega


def test_vision_is_suppressed_through_the_settling_window():
    """Suppressing only *during* the turn is not enough. The flow estimate is
    computed between consecutive frames, so the tick after a saccade is still
    contaminated; that leftover fired the escape, which fired the next
    saccade, and the agent oscillated between -37 and +46 degrees."""
    s = _steering()
    s.step(0.5, 0.0, 0.0)                     # saccade until t=0.3, refractory to 0.6
    during = s.step(0.0, 1e6, 0.15)           # huge escape evidence, mid-saccade
    settling = s.step(0.0, 1e6, 0.45)         # huge escape evidence, settling
    assert not during.estop
    assert not settling.estop
    assert s.step(0.0, 1e6, 0.7).estop        # vision counts again


def test_escape_brakes_without_seizing_the_steering():
    """An earlier version overrode omega with a full-rate turn on escape and
    drove the agent into the nearest wall."""
    s = _steering()
    cmd = s.step(0.0, 1e6, 0.0)
    assert cmd.estop
    assert cmd.v == s.v_escape
    # It does commit a turn, using whatever weak side evidence exists to pick
    # a direction -- but at the ordinary saccade rate, not a special one.
    assert abs(cmd.omega) == pytest.approx(s.saccade_rate)


def test_turn_offset_is_subtracted_before_the_threshold():
    s = _steering(turn_offset=0.5)
    # 0.52 raw is only 0.02 once the resting imbalance is nulled: no saccade.
    assert s.step(0.52, 0.0, 0.0).omega == 0.0


def test_reset_clears_saccade_state():
    s = _steering()
    s.step(0.5, 0.0, 0.0)
    s.reset()
    assert s.step(0.0, 0.0, 0.05).omega == 0.0


def test_cruise_is_issued_before_any_percept_exists():
    s = _steering()
    cmd = s.cruise()
    assert cmd.omega == 0.0 and not cmd.estop and cmd.v == s.v_cruise


# ---------------------------------------------------------------------------
# turn_ema: the sustained-weak-bias trigger added after diagnose_trial found
# a wall struck by a signal that stayed correctly signed for 7+ seconds but
# never once crossed saccade_threshold on a single tick.
# ---------------------------------------------------------------------------


def test_ema_trace_matches_the_textbook_recurrence():
    out = ema_trace([1.0, 1.0, 1.0], alpha=0.5)
    assert out == pytest.approx([0.5, 0.75, 0.875])


def test_ema_trace_starts_from_zero_each_call():
    """Pure and stateless: calling it twice must not remember the first call,
    the same guarantee `calibrate_controller` relies on per trial."""
    first = ema_trace([1.0, 1.0], alpha=0.5)
    second = ema_trace([1.0, 1.0], alpha=0.5)
    assert first == second


def test_weak_persistent_bias_triggers_via_ema_when_instant_never_fires():
    """The exact shape diagnose_trial found: every single tick under
    saccade_threshold, but the same sign for long enough that the smoothed
    signal isn't."""
    s = _steering(saccade_threshold=0.5, turn_ema_alpha=0.2, turn_ema_threshold=0.05)
    omega = 0.0
    for i in range(40):
        cmd = s.step(0.08, 0.0, i * 0.1)
        if cmd.omega != 0.0:
            omega = cmd.omega
            break
    assert omega != 0.0, "a sustained sub-threshold bias should eventually fire a saccade"


def test_symmetric_noise_never_accumulates_in_the_ema():
    """A sign-alternating signal is what the instantaneous noise floor is
    calibrated against; the EMA must not be more trigger-happy than that on
    the same kind of noise."""
    s = _steering(saccade_threshold=0.5, turn_ema_alpha=0.2, turn_ema_threshold=0.05)
    for i in range(40):
        turn = 0.08 if i % 2 == 0 else -0.08
        cmd = s.step(turn, 0.0, i * 0.1)
        assert cmd.omega == 0.0


def test_ema_disabled_by_default_does_not_change_existing_behaviour():
    """turn_ema_threshold defaults to inf, same convention as the other two
    calibrated thresholds -- an uncalibrated controller must behave exactly
    as it did before this was added."""
    s = _steering(saccade_threshold=0.5)
    for i in range(60):
        assert s.step(0.08, 0.0, i * 0.1).omega == 0.0


def test_ema_resets_after_it_fires_a_saccade():
    s = _steering(saccade_threshold=0.5, turn_ema_alpha=0.5, turn_ema_threshold=0.05)
    t = 0.0
    fired = False
    for i in range(20):
        cmd = s.step(0.3, 0.0, t)
        t += 0.1
        if cmd.omega != 0.0 and not fired:
            fired = True
            # The tick right after a saccade completes and clears refractory
            # should not immediately refire on stale accumulated evidence.
            t = s._refractory_until + 0.01
            post = s.step(0.0, 0.0, t)
            assert post.telemetry["turn_ema"] == pytest.approx(0.0)
            break
    assert fired


def test_ema_only_accumulates_while_vision_is_valid():
    """Rotational flow during a saccade or its settling window is exactly
    the contamination the instantaneous check is already gated against;
    the running average must not ingest it either."""
    s = _steering(saccade_threshold=0.5, turn_ema_alpha=0.9, turn_ema_threshold=0.05)
    s.step(0.3, 0.0, 0.0)                       # commits a saccade
    during = s.step(0.9, 0.0, 0.1)               # huge, but mid-saccade
    assert during.telemetry["turn_ema"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_calibration_round_trips_through_json(tmp_path):
    c = Calibration(drive_center={"left": 0.1, "right": 0.12}, drive_scale=0.05,
                    turn_offset=0.058, saccade_threshold=0.13,
                    estop_threshold=88.0, encoder={"mirror": "anatomical"},
                    source={"backend": "connectome"})
    path = c.save(tmp_path / "cal.json")
    assert Calibration.load(path) == c


def test_calibration_round_trips_infinite_thresholds(tmp_path):
    """`inf` is not valid JSON, so it is stored as a sentinel. A default
    calibration means "never turn, never brake" and must survive a save."""
    c = Calibration()
    path = c.save(tmp_path / "default.json")
    assert json.loads(path.read_text())["saccade_threshold"] == "inf"
    assert Calibration.load(path).saccade_threshold == float("inf")


def test_default_calibration_is_not_measured():
    assert not Calibration().is_measured()
    assert Calibration(drive_scale=0.05, saccade_threshold=0.1).is_measured()


def test_calibration_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown calibration fields"):
        Calibration.from_dict({"drive_scale": 0.1, "gain": 3.0})


def test_calibration_rejects_a_newer_schema():
    with pytest.raises(ValueError, match="newer"):
        Calibration.from_dict({"schema_version": 99})


def test_calibration_flags_an_encoder_mismatch():
    """A calibration measured with the ground masked out does not transfer to
    one that uses the whole frame, and the mismatch is otherwise silent."""
    c = Calibration(encoder={"row_band": [0.0, 0.6], "mirror": "anatomical"})
    assert c.check_compatible({"row_band": [0.0, 0.6], "mirror": "anatomical"}) == []
    warnings = c.check_compatible({"row_band": [0.0, 1.0], "mirror": "anatomical"})
    assert len(warnings) == 1 and "row_band" in warnings[0]


def test_calibration_missing_file_points_at_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="flyguard.calibrate"):
        Calibration.load(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# sensor_msgs/Image decoding (no ROS2 needed -- the decoder duck-types)
# ---------------------------------------------------------------------------


class _FakeImage:
    def __init__(self, array, encoding, step=None):
        self.height, self.width = array.shape[:2]
        channels = 1 if array.ndim == 2 else array.shape[2]
        self.encoding = encoding
        self.step = step if step is not None else self.width * channels
        row_bytes = array.reshape(self.height, -1)
        pad = self.step - row_bytes.shape[1]
        if pad:
            row_bytes = np.hstack([row_bytes, np.zeros((self.height, pad), np.uint8)])
        self.data = row_bytes.astype(np.uint8).tobytes()


def test_image_decode_round_trips_mono_and_colour():
    from flyguard.runtime import image_to_array

    mono = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert np.array_equal(image_to_array(_FakeImage(mono, "mono8")), mono)

    rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    assert np.array_equal(image_to_array(_FakeImage(rgb, "rgb8")), rgb)


def test_image_decode_honours_a_padded_row_stride():
    """A padded stride read as tightly packed shears the image diagonally,
    which reads downstream as a broken detector rather than a broken decode."""
    from flyguard.runtime import image_to_array

    rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    padded = _FakeImage(rgb, "rgb8", step=4 * 3 + 5)
    assert np.array_equal(image_to_array(padded), rgb)


def test_image_decode_drops_the_alpha_channel():
    from flyguard.runtime import image_to_array

    rgba = np.arange(48, dtype=np.uint8).reshape(3, 4, 4)
    assert image_to_array(_FakeImage(rgba, "rgba8")).shape == (3, 4, 3)


def test_image_decode_names_the_encoding_it_cannot_handle():
    from flyguard.runtime import image_to_array

    with pytest.raises(ValueError, match="16UC1"):
        image_to_array(_FakeImage(np.zeros((2, 2), np.uint8), "16UC1"))


def test_image_decode_rejects_a_truncated_buffer():
    from flyguard.runtime import image_to_array

    msg = _FakeImage(np.zeros((3, 4), np.uint8), "mono8")
    msg.height = 9   # claim more rows than the buffer holds
    with pytest.raises(ValueError, match="truncated"):
        image_to_array(msg)


def test_decoded_mono_survives_grayscale_conversion():
    """Regression: `to_grayscale` averaged over the last axis unconditionally,
    which collapsed an already-2-D mono frame to 1-D and surfaced much later
    as an opaque pyramid-slicing IndexError."""
    from flyguard.optical_flow import to_grayscale
    from flyguard.runtime import image_to_array

    mono = image_to_array(_FakeImage(np.full((8, 8), 128, np.uint8), "mono8"))
    assert to_grayscale(mono).shape == (8, 8)
