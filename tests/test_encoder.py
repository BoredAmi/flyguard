from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flyguard.encoder import (
    DIRECTION_UNIT_VECTOR,
    SUBTYPE_UNIT_VECTOR,
    encode_drive,
    encode_i_ext,
    expanding_flow_2d,
    load_column_assignment,
    poisson_stim_fn,
    translational_flow_2d,
)

# The real column_assignment.csv lives outside this repo (see
# CLAUDE.md "Actual Codex file schema" -- raw Codex CSVs are not committed,
# only the extracted .npz). Integration tests against it are skipped when
# it's not present, e.g. in CI.
REAL_CSV = Path(__file__).resolve().parents[2] / "flywire" / "column_assignment.csv"


def _fake_columns() -> pd.DataFrame:
    """Four synthetic T4/T5 columns, one per subtype, all at distinct
    positions -- enough to unit-test the encoding math without touching the
    real (uncommitted) CSV."""
    rows = [
        # Placed so each column's own radial-outward direction matches its
        # subtype's preferred vector: 'a' prefers -x, so it sits on the -x
        # side (outward there is -x); 'b' prefers +x and sits on +x; etc.
        {"root_id": 1, "hemisphere": "right", "type": "T4a", "subtype": "a", "x": -5.0, "y": 0.0},
        {"root_id": 2, "hemisphere": "right", "type": "T4b", "subtype": "b", "x": 5.0, "y": 0.0},
        {"root_id": 3, "hemisphere": "right", "type": "T5c", "subtype": "c", "x": 0.0, "y": 5.0},
        {"root_id": 4, "hemisphere": "right", "type": "T5d", "subtype": "d", "x": 0.0, "y": -5.0},
    ]
    df = pd.DataFrame(rows)
    vecs = np.stack(df["subtype"].map(SUBTYPE_UNIT_VECTOR).to_numpy())
    df["pref_vx"] = vecs[:, 0]
    df["pref_vy"] = vecs[:, 1]
    return df


def test_subtype_vectors_are_four_distinct_unit_vectors():
    assert len(DIRECTION_UNIT_VECTOR) == 4
    for v in DIRECTION_UNIT_VECTOR.values():
        assert np.linalg.norm(v) == pytest.approx(1.0)
    vectors = np.stack(list(DIRECTION_UNIT_VECTOR.values()))
    assert np.linalg.matrix_rank(vectors) == 2  # span both axes


def test_encode_drive_peaks_for_aligned_flow_zero_for_orthogonal():
    columns = _fake_columns()
    # Flow blowing in the -x direction matches subtype 'a' (front_to_back)
    # exactly, and is orthogonal to 'c'/'d' (up/down).
    vx = np.full(4, -1.0)
    vy = np.zeros(4)
    drive = encode_drive(columns, vx, vy)
    assert drive.loc[1] == pytest.approx(1.0)   # subtype a: perfectly aligned
    assert drive.loc[2] == pytest.approx(0.0)   # subtype b: opposite, rectified to 0
    assert drive.loc[3] == pytest.approx(0.0, abs=1e-9)   # subtype c: orthogonal
    assert drive.loc[4] == pytest.approx(0.0, abs=1e-9)   # subtype d: orthogonal


def test_expanding_flow_2d_points_radially_outward():
    x = np.array([1.0, -1.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, -1.0])
    vx, vy = expanding_flow_2d(x, y)
    np.testing.assert_allclose(vx, [1.0, -1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(vy, [0.0, 0.0, 1.0, -1.0], atol=1e-9)


def test_expanding_flow_2d_origin_is_zero_not_nan():
    vx, vy = expanding_flow_2d(np.array([0.0]), np.array([0.0]))
    assert vx[0] == 0.0 and vy[0] == 0.0


def test_expanding_flow_drives_all_four_subtypes_equally():
    """The core discrimination claim, now on real per-neuron geometry
    instead of the abstract ring: an expanding flow field drives every
    subtype's column with equal, positive drive (each column's local
    outward direction matches its own subtype's preferred direction by
    construction of the fake layout)."""
    columns = _fake_columns()
    vx, vy = expanding_flow_2d(columns["x"].to_numpy(), columns["y"].to_numpy())
    drive = encode_drive(columns, vx, vy)
    assert (drive > 0.99).all()


def test_translation_drives_at_most_half_the_subtypes():
    columns = _fake_columns()
    vx, vy = translational_flow_2d(columns["x"].to_numpy(), direction_deg=0.0)
    drive = encode_drive(columns, vx, vy)
    assert (drive > 0.5).sum() <= 2
    assert (drive < 1e-9).sum() >= 1


def test_encode_i_ext_aligns_to_subnetwork_ordering_and_scales_by_peak_weight():
    columns = _fake_columns()
    vx, vy = expanding_flow_2d(columns["x"].to_numpy(), columns["y"].to_numpy())
    subnetwork_ids = np.array([4, 99, 2, 1, 3])  # arbitrary order, includes an unknown id
    i_ext = encode_i_ext(columns, vx, vy, subnetwork_ids, peak_weight=30.0)
    assert i_ext.shape == (5,)
    assert i_ext[1] == 0.0  # root_id 99 not in columns -> zero drive
    assert i_ext[3] == pytest.approx(30.0)  # root_id 1, subtype a, fully aligned
    assert (i_ext >= 0).all()


def test_poisson_stim_fn_impulses_are_sparse_not_constant():
    """Regression test for the encoder methodology pitfall documented in
    CLAUDE.md: feeding a static drive-derived value into i_ext on every
    step saturates the network (peak_weight alone exceeds the spike
    threshold). `poisson_stim_fn` must instead emit impulses only on a
    minority of steps, even for a neuron with maximal drive."""
    columns = _fake_columns()
    vx, vy = expanding_flow_2d(columns["x"].to_numpy(), columns["y"].to_numpy())
    subnetwork_ids = np.array([1, 2, 3, 4])
    stim_fn = poisson_stim_fn(
        columns, vx, vy, subnetwork_ids, dt=1e-4, base_rate_hz=250.0, peak_weight=30.0, seed=0
    )
    hits = np.array([stim_fn(k)[0] > 0 for k in range(5000)])  # root_id 1, drive=1.0
    rate_fraction = hits.mean()
    # expected ~ drive * base_rate_hz * dt = 1.0 * 250 * 1e-4 = 0.025
    assert 0.01 < rate_fraction < 0.05, f"impulse fraction {rate_fraction:.3f} is not sparse"


def test_poisson_stim_fn_zero_drive_neuron_never_fires():
    columns = _fake_columns()
    # Translation straight along +x: subtypes c/d (pref vectors (0,+-1)) are
    # exactly orthogonal to this flow everywhere -> rectified drive = 0.
    x = columns["x"].to_numpy()
    vx, vy = translational_flow_2d(x, direction_deg=0.0)
    subnetwork_ids = np.array([1, 2, 3, 4])
    stim_fn = poisson_stim_fn(
        columns, vx, vy, subnetwork_ids, dt=1e-4, base_rate_hz=250.0, peak_weight=30.0, seed=0
    )
    # root_id 3 = subtype c (pref (0, 1)), orthogonal to vx=1, vy=0 -> never forced.
    hits = np.array([stim_fn(k)[2] > 0 for k in range(2000)])
    assert not hits.any()


def test_poisson_stim_fn_is_deterministic():
    columns = _fake_columns()
    vx, vy = expanding_flow_2d(columns["x"].to_numpy(), columns["y"].to_numpy())
    subnetwork_ids = np.array([1, 2, 3, 4])
    a = poisson_stim_fn(columns, vx, vy, subnetwork_ids, dt=1e-4, seed=5)
    b = poisson_stim_fn(columns, vx, vy, subnetwork_ids, dt=1e-4, seed=5)
    seq_a = np.array([a(k) for k in range(200)])
    seq_b = np.array([b(k) for k in range(200)])
    np.testing.assert_array_equal(seq_a, seq_b)


@pytest.mark.skipif(not REAL_CSV.exists(), reason="real column_assignment.csv not present (not committed)")
def test_real_column_assignment_loads_and_has_both_hemispheres():
    df = load_column_assignment(REAL_CSV)
    assert len(df) > 1000
    assert set(df["subtype"]) == {"a", "b", "c", "d"}
    assert set(df["hemisphere"]) >= {"left", "right"}
    assert np.isfinite(df[["pref_vx", "pref_vy"]].to_numpy()).all()


@pytest.mark.skipif(not REAL_CSV.exists(), reason="real column_assignment.csv not present (not committed)")
def test_real_column_assignment_side_filter():
    right = load_column_assignment(REAL_CSV, side="right")
    left = load_column_assignment(REAL_CSV, side="left")
    assert (right["hemisphere"] == "right").all()
    assert (left["hemisphere"] == "left").all()
    assert set(right["root_id"]).isdisjoint(set(left["root_id"]))
