import numpy as np
import pandas as pd
import pytest

from flyguard.encoder import (
    column_pixel_bounds,
    column_to_pixel,
    flow_from_frame_pair,
    sample_flow_at_columns,
)


def _fake_columns():
    return pd.DataFrame({
        "root_id": [1, 2, 3, 4],
        "x": [-9.0, 8.0, 0.0, -9.0],
        "y": [-29.0, 30.0, 0.0, 30.0],
    })


def test_column_pixel_bounds_matches_min_max():
    cols = _fake_columns()
    x_bounds, y_bounds = column_pixel_bounds(cols)
    assert x_bounds == (-9.0, 8.0)
    assert y_bounds == (-29.0, 30.0)


def test_column_to_pixel_corners_map_to_image_corners():
    x_bounds, y_bounds = (-9.0, 8.0), (-29.0, 30.0)
    resolution = 100
    # min x, min y -> bottom-left in column space -> (col=0, row=max) since +y is up
    px, py = column_to_pixel(np.array([-9.0]), np.array([-29.0]), resolution, x_bounds, y_bounds)
    assert px[0] == pytest.approx(0.0)
    assert py[0] == pytest.approx(resolution - 1)
    # max x, max y -> top-right -> (col=max, row=0)
    px, py = column_to_pixel(np.array([8.0]), np.array([30.0]), resolution, x_bounds, y_bounds)
    assert px[0] == pytest.approx(resolution - 1)
    assert py[0] == pytest.approx(0.0)


def test_column_to_pixel_center_maps_to_image_center():
    x_bounds, y_bounds = (-10.0, 10.0), (-10.0, 10.0)
    resolution = 101
    px, py = column_to_pixel(np.array([0.0]), np.array([0.0]), resolution, x_bounds, y_bounds)
    assert px[0] == pytest.approx(50.0)
    assert py[0] == pytest.approx(50.0)


def test_sample_flow_at_columns_picks_up_known_field_values():
    resolution = 20
    x_bounds, y_bounds = (0.0, 1.0), (0.0, 1.0)
    # a flow field that's a simple linear ramp in x, constant in y
    vx_img = np.tile(np.linspace(0, 1, resolution), (resolution, 1))
    vy_img = np.zeros((resolution, resolution))

    cols = pd.DataFrame({"root_id": [1, 2], "x": [0.0, 1.0], "y": [0.5, 0.5]})
    flow_vx, flow_vy = sample_flow_at_columns(vx_img, vy_img, cols, x_bounds, y_bounds)
    assert flow_vx[0] == pytest.approx(0.0, abs=0.05)
    assert flow_vx[1] == pytest.approx(1.0, abs=0.05)
    np.testing.assert_allclose(flow_vy, 0.0, atol=1e-9)


def test_flow_from_frame_pair_returns_arrays_aligned_with_columns():
    rng = np.random.default_rng(0)
    resolution = 32
    frame1 = rng.integers(0, 256, size=(resolution, resolution, 3), dtype=np.uint8)
    frame2 = frame1.copy()  # identical frames -> zero flow everywhere
    cols = pd.DataFrame({
        "root_id": [1, 2, 3],
        "x": [-1.0, 0.0, 1.0],
        "y": [-1.0, 0.0, 1.0],
    })
    flow_vx, flow_vy = flow_from_frame_pair(frame1, frame2, cols)
    assert flow_vx.shape == (3,)
    assert flow_vy.shape == (3,)
    np.testing.assert_allclose(flow_vx, 0.0, atol=1e-6)
    np.testing.assert_allclose(flow_vy, 0.0, atol=1e-6)


def test_flow_from_frame_pair_detects_real_shift():
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(1)
    resolution, pad = 48, 8
    base = gaussian_filter(rng.random((resolution + 2 * pad, resolution + 2 * pad)), sigma=2.0)
    img1 = (base[pad:pad + resolution, pad:pad + resolution] * 255).astype(np.uint8)
    img2 = (base[pad:pad + resolution, pad - 3:pad - 3 + resolution] * 255).astype(np.uint8)
    frame1 = np.stack([img1] * 3, axis=-1)
    frame2 = np.stack([img2] * 3, axis=-1)

    cols = pd.DataFrame({"root_id": [1], "x": [0.0], "y": [0.0]})
    x_bounds, y_bounds = (-1.0, 1.0), (-1.0, 1.0)
    flow_vx, flow_vy = flow_from_frame_pair(frame1, frame2, cols, x_bounds, y_bounds)
    assert flow_vx[0] > 1.5, f"expected to detect rightward shift, got vx={flow_vx[0]:.2f}"
