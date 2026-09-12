import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
mujoco = pytest.importorskip("mujoco", reason="mujoco is an optional extra (pip install flyguard[mujoco])")

from flyguard.mujoco_world import (  # noqa: E402
    SCENE_XML,
    TrialParams,
    apparent_area_px,
    generate_dataset,
    make_trial_params,
    matched_depth_trial_params,
    render_trial,
    trajectory,
)


def _make_renderer():
    """Skip (rather than fail) if a MuJoCo model loads but headless
    rendering itself isn't available in this environment (no EGL/GPU,
    e.g. some CI runners)."""
    try:
        model = mujoco.MjModel.from_xml_string(SCENE_XML)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=32, width=32)
        return model, data, renderer
    except Exception as e:  # noqa: BLE001 - genuinely any backend failure means "skip"
        pytest.skip(f"headless MuJoCo rendering unavailable: {e}")


def test_trajectory_looming_moves_toward_camera():
    p = TrialParams(condition="looming", n_frames=10, dt=1 / 30, depth0=6.0, radius=0.3, speed=2.0)
    pos = trajectory(p)
    assert pos.shape == (10, 2)
    assert np.all(np.diff(pos[:, 0]) <= 0)  # x (depth) strictly decreases
    assert pos[:, 0].min() >= p.radius * 1.5  # never clips through the camera


def test_trajectory_translation_moves_laterally_at_constant_depth():
    p = TrialParams(condition="translation", n_frames=10, dt=1 / 30, depth0=4.0, radius=0.3, speed=2.0)
    pos = trajectory(p)
    assert np.all(pos[:, 0] == p.depth0)  # depth constant
    assert not np.all(pos[:, 1] == pos[0, 1])  # y actually moves


def test_trajectory_rejects_unknown_condition():
    p = TrialParams(condition="bogus", n_frames=5, dt=1 / 30, depth0=1.0, radius=0.3, speed=1.0)
    with pytest.raises(ValueError):
        trajectory(p)


def test_make_trial_params_deterministic_given_seed():
    a = make_trial_params("looming", seed=42)
    b = make_trial_params("looming", seed=42)
    assert a == b


def test_matched_depth_trial_params_same_depth_both_conditions():
    """Regression test for the real confound found while building the
    image-based encoder: make_trial_params draws looming/translation from
    different depth ranges, inflating translation's apparent object size
    and reversing discrimination in a size-sensitive drive scheme (see
    the project notes "Pilot findings"). matched_depth_trial_params must give both
    conditions the identical starting depth."""
    loom = matched_depth_trial_params("looming", seed=1, depth0=5.0)
    trans = matched_depth_trial_params("translation", seed=1, depth0=5.0)
    assert loom.depth0 == 5.0
    assert trans.depth0 == 5.0
    assert loom.radius == trans.radius


def test_matched_depth_trial_params_rejects_unknown_condition():
    with pytest.raises(ValueError):
        matched_depth_trial_params("bogus", seed=1)


def test_matched_depth_trial_params_deterministic_given_seed():
    a = matched_depth_trial_params("translation", seed=7)
    b = matched_depth_trial_params("translation", seed=7)
    assert a == b


def test_render_trial_looming_expands_monotonically_in_pixels():
    model, data, renderer = _make_renderer()
    try:
        p = TrialParams(condition="looming", n_frames=20, dt=1 / 30, depth0=6.0, radius=0.3, speed=2.0)
        frames = render_trial(model, renderer, data, p)
        assert frames.shape == (20, 32, 32, 3)
        assert frames.dtype == np.uint8
        sizes = apparent_area_px(frames)
        assert sizes[-1] > 1.5 * sizes[0], "disc should visibly grow over the trial"
        assert np.all(np.diff(sizes) >= -2), "apparent size should not shrink (looming)"
    finally:
        renderer.close()


def test_render_trial_translation_stays_roughly_constant_size():
    model, data, renderer = _make_renderer()
    try:
        p = TrialParams(condition="translation", n_frames=20, dt=1 / 30, depth0=4.0, radius=0.3, speed=2.0)
        frames = render_trial(model, renderer, data, p)
        sizes = apparent_area_px(frames)
        assert sizes.std() / sizes.mean() < 0.05, "apparent size should stay roughly flat (translation)"
    finally:
        renderer.close()


def test_generate_dataset_end_to_end(tmp_path):
    out = tmp_path / "dataset.npz"
    generate_dataset(out, n_trials_per_condition=2, n_frames=15, resolution=32, seed=0)
    z = np.load(out)
    assert z["frames"].shape == (4, 15, 32, 32, 3)
    assert set(z["condition"]) == {"looming", "translation"}
    assert (z["condition"] == "looming").sum() == 2
    assert (z["condition"] == "translation").sum() == 2

    for i in range(z["frames"].shape[0]):
        sizes = apparent_area_px(z["frames"][i])
        if str(z["condition"][i]) == "looming":
            assert sizes[-1] > sizes[0]
        else:
            assert sizes.std() / sizes.mean() < 0.05


def test_generate_dataset_is_deterministic_given_seed(tmp_path):
    out_a = tmp_path / "a.npz"
    out_b = tmp_path / "b.npz"
    generate_dataset(out_a, n_trials_per_condition=1, n_frames=10, resolution=24, seed=7)
    generate_dataset(out_b, n_trials_per_condition=1, n_frames=10, resolution=24, seed=7)
    za, zb = np.load(out_a), np.load(out_b)
    np.testing.assert_array_equal(za["frames"], zb["frames"])
    np.testing.assert_array_equal(za["params"], zb["params"])
