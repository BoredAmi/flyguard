import math
import os

import numpy as np
import pytest

# Skip the whole module rather than failing collection: an optional
# dependency missing must not abort the entire test run. This has to sit
# above the flyguard.arena import, which pulls in mujoco itself.
mujoco = pytest.importorskip("mujoco", reason="mujoco is an optional extra")

from flyguard.arena import (
    AgentState,
    collision_kind,
    ArenaConfig,
    build_arena_xml,
    clearance,
    collides,
    sample_obstacles,
    step_agent,
)


def test_sample_obstacles_is_deterministic_for_a_seed():
    a = sample_obstacles(ArenaConfig(seed=3))
    b = sample_obstacles(ArenaConfig(seed=3))
    c = sample_obstacles(ArenaConfig(seed=4))
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_obstacles_sit_on_the_x_grid_and_inside_the_jitter_band():
    cfg = ArenaConfig(seed=7)
    obs = sample_obstacles(cfg)
    assert len(obs) == cfg.n_obstacles
    expected_x = cfg.first_obstacle_x + np.arange(cfg.n_obstacles) * cfg.obstacle_spacing
    np.testing.assert_allclose(obs[:, 0], expected_x)
    assert np.all(np.abs(obs[:, 1]) <= cfg.y_jitter)


def test_every_arena_leaves_a_passable_gap():
    """The geometry claim in ArenaConfig's docstring, checked rather than
    asserted in prose: an obstacle can never squeeze the corridor below the
    agent's width."""
    cfg = ArenaConfig()
    widest_reach = cfg.y_jitter + cfg.obstacle_radius
    gap = cfg.lane_half_width - widest_reach
    assert gap - cfg.agent_radius > 0.5


def test_n_obstacles_zero_gives_an_empty_arena_not_a_negative_corridor():
    cfg = ArenaConfig(n_obstacles=0)
    assert len(sample_obstacles(cfg)) == 0
    assert cfg.goal_x > cfg.first_obstacle_x


def test_corridor_length_override_wins():
    """Calibration builds empty corridors at the *measured* arena's length;
    without the override `n_obstacles=0` would shrink the world."""
    cfg = ArenaConfig(n_obstacles=0, corridor_length=25.5)
    assert cfg.goal_x == pytest.approx(25.5)


def test_step_agent_is_unicycle_kinematics():
    s = step_agent(AgentState(0.0, 0.0, 0.0), v=2.0, omega=0.0, dt=0.5)
    assert (s.x, s.y) == pytest.approx((1.0, 0.0))
    s = step_agent(AgentState(0.0, 0.0, math.pi / 2), v=2.0, omega=0.0, dt=0.5)
    assert (s.x, s.y) == pytest.approx((0.0, 1.0), abs=1e-9)


def test_positive_omega_turns_toward_positive_y():
    """The sign convention the whole steering law depends on."""
    s = step_agent(AgentState(0.0, 0.0, 0.0), v=1.0, omega=1.0, dt=0.1)
    assert s.heading > 0
    s2 = step_agent(s, v=1.0, omega=0.0, dt=0.5)
    assert s2.y > s.y


def test_collides_detects_obstacles_and_walls():
    cfg = ArenaConfig(obstacle_radius=0.5, agent_radius=0.3, lane_half_width=4.0)
    obstacles = np.array([[5.0, 0.0]])
    assert collides(AgentState(5.0, 0.0, 0.0), obstacles, cfg)
    assert collides(AgentState(5.0, 0.75, 0.0), obstacles, cfg)   # 0.75 < 0.5+0.3
    assert not collides(AgentState(5.0, 0.9, 0.0), obstacles, cfg)
    # walls at |y| = 4.0, agent radius 0.3
    assert collides(AgentState(5.0, 3.8, 0.0), obstacles, cfg)
    assert not collides(AgentState(5.0, 3.5, 0.0), obstacles, cfg)


def test_clearance_is_negative_exactly_when_colliding():
    cfg = ArenaConfig()
    obstacles = sample_obstacles(cfg)
    for state in (AgentState(0.0, 0.0, 0.0), AgentState(6.0, 0.0, 0.0),
                  AgentState(10.0, 3.9, 0.0)):
        assert (clearance(state, obstacles, cfg) <= 0.0) == collides(state, obstacles, cfg)


def test_clearance_handles_an_empty_arena():
    cfg = ArenaConfig(n_obstacles=0)
    assert clearance(AgentState(5.0, 0.0, 0.0), np.zeros((0, 2)), cfg) > 0


def test_build_arena_xml_emits_one_geom_per_obstacle():
    cfg = ArenaConfig(seed=1)
    xml = build_arena_xml(sample_obstacles(cfg), cfg)
    for i in range(cfg.n_obstacles):
        assert f'name="obstacle{i}"' in xml
    assert 'mocap="true"' in xml
    assert 'camera name="eye"' in xml


# --- rendering-dependent checks -------------------------------------------

os.environ.setdefault("MUJOCO_GL", "egl")

from flyguard.arena import ArenaRenderer  # noqa: E402


def _renderer(obstacles, cfg, res=96):
    try:
        return ArenaRenderer(obstacles, cfg, res)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"headless rendering unavailable: {exc}")


def test_arena_xml_compiles_and_renders_with_contrast():
    cfg = ArenaConfig(seed=0)
    r = _renderer(sample_obstacles(cfg), cfg)
    try:
        frame = r.render(AgentState(0.0, 0.0, 0.0))
    finally:
        r.close()
    assert frame.shape == (96, 96, 3)
    # A scene too dim or too flat to threshold has bitten this project before
    # (see the project notes on mujoco_world's lighting bug); require real structure.
    assert frame.astype(float).std() > 20.0


def test_image_side_matches_world_side():
    """World +y (the agent's left) must project to the LEFT half of the
    image. `flyguard.avoid.BilateralEncoder` splits the frame on exactly this
    assumption, and every steering sign downstream depends on it."""
    cfg = ArenaConfig(n_obstacles=1, first_obstacle_x=6.0)
    empty = _renderer(np.zeros((0, 2)), cfg)
    try:
        base = empty.render(AgentState(0.0, 0.0, 0.0)).astype(float)
    finally:
        empty.close()

    centroids = {}
    for label, oy in (("left", 2.0), ("right", -2.0)):
        r = _renderer(np.array([[6.0, oy]]), cfg)
        try:
            frame = r.render(AgentState(0.0, 0.0, 0.0)).astype(float)
        finally:
            r.close()
        changed = np.abs(frame - base).mean(axis=-1).sum(axis=0)
        centroids[label] = float((changed * np.arange(len(changed))).sum() / changed.sum())

    mid = 96 / 2
    assert centroids["left"] < mid < centroids["right"]


def test_heading_rotates_the_view():
    cfg = ArenaConfig(seed=0)
    r = _renderer(sample_obstacles(cfg), cfg)
    try:
        straight = r.render(AgentState(4.0, 0.0, 0.0)).astype(float)
        turned = r.render(AgentState(4.0, 0.0, 0.4)).astype(float)
        same = r.render(AgentState(4.0, 0.0, 0.0)).astype(float)
    finally:
        r.close()
    assert np.abs(straight - turned).mean() > 1.0
    np.testing.assert_array_equal(straight, same)  # rendering is deterministic


# --- what was hit, not just that something was -----------------------------


def test_collision_kind_separates_wall_from_obstacle():
    """A pillar strike is a detection failure; a wall strike is usually a
    steering one -- a controller that committed a turn and never corrected.
    Reporting them as one number hides which is happening."""
    cfg = ArenaConfig(lane_half_width=4.0, agent_radius=0.35, obstacle_radius=0.45)
    obstacles = np.array([[6.0, 0.0]])

    clear = AgentState(x=2.0, y=0.0, heading=0.0)
    assert collision_kind(clear, obstacles, cfg) == "none"

    on_pillar = AgentState(x=6.0, y=0.0, heading=0.0)
    assert collision_kind(on_pillar, obstacles, cfg) == "obstacle"

    # Far from any pillar, but against the -y wall.
    on_wall = AgentState(x=2.0, y=-(4.0 - 0.35), heading=0.0)
    assert collision_kind(on_wall, obstacles, cfg) == "wall"


def test_collision_kind_agrees_with_collides():
    cfg = ArenaConfig()
    obstacles = sample_obstacles(cfg)
    for x, y in ((1.0, 0.0), (6.0, 0.0), (3.0, 3.7), (12.0, -3.9)):
        state = AgentState(x=x, y=y, heading=0.0)
        hit = collides(state, obstacles, cfg)
        kind = collision_kind(state, obstacles, cfg)
        assert hit == (kind != "none"), (x, y, hit, kind)


def test_collision_kind_handles_an_empty_corridor():
    """Calibration runs use zero obstacles; the wall is then the only surface."""
    cfg = ArenaConfig(n_obstacles=0)
    empty = np.zeros((0, 2))
    assert collision_kind(AgentState(1.0, 0.0, 0.0), empty, cfg) == "none"
    assert collision_kind(AgentState(1.0, -3.7, 0.0), empty, cfg) == "wall"
