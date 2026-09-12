"""A 3-D arena for *closed-loop* obstacle avoidance -- the first place in
this project where the connectome's output feeds back into what the camera
sees next.

Everything before this is open loop: `mujoco_world` scripts a stimulus, the
detector watches it, and the detector's opinion changes nothing. Here the
agent drives through a corridor of obstacles, and its own steering decides
what the next frame contains. That closes the loop, and it changes what can go wrong --
an open-loop detector that is merely noisy becomes, in closed loop, a
detector that can steer itself into the thing it was supposed to avoid.

Design choices, and why:

* **The agent is kinematic (unicycle), not a physics body.** Same reasoning
  as `mujoco_world`'s mocap stimulus: there is no contact dynamics worth
  simulating here, only a trajectory, so integrating pose in closed form
  keeps runs bit-reproducible and independent of CPU speed. Collisions are
  detected geometrically (`collides`) rather than by MuJoCo contact, so a
  collision is an *outcome the experiment records*, not a force that
  perturbs the trajectory.
* **The camera rides in the mocap body.** Heading is a quaternion about z;
  the camera's local axes are the same as `mujoco_world`'s fixed eye, so at
  heading 0 it looks down world +x with world +z up.
* **fovy=90.** A fly's visual field is panoramic; MuJoCo's 45-degree default
  is far too narrow to see an obstacle in time to steer around it. 90
  degrees is still not panoramic -- it is the widest a rectilinear pinhole
  projection stays usable at -- and the gap is a stated limitation, not a
  solved problem.
* **Walls bound the corridor.** Without them the cheapest winning policy is
  "steer far off to one side and cruise past everything," which measures
  nothing. Walls make the task a genuine slalom and count as collisions.

Obstacles are placed at fixed x spacing with jittered y, so every arena is
the same difficulty in the one way that matters (how many obstacles you must
pass) and differs in the way that matters for generalisation (where they
are).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

# Matches `mujoco_world.CAMERA_HALF_FOV_TAN`'s role but for this scene's
# wider camera: tan(45 deg) for fovy=90 with a square image.
CAMERA_HALF_FOV_TAN = 1.0


@dataclass(frozen=True)
class ArenaConfig:
    """Geometry of one randomised corridor.

    The defaults are sized so the task is neither trivial nor impossible,
    which is a property of the numbers rather than of taste:

    * An obstacle blocks a straight run when its centre falls within
      `obstacle_radius + agent_radius` = 0.80 m of the axis. With y drawn
      uniformly from +-1.6 m that is p = 0.5 per obstacle, so a blind
      straight-line run through 7 of them collides with probability
      1 - 0.5**7 ~ 0.99. The vision-free baseline should almost always fail,
      and measured over 10 arenas it fails on all of them.
    * The widest gap an obstacle can leave against a wall is
      `lane_half_width - (y_jitter + obstacle_radius)` = 1.95 m, which is
      1.60 m of clearance for a 0.35 m agent. Every arena is passable.

    An earlier, narrower corridor left only 0.5 m of clearance and made
    wall collisions -- not obstacle collisions -- the thing being measured.
    """

    n_obstacles: int = 7
    lane_half_width: float = 4.0     # corridor half-width (walls at +-this)
    obstacle_radius: float = 0.45
    obstacle_height: float = 3.0
    first_obstacle_x: float = 6.0    # clear run-up so the agent is up to speed
    obstacle_spacing: float = 3.5
    y_jitter: float = 1.6            # obstacles land in [-y_jitter, +y_jitter]
    agent_radius: float = 0.35
    eye_height: float = 0.7
    seed: int = 0
    corridor_length: float | None = None

    @property
    def goal_x(self) -> float:
        """Finishing line: two metres past the last obstacle, unless
        `corridor_length` overrides it. The override exists so an *empty*
        calibration corridor can be built at exactly the same length as the
        arenas being measured -- otherwise `n_obstacles=0` would shrink the
        world and change the very flow statistics being calibrated."""
        if self.corridor_length is not None:
            return self.corridor_length
        return self.first_obstacle_x + max(self.n_obstacles - 1, 0) * self.obstacle_spacing + 2.0


@dataclass(frozen=True)
class AgentState:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0  # radians, 0 = facing world +x


def sample_obstacles(cfg: ArenaConfig) -> np.ndarray:
    """Obstacle centres, shape [n_obstacles, 2].

    x is on a fixed grid so every arena poses the same number of passes;
    only y is randomised. Successive obstacles are nudged apart in y when
    they would otherwise line up, so the corridor never degenerates into a
    straight unobstructed shot.
    """
    rng = np.random.default_rng(cfg.seed)
    xs = cfg.first_obstacle_x + np.arange(cfg.n_obstacles) * cfg.obstacle_spacing
    ys = rng.uniform(-cfg.y_jitter, cfg.y_jitter, size=cfg.n_obstacles)
    return np.stack([xs, ys], axis=1)


_XML_HEAD = """
<mujoco>
  <visual>
    <headlight ambient="0.85 0.85 0.85" diffuse="0.5 0.5 0.5" specular="0 0 0"/>
    <map znear="0.02" zfar="120"/>
  </visual>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1=".42 .44 .50" rgb2=".66 .68 .74"
             width="300" height="300"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="30 30" reflectance="0"/>
    <texture name="wall_tex" type="cube" builtin="checker" rgb1=".28 .31 .38" rgb2=".58 .61 .68"
             width="128" height="128"/>
    <material name="wall_mat" texture="wall_tex" texrepeat="20 3" reflectance="0"/>
    <texture name="obst_tex" type="cube" builtin="checker" rgb1=".04 .04 .06" rgb2=".72 .45 .22"
             width="128" height="128"/>
    <material name="obst_mat" texture="obst_tex" texrepeat="4 2" reflectance="0"/>
  </asset>
  <worldbody>
    <light pos="0 0 14" dir="0 0 -1" diffuse=".55 .55 .55" directional="true"/>
"""

_XML_TAIL = """
  </worldbody>
</mujoco>
"""


def build_arena_xml(obstacles: np.ndarray, cfg: ArenaConfig) -> str:
    """MuJoCo XML for one arena: textured floor, two corridor walls, a far
    backdrop, one cylinder per obstacle, and the mocap body carrying the eye.

    Every surface is textured on purpose. Lucas-Kanade needs local intensity
    structure to estimate flow at all; a flat-shaded obstacle would produce
    flow only at its silhouette, which is a property of the renderer rather
    than of the stimulus and would quietly make the task about edges.
    """
    half = cfg.lane_half_width
    end_x = cfg.goal_x + 6.0
    parts = [_XML_HEAD]
    parts.append(
        f'    <geom name="floor" type="plane" pos="{end_x / 2:.3f} 0 0" '
        f'size="{end_x:.3f} {half + 6:.3f} 0.1" material="floor_mat"/>\n'
    )
    for name, sign in (("wall_pos_y", 1.0), ("wall_neg_y", -1.0)):
        parts.append(
            f'    <geom name="{name}" type="box" '
            f'pos="{end_x / 2:.3f} {sign * half:.3f} 1.5" '
            f'size="{end_x / 2:.3f} 0.12 1.5" material="wall_mat"/>\n'
        )
    # Far backdrop: gives the horizon some texture without ever looming
    # during a run (it sits well beyond the finishing line).
    parts.append(
        f'    <geom name="backdrop" type="box" pos="{end_x + 14:.3f} 0 4" '
        f'size="0.2 {half + 6:.3f} 4" material="wall_mat"/>\n'
    )
    for i, (ox, oy) in enumerate(obstacles):
        parts.append(
            f'    <geom name="obstacle{i}" type="cylinder" '
            f'pos="{ox:.3f} {oy:.3f} {cfg.obstacle_height / 2:.3f}" '
            f'size="{cfg.obstacle_radius:.3f} {cfg.obstacle_height / 2:.3f}" '
            f'material="obst_mat"/>\n'
        )
    parts.append(
        f'    <body name="agent" mocap="true" pos="0 0 {cfg.eye_height:.3f}">\n'
        f'      <camera name="eye" fovy="90" pos="0 0 0" xyaxes="0 -1 0 0 0 1"/>\n'
        f'    </body>\n'
    )
    parts.append(_XML_TAIL)
    return "".join(parts)


def step_agent(state: AgentState, v: float, omega: float, dt: float) -> AgentState:
    """Unicycle kinematics, forward Euler. `omega > 0` turns left (toward
    world +y), which is the agent's left, which is the *left* half of the
    rendered image -- see `flyguard.avoid` for the image-side convention."""
    return AgentState(
        x=state.x + v * math.cos(state.heading) * dt,
        y=state.y + v * math.sin(state.heading) * dt,
        heading=state.heading + omega * dt,
    )


def clearance(state: AgentState, obstacles: np.ndarray, cfg: ArenaConfig) -> float:
    """Signed gap (metres) to the nearest obstacle surface or wall. Negative
    means the agent's disc is already overlapping something."""
    if len(obstacles):
        d_obst = np.hypot(obstacles[:, 0] - state.x, obstacles[:, 1] - state.y).min()
        gap_obst = float(d_obst - cfg.obstacle_radius - cfg.agent_radius)
    else:
        gap_obst = float("inf")
    gap_wall = float(cfg.lane_half_width - cfg.agent_radius - abs(state.y))
    return min(gap_obst, gap_wall)


def collides(state: AgentState, obstacles: np.ndarray, cfg: ArenaConfig) -> bool:
    return clearance(state, obstacles, cfg) <= 0.0


class ArenaRenderer:
    """Holds the compiled model and an EGL renderer for one arena."""

    def __init__(self, obstacles: np.ndarray, cfg: ArenaConfig, resolution: int = 128):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_string(build_arena_xml(obstacles, cfg))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=resolution, width=resolution)
        self.resolution = resolution

    def render(self, state: AgentState) -> np.ndarray:
        """Render the agent's eye view at the given pose. [res, res, 3] uint8."""
        half = state.heading / 2.0
        self.data.mocap_pos[0] = [state.x, state.y, self.cfg.eye_height]
        self.data.mocap_quat[0] = [math.cos(half), 0.0, 0.0, math.sin(half)]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="eye")
        return self.renderer.render()

    def close(self):
        self.renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
