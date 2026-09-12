"""
ROS2 node: real image-based vision front-end for looming_node.

**Not the node for a robot -- use `camera_node` for that.** This one renders
its own MuJoCo scene, so it cannot see a real camera; it exists to exercise
the vision path end to end with no hardware attached. `camera_node` runs the
same pipeline against `sensor_msgs/Image` from any source.

Publishes `flyguard/lplc2_drive` (`std_msgs/Float32`, [0,1]), computed from
a live headless MuJoCo scene via real optical flow
(`flyguard.optical_flow.lucas_kanade_flow_iterative`) sampled at the real
T4/T5 column population (`flyguard.encoder.flow_from_frame_pair` +
`encode_drive`) -- **not** the scripted
`flyguard.stimuli.net_opponency_drive` demo_stimulus_node computes from
synthetic ground-truth flow. This is `flyguard/validate_real_encoder.py`'s
methodology wired into the live control loop, in the "feed pre-computed
optic flow features directly into LPLC2, skipping T4/T5" mode CLAUDE.md's
hardware-constraints note calls for -- the T4/T5 population is used only
to compute a flow-driven scalar here (no connectome simulation at this
stage; that happens downstream, in looming_node's own small real circuit).
Measured cost on the reference machine: ~44 ms/tick (render + flow) at
128px, comfortably sustaining this node's default 15 Hz.

**Expect a weaker, noisier signal than `demo_stimulus_node`'s scripted
version -- this is not a bug to chase.** CLAUDE.md's "Pilot findings"
documents why: real pixel flow exists only near a rendered object's own
silhouette boundary, not across the whole visual field the synthetic flow
generators assumed, and the measured discrimination margin is real but
modest (1.15x ratio, 60% threshold accuracy on a held-out trial set, real
connectome, matched-depth trials). This node inherits that same
characterized limitation live; it does not somehow do better just by
running online, and `looming_node`'s e-stop will accordingly be less
crisp here than under `demo_stimulus_node`.

The disc trajectory is generated live from elapsed wall-clock time using
the same closed-form position formulas as `mujoco_world.trajectory`
(looming: constant-velocity approach, clipped so it never reaches the
camera; translation: a centered lateral sweep at constant depth), just
evaluated continuously rather than as a discrete pre-rendered array, since
this node runs as an unbounded live loop rather than a fixed-length trial.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

from flyguard.paths import default_columns_path
from flyguard.encoder import column_pixel_bounds, encode_drive, flow_from_frame_pair, load_column_assignment
from flyguard.mujoco_world import SCENE_XML

# Empirical two-point normaliser, not a theoretical bound: measured on
# matched-depth trials (flyguard/validate_real_encoder.py, depth0=5.0,
# right hemisphere) per-frame-pair summed drive across ~5900 columns was
# looming 466.6-735.0 (mean 566.3) vs. translation 273.2-509.7 (mean
# 400.9). A single-ended normaliser (sum / some_max) was tried first and
# put the translation baseline itself at drive~0.35-0.65 live -- LPi's
# (1-drive) inhibition in looming_node never got the contrast it needs,
# and /flyguard/estop latched permanently true even through "cruise"
# phases (same class of problem as the LPi-omission bug documented for
# looming_node, different cause: here the *signal*, not the *circuit*,
# lacked contrast). Fixed by anchoring both ends of the linear map to the
# measured means instead of one arbitrary max -- translation's mean maps
# near 0, looming's mean maps near 1, matching the [0,1] "confidently
# non-looming .. confidently looming" semantic looming_node expects.
# Recalibrate (rerun validate_real_encoder.py's diagnostic) if depth0,
# side, or resolution change enough to shift the real drive-sum range.
DRIVE_SUM_LOW = 400.0   # ~ translation mean
DRIVE_SUM_HIGH = 570.0  # ~ looming mean


def _default_columns_path() -> str:
    """
    Locate the FlyWire retinotopic map.

    See `flyguard.paths`: searches $FLYGUARD_COLUMNS, the working directory,
    ~/flywire/ and the packaged data directory, in that order.
    """
    return default_columns_path()


class VisionNode(Node):
    def __init__(self):
        super().__init__("vision_node")
        self.declare_parameter("columns_csv", _default_columns_path())
        self.declare_parameter("side", "right")
        self.declare_parameter("resolution", 128)
        self.declare_parameter("publish_rate_hz", 15.0)
        self.declare_parameter("cruise_s", 4.0)
        self.declare_parameter("loom_s", 2.0)
        self.declare_parameter("depth0", 5.0)
        self.declare_parameter("radius", 0.3)
        self.declare_parameter("loom_speed", 2.0)
        self.declare_parameter("window", 15)
        self.declare_parameter("n_iters", 2)

        columns_csv = Path(self.get_parameter("columns_csv").value)
        side = self.get_parameter("side").value
        resolution = int(self.get_parameter("resolution").value)
        rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.cruise_s = float(self.get_parameter("cruise_s").value)
        self.loom_s = float(self.get_parameter("loom_s").value)
        self.depth0 = float(self.get_parameter("depth0").value)
        self.radius = float(self.get_parameter("radius").value)
        self.loom_speed = float(self.get_parameter("loom_speed").value)
        self.window = int(self.get_parameter("window").value)
        self.n_iters = int(self.get_parameter("n_iters").value)

        if not columns_csv.exists():
            raise RuntimeError(
                f"column_assignment.csv not found at {columns_csv} -- the raw Codex CSVs "
                f"are not committed (see CLAUDE.md); pass --ros-args -p columns_csv:=<path>"
            )
        self.get_logger().info(f"loading T4/T5 columns from {columns_csv}")
        self.columns = load_column_assignment(columns_csv, side=side)
        self.x_bounds, self.y_bounds = column_pixel_bounds(self.columns)
        self.get_logger().info(f"{len(self.columns)} columns loaded (side={side})")

        self.model = mujoco.MjModel.from_xml_string(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=resolution, width=resolution)

        self.pub = self.create_publisher(Float32, "flyguard/lplc2_drive", 10)
        self.prev_frame = None
        self.t = 0.0
        self.dt = 1.0 / rate_hz
        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f"vision_node up: {self.cruise_s}s cruise / {self.loom_s}s looming cycle, "
            f"{rate_hz} Hz, real optical flow (not scripted)"
        )

    def _scene_xy(self, t: float) -> tuple[float, float]:
        """Disc (x, y) at wall-clock time t within the repeating cruise/loom cycle."""
        cycle = self.cruise_s + self.loom_s
        phase = t % cycle
        if phase < self.cruise_s:
            # translation: centered lateral sweep at constant depth, same
            # formula as mujoco_world.trajectory's translation branch
            half_width = self.depth0 * 0.4142  # CAMERA_HALF_FOV_TAN
            max_excursion = 0.6 * half_width
            speed = (2 * max_excursion) / self.cruise_s
            y = speed * phase - speed * self.cruise_s / 2.0
            return self.depth0, y
        loom_t = phase - self.cruise_s
        x = max(self.depth0 - self.loom_speed * loom_t, self.radius * 1.5)
        return x, 0.0

    def _render(self, x: float, y: float) -> np.ndarray:
        self.data.mocap_pos[0] = [x, y, 0.0]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="eye")
        return self.renderer.render()

    def _tick(self):
        x, y = self._scene_xy(self.t)
        frame = self._render(x, y)
        self.t += self.dt

        if self.prev_frame is not None:
            flow_vx, flow_vy = flow_from_frame_pair(
                self.prev_frame, frame, self.columns, self.x_bounds, self.y_bounds,
                window=self.window, n_iters=self.n_iters,
            )
            drive = encode_drive(self.columns, flow_vx, flow_vy)
            span = DRIVE_SUM_HIGH - DRIVE_SUM_LOW
            normalized = float(np.clip((drive.sum() - DRIVE_SUM_LOW) / span, 0.0, 1.0))
            self.pub.publish(Float32(data=normalized))
        self.prev_frame = frame

    def destroy_node(self):
        self.renderer.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
