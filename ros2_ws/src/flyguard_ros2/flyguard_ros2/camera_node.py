"""
ROS2 node: the connectome collision detector on a real camera.

    ros2 run flyguard_ros2 camera_node --ros-args \
        -p image_topic:=/camera/image_raw \
        -p columns_csv:=/path/to/column_assignment.csv \
        -p calibration:=/path/to/my_camera.json

Subscribes to `sensor_msgs/Image` from any camera -- USB, a depth camera's RGB
stream, a Gazebo plugin, a rosbag -- and publishes `geometry_msgs/Twist` plus
telemetry. This is the node to use on a robot; `vision_node` renders its own
MuJoCo scene and exists only to exercise the pipeline without hardware, while
`looming_node` takes a pre-computed drive scalar rather than pixels.

## Read this before flying it

**`estop` is anatomical. `angular.z` is not.** LPLC2/LC4 project onto the
Giant Fiber escape pathway with zero direct edges onto the steering descending
neurons, so braking is a pathway this repo measured and turning is an
engineering addition on top. See `flyguard.runtime.steering`.

**It needs a calibration measured on your camera.** The four constants in the
calibration file depend on field of view, resolution, world texture and
vehicle speed. Without one the node cruises straight and never brakes, and it
says so at startup rather than pretending. Measure one with
`python -m flyguard.calibrate`.

**Run the `flow` backend alongside it at least once.** `-p backend:=flow`
keeps every gain, normalisation and calibration constant identical and removes
only the 530 neurons. On this project's simulated corridor benchmark the
circuit did *not* beat that baseline, and a real robot is a harsher test. If
you report what the connectome contributes, report it against this.

## Timing

The circuit costs real time (~250 ms of LIF per 100 ms tick on one core at the
default rates), so the node **drops frames rather than queueing them**: it
keeps the most recent image and processes one per timer tick. A queue would
grow without bound and steer on stale pixels, which is worse than a lower
frame rate. `flyguard/dropped_frames` publishes the running count so a
too-slow tick rate is visible rather than silent.
"""

from __future__ import annotations

import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Int32

from flyguard.runtime import Calibration, FlyGuardPilot
from flyguard.runtime.circuit import default_npz_path
from flyguard.runtime.ros_image import image_to_array


class CameraNode(Node):
    def __init__(self):
        super().__init__("flyguard_camera_node")

        self.declare_parameter("image_topic", "image_raw")
        self.declare_parameter("columns_csv", "")
        self.declare_parameter("calibration", "")
        self.declare_parameter("data_npz", str(default_npz_path()))
        self.declare_parameter("backend", "connectome")
        self.declare_parameter("control_rate_hz", 10.0)
        self.declare_parameter("cruise_linear_x", 0.3)
        self.declare_parameter("escape_linear_x", 0.09)
        self.declare_parameter("max_angular_z", 1.6)
        self.declare_parameter("stale_frame_timeout_s", 1.0)

        image_topic = self.get_parameter("image_topic").value
        columns_csv = self.get_parameter("columns_csv").value
        calibration_path = self.get_parameter("calibration").value
        backend = self.get_parameter("backend").value
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.stale_timeout_s = float(self.get_parameter("stale_frame_timeout_s").value)

        if not columns_csv:
            raise RuntimeError(
                "parameter `columns_csv` is required: the FlyWire "
                "column_assignment.csv that says where each real T4/T5 neuron "
                "looks. Download it from https://codex.flywire.ai (snapshot 783)."
            )

        if calibration_path:
            calibration = Calibration.load(Path(calibration_path))
            self.get_logger().info(f"calibration: {calibration_path}")
        else:
            calibration = Calibration()
            self.get_logger().warn(
                "no `calibration` parameter: the pilot will cruise straight and "
                "never brake. Measure one with `python -m flyguard.calibrate`."
            )

        self.pilot = FlyGuardPilot(
            columns_csv=Path(columns_csv),
            calibration=calibration,
            backend=backend,
            npz_path=Path(self.get_parameter("data_npz").value),
            tick_hz=self.control_rate_hz,
            v_cruise=float(self.get_parameter("cruise_linear_x").value),
            v_escape=float(self.get_parameter("escape_linear_x").value),
            saccade_rate=float(self.get_parameter("max_angular_z").value),
            logger=self.get_logger().warn,
        )
        self.get_logger().info(f"backend: {backend}")

        self._latest = None          # most recent image, replaced not queued
        self._latest_stamp = 0.0
        self._dropped = 0
        self._seen = 0

        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.estop_pub = self.create_publisher(Bool, "flyguard/estop", 10)
        self.rate_pub = self.create_publisher(Float32, "flyguard/lplc2_rate", 10)
        self.turn_pub = self.create_publisher(Float32, "flyguard/turn", 10)
        self.dropped_pub = self.create_publisher(Int32, "flyguard/dropped_frames", 10)

        self.create_subscription(Image, image_topic, self._on_image, 1)
        self.create_timer(1.0 / self.control_rate_hz, self._tick)
        self.get_logger().info(
            f"flyguard camera_node up: {image_topic} -> cmd_vel "
            f"@ {self.control_rate_hz} Hz"
        )

    def _on_image(self, msg: Image):
        # Replace, never append. See the timing note in the module docstring.
        if self._latest is not None:
            self._dropped += 1
        try:
            self._latest = image_to_array(msg)
        except ValueError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=5.0)
            self._latest = None
            return
        self._latest_stamp = time.monotonic()
        self._seen += 1

    def _publish_stop(self, reason: str):
        self.get_logger().warn(reason, throttle_duration_sec=5.0)
        self.cmd_pub.publish(Twist())
        self.estop_pub.publish(Bool(data=True))

    def _tick(self):
        frame, self._latest = self._latest, None
        now = time.monotonic()

        if frame is None:
            # No image this tick. If the stream has been quiet longer than the
            # timeout, stop the vehicle and drop the frame history -- a pair
            # straddling a gap produces a huge spurious flow field, which would
            # read as an obstacle that is not there.
            if self._seen and now - self._latest_stamp > self.stale_timeout_s:
                self.pilot.reset()
                self._publish_stop(
                    f"no camera frame for {now - self._latest_stamp:.1f}s -- stopping"
                )
            return

        cmd = self.pilot.step(frame)
        twist = Twist()
        twist.linear.x = float(cmd.v)
        twist.angular.z = float(cmd.omega)
        self.cmd_pub.publish(twist)
        self.estop_pub.publish(Bool(data=bool(cmd.estop)))
        self.dropped_pub.publish(Int32(data=self._dropped))

        telemetry = cmd.telemetry
        self.turn_pub.publish(Float32(data=float(telemetry.get("turn", 0.0))))
        rate = 0.5 * (telemetry.get("lplc2_left_hz", 0.0)
                      + telemetry.get("lplc2_right_hz", 0.0))
        self.rate_pub.publish(Float32(data=float(rate)))


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
