"""
Companion demo node: publishes a scripted `flyguard/lplc2_drive` signal.

Exists so `looming_node` can be exercised and demoed without a camera or
MuJoCo wired in yet (the real image-based encoder is still future work --
see the project notes). Not a vision pipeline -- it reuses
`flyguard.stimuli`'s already-validated radial-opponency computation
(`arm_responses`/`net_opponency_drive`), alternating "cruise" (pure
translation, opponency drive ~0) and "looming event" (pure expansion,
opponency drive ~3.59, the proven ring-circuit maximum -- see
`flyguard/stimuli.py` "CONFIRMED" result) phases on a timer.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

from flyguard.stimuli import (
    arm_responses,
    expanding_flow,
    net_opponency_drive,
    translational_flow,
    visual_field_positions,
)

# net_opponency_drive for a pure expanding stimulus on the ring model --
# the proven maximum from flyguard/stimuli.py's own "CONFIRMED" result,
# used here only to normalise the demo drive signal into [0, 1].
LOOM_MAX_DRIVE = 3.5927


class DemoStimulusNode(Node):
    def __init__(self):
        super().__init__("demo_stimulus_node")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("cruise_s", 4.0)
        self.declare_parameter("loom_s", 2.0)

        rate = float(self.get_parameter("publish_rate_hz").value)
        self.cruise_s = float(self.get_parameter("cruise_s").value)
        self.loom_s = float(self.get_parameter("loom_s").value)

        self.positions = visual_field_positions()
        self.pub = self.create_publisher(Float32, "flyguard/lplc2_drive", 10)
        self.t = 0.0
        self.dt = 1.0 / rate
        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f"demo_stimulus_node: {self.cruise_s}s cruise / {self.loom_s}s looming, repeating"
        )

    def _tick(self):
        cycle = self.cruise_s + self.loom_s
        phase = self.t % cycle
        if phase < self.cruise_s:
            drive = net_opponency_drive(
                arm_responses(translational_flow(self.positions, 0.0), self.positions)
            )
        else:
            drive = net_opponency_drive(
                arm_responses(expanding_flow(self.positions), self.positions)
            )
        normalized = max(0.0, min(1.0, drive / LOOM_MAX_DRIVE))
        self.pub.publish(Float32(data=normalized))
        self.t += self.dt


def main(args=None):
    rclpy.init(args=args)
    node = DemoStimulusNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
