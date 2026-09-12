"""
ROS2 node: real FAFB v783 connectome as a live collision detector.

**Takes a drive scalar, not pixels.** For a robot use `camera_node`, which
does the vision itself; this node is the circuit half alone, useful for
exercising or profiling it with a scripted input (`demo_stimulus_node`).

Runs a ~530-neuron "core control circuit" (LPLC2 + LC4 + all real LPi
subtypes + DNp01, real synapses only -- see
`flyguard.extract.subset_subnetwork` and CLAUDE.md "Real-data
performance") as a persistent LIF simulation and publishes `/cmd_vel`.

Why this specific circuit, and not the full periphery: the full extracted
subnetwork (`data/looming.npz`, 18k neurons including T4/T5) runs at only
0.8x realtime on one CPU core -- too slow for a live control loop. This
node bypasses T4/T5 entirely and drives LPLC2 and LPi directly with an
externally supplied drive signal (`flyguard/lplc2_drive`, a normalised
[0,1] scalar computed upstream -- e.g. by `demo_stimulus_node` for now, or
eventually a real image-based encoder), the same dual-channel-excitation
-plus-inhibition idiom already proven in `flyguard.stimuli.simulate_condition`
(force-spiking both T4T5 and LPi relay neurons from a single computed
drive). What happens downstream (LPLC2/LPi -> DNp01, real anatomical
synapses, no learned weights) is exactly the real connectome.

**LPi is in the core circuit for a reason found by testing, not assumed
up front.** The first version of this node used only LPLC2 + LC4 + DNp01
(no inhibitory neurons at all) and drove LPLC2 alone. Running it live
showed `/flyguard/estop` latching permanently true even through the demo
stimulus's multi-second "cruise" (non-looming) phases -- the reduced
circuit, with zero inhibition, sustains runaway recurrent excitation once
triggered, structurally the same failure mode already characterized as the
"LPi ablated" condition in `validate_ablation.py`'s real-connectome
ablation study (this project had, by omission, built exactly that ablated
circuit into the ROS2 node). Fix: include the real LPi population and
drive it too, with rate `(1 - drive) * base_rate_hz` -- LPi should be
*more* active exactly when the visual evidence looks like non-looming
motion, matching radial motion opponency's own logic and restoring the
real inhibitory synapses (`LPi -> LPLC2`, 989 edges, -11124 weight, same
numbers as everywhere else in this project) that keep the circuit from
just staying maximally excited.

**Why only an e-stop, not steering.** CLAUDE.md's cell-type table lists
DNa01/DNa02 (steering) and MDN (reverse) as robot analogues for LPLC2/LC4's
downstream targets, but the actual extracted connectivity
(`data/looming.npz`, hops=1) shows LPLC2/LC4 project directly and purely
excitatorily to DNp01/DNp02/DNp04/DNp11 (231/103/252/83 edges) and have
**zero** direct edges to DNa01, DNa02, or MDN. That's textbook Giant Fiber
escape-pathway anatomy (a fast, dedicated reflex, not a steering circuit),
not a bug -- so this node only ever commands stop/go, matching what's
actually wired, rather than fabricating a steering signal the connectome
doesn't support at this hop distance.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Float32

from flyguard.runtime import CoreCircuit
from flyguard.runtime.circuit import default_npz_path


def _default_data_path() -> str:
    return str(default_npz_path())


class LoomingNode(Node):
    def __init__(self):
        super().__init__("looming_node")

        self.declare_parameter("data_npz", _default_data_path())
        self.declare_parameter("cruise_linear_x", 0.3)
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("lif_dt", 1e-4)
        self.declare_parameter("base_rate_hz", 250.0)
        self.declare_parameter("peak_weight", 30.0)
        self.declare_parameter("estop_cooldown_s", 0.5)

        npz_path = Path(self.get_parameter("data_npz").value)
        self.cruise_linear_x = float(self.get_parameter("cruise_linear_x").value)
        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.base_rate_hz = float(self.get_parameter("base_rate_hz").value)
        self.peak_weight = float(self.get_parameter("peak_weight").value)
        self.estop_cooldown_s = float(self.get_parameter("estop_cooldown_s").value)

        self.get_logger().info(f"loading core circuit from {npz_path}")
        # One shared implementation: `flyguard.runtime.CoreCircuit` is the same
        # object the closed-loop benchmark and the demo recorder run, so this
        # node cannot drift away from the numbers the repo reports.
        self.circuit = CoreCircuit(
            npz_path,
            tick_hz=control_rate_hz,
            lif_dt=float(self.get_parameter("lif_dt").value),
            base_rate_hz=self.base_rate_hz,
            peak_weight=self.peak_weight,
            seed=0,
            record=("LPLC2", "DNp01"),
        )
        self.get_logger().info(self.circuit.describe())

        self._drive = 0.0  # last received flyguard/lplc2_drive value, zero-order hold
        self._estop_until = 0.0  # monotonic time the current e-stop latch expires

        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.lplc2_rate_pub = self.create_publisher(Float32, "flyguard/lplc2_rate", 10)
        self.estop_pub = self.create_publisher(Bool, "flyguard/estop", 10)

        self.create_subscription(Float32, "flyguard/lplc2_drive", self._on_drive, 10)
        self.timer = self.create_timer(1.0 / control_rate_hz, self._tick)

        self.get_logger().info(
            f"looming_node up: {self.circuit.n_steps_per_tick} LIF steps/tick "
            f"@ {control_rate_hz} Hz control rate"
        )

    def _on_drive(self, msg: Float32):
        self._drive = float(np.clip(msg.data, 0.0, 1.0))

    def _tick(self):
        response = self.circuit.step(self._drive)
        dnp01_fired = response.any_spike("DNp01")
        # measured LPLC2 population rate (not just the input drive echoed back)
        lplc2_hz = response.rate_hz("LPLC2")

        now = time.monotonic()
        if dnp01_fired:
            self._estop_until = now + self.estop_cooldown_s
        estop_active = now < self._estop_until

        cmd = Twist()
        cmd.linear.x = 0.0 if estop_active else self.cruise_linear_x
        self.cmd_pub.publish(cmd)
        self.lplc2_rate_pub.publish(Float32(data=lplc2_hz))
        self.estop_pub.publish(Bool(data=estop_active))


def main(args=None):
    rclpy.init(args=args)
    node = LoomingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
