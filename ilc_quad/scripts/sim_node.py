#!/usr/bin/env python3
"""MuJoCo simulation node: owns the physics, publishes state, accepts torques.

The only thing in the system that touches `MjModel`/`MjData`, so there is exactly
one authority on the robot's state. It runs no control law of its own -- it holds
whatever torque was last commanded on `joint_torque_cmd` and reports what came of
it, leaving the control entirely to whatever you connect.

    subscribes  joint_torque_cmd  Float64MultiArray, 12 torques in canonical order
    publishes   joint_states      JointState, canonical order; effort is the
                                  *applied* torque (see below)
                base_odom         Odometry, twist in the base frame
                foot_contacts     Float64MultiArray, 4 normal forces FL FR RL RR
                /clock            monotonic sim time
    service     reset_trial       Trigger; returns the sim time of the reset

Three decisions exist because the controller is a separate process, which means a
command can arrive late, early, or not at all:

    applied torque  `JointState.effort` carries the torque the actuators actually
                    developed, read back after MuJoCo clamped it -- not the
                    command that was received. Where a motor saturated the two
                    differ, and it is the applied one that produced the motion, so
                    a controller that logs its own intended torque is logging a
                    trial that did not happen.
    zero-order hold the last received command is held until a new one arrives, so
                    a dropped message degrades a run slightly rather than dropping
                    the robot.
    monotonic clock `/clock` never goes backwards, resets included. Zeroing sim
                    time on each reset is the obvious alternative and it makes
                    every `use_sim_time` subscriber in the graph step through time
                    travel. `reset_trial` reports the sim time it happened at
                    instead, which is enough to recover run-relative time:
                    subtract it from any message's header stamp.

Best-effort depth-1 QoS for state and commands. For a 250 Hz stream the newest
sample is the only one worth having, and reliable QoS would buy retransmission of
a sample that is already stale at the cost of latency on the one that isn't.

`control_rate_hz` must divide the physics timestep evenly -- see
`QuadModel.substeps_for`. 250 Hz is 2 steps of 2 ms; 200 Hz would be 2.5 and is
rejected at startup.
"""

from __future__ import annotations

import mujoco
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from ilc_quad.sim_quad_model import CANONICAL_JOINT_NAMES, QuadModel, default_menagerie_root

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.VOLATILE,
)

# /clock wants reliable delivery: a dropped clock sample stalls every
# use_sim_time subscriber until the next one, and it is one small message a tick.
CLOCK_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class MujocoSimNode(Node):
    def __init__(self):
        super().__init__("mujoco_sim")

        self.declare_parameter("robot", "go2")
        self.declare_parameter("menagerie_root", default_menagerie_root())
        self.declare_parameter("control_rate_hz", 250.0)
        self.declare_parameter("realtime_factor", 1.0)
        self.declare_parameter("viewer", False)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("odom_frame", "odom")

        # The sim node is the clock's source, so it cannot run on it: a timer
        # waiting for /clock that only this node's timer can advance never fires.
        if self.get_parameter("use_sim_time").value:
            raise RuntimeError(
                "mujoco_sim must run with use_sim_time:=False -- it publishes "
                "/clock, so driving its own timers from /clock deadlocks. Only "
                "the other nodes in the graph should set use_sim_time:=True."
            )

        robot = self.get_parameter("robot").value
        root = self.get_parameter("menagerie_root").value
        self.qm = QuadModel(robot, root)
        self.get_logger().info(f"loaded {self.qm}")

        self.substeps = self.qm.substeps_for(
            float(self.get_parameter("control_rate_hz").value)
        )
        self.control_period = self.substeps * self.qm.dt
        self.base_frame = self.get_parameter("base_frame").value
        self.odom_frame = self.get_parameter("odom_frame").value

        # Sim time, monotonic for the life of the node.
        self.sim_time = 0.0
        # Zero-order-held command, canonical order.
        self.held_torque = np.zeros(12)
        self.commands_received = 0

        self.clock_pub = self.create_publisher(Clock, "/clock", CLOCK_QOS)
        self.joint_pub = self.create_publisher(JointState, "joint_states", SENSOR_QOS)
        self.odom_pub = self.create_publisher(Odometry, "base_odom", SENSOR_QOS)
        self.contact_pub = self.create_publisher(
            Float64MultiArray, "foot_contacts", SENSOR_QOS
        )
        self.create_subscription(
            Float64MultiArray, "joint_torque_cmd", self._on_torque_cmd, SENSOR_QOS
        )
        self.create_service(Trigger, "reset_trial", self._on_reset_trial)

        self.viewer = None
        if self.get_parameter("viewer").value:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.qm.model, self.qm.data)

        rtf = float(self.get_parameter("realtime_factor").value)
        # rtf <= 0 means "as fast as the executor will go". The period still has
        # to be nonzero or the timer starves the command subscription.
        wall_period = self.control_period / rtf if rtf > 0 else 1e-4
        self.create_timer(wall_period, self._step)

        self.get_logger().info(
            f"{self.substeps} physics substeps/tick "
            f"({1.0 / self.control_period:.0f} Hz control, {self.qm.dt * 1e3:.1f} ms "
            f"physics), realtime_factor={rtf if rtf > 0 else 'free'}"
        )

    # -- callbacks -----------------------------------------------------------

    def _on_torque_cmd(self, msg: Float64MultiArray) -> None:
        if len(msg.data) != 12:
            self.get_logger().warn(
                f"ignoring torque command of length {len(msg.data)}, expected 12"
            )
            return
        self.held_torque = np.asarray(msg.data, dtype=float)
        self.commands_received += 1

    def _on_reset_trial(self, request, response):
        """Reset to the home keyframe and report the sim time it happened at.

        The timestamp in `message` is how the controller recovers trial-relative
        time without the clock having to jump backwards: trial time for any
        message is its `header.stamp` minus this value.
        """
        self.qm.reset_home()
        self.held_torque = np.zeros(12)
        response.success = True
        response.message = f"{self.sim_time:.9f}"
        self.get_logger().info(f"trial reset at sim t={self.sim_time:.3f}s")
        return response

    # -- main loop -----------------------------------------------------------

    def _step(self) -> None:
        applied = self.qm.set_torques(self.held_torque)
        for _ in range(self.substeps):
            mujoco.mj_step(self.qm.model, self.qm.data)
        self.sim_time += self.control_period

        # Read back post-step: this is the torque the actuators developed, which
        # differs from `applied` wherever a motor saturated.
        self._publish(self.qm.applied_torques())

        if self.viewer is not None:
            if not self.viewer.is_running():
                self.get_logger().info("viewer closed, shutting down")
                raise SystemExit(0)
            self.viewer.sync()

    def _publish(self, effort: np.ndarray) -> None:
        stamp = self._stamp()

        clock = Clock()
        clock.clock = stamp
        self.clock_pub.publish(clock)

        js = JointState()
        js.header.stamp = stamp
        js.name = list(CANONICAL_JOINT_NAMES)
        js.position = self.qm.joint_positions().tolist()
        js.velocity = self.qm.joint_velocities().tolist()
        js.effort = effort.tolist()
        self.joint_pub.publish(js)

        self.odom_pub.publish(self._odometry(stamp))

        contacts = Float64MultiArray()
        contacts.data = self.qm.foot_normal_forces().tolist()
        self.contact_pub.publish(contacts)

    def _odometry(self, stamp) -> Odometry:
        """Base state as Odometry, with the twist moved into the base frame.

        `QuadModel.base_velocity_body` does the frame conversion, and the
        standalone rollout logs the same call, so a trial recorded through ROS and
        one recorded in-process report the twist in the same frame.
        """
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        pos = self.qm.base_position()
        quat = self.qm.base_quat()  # (w, x, y, z)
        vel = self.qm.base_velocity_body()

        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = pos
        odom.pose.pose.orientation.w = float(quat[0])
        odom.pose.pose.orientation.x = float(quat[1])
        odom.pose.pose.orientation.y = float(quat[2])
        odom.pose.pose.orientation.z = float(quat[3])

        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = vel[:3]
        odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z = vel[3:]
        return odom

    def _stamp(self):
        from builtin_interfaces.msg import Time

        stamp = Time()
        stamp.sec = int(self.sim_time)
        stamp.nanosec = int(round((self.sim_time - int(self.sim_time)) * 1e9))
        # Rounding can land exactly on a second; normalize or subscribers see a
        # stamp with nanosec == 1e9.
        if stamp.nanosec >= 1_000_000_000:
            stamp.sec += 1
            stamp.nanosec -= 1_000_000_000
        return stamp


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = None
    try:
        node = MujocoSimNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if node is not None:
            if node.viewer is not None:
                node.viewer.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
