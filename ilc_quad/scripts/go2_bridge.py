#!/usr/bin/env python3
"""Bridge between this package's topics and a real Go2 over unitree_ros2.

Makes the robot look like `sim_node` to a controller, so `ilc_jump_sim.py` drives
hardware and simulation alike:

    subscribes  /lowstate     unitree_go/LowState   (from the robot)
                joint_cmd     Float64MultiArray, [q, dq, kp, kd, tau] x 12, canonical
    publishes   /lowcmd       unitree_go/LowCmd     (to the robot), at `rate_hz`
                joint_states  JointState, canonical order; effort = tau_est
                foot_contacts Float64MultiArray, 4 foot forces FL FR RL RR (raw sensor)
    service     clear_fault   Trigger: release a latched damping fault

The PD in joint_cmd runs on the motor drivers, which is what LowCmd's per-motor
q/dq/kp/kd/tau is for; this node only re-orders legs, clips and signs the message.

Safety, independent of the controller:
    watchdog     no joint_cmd for `cmd_timeout` s -> damping (kp = 0, kd = damp_kd,
                 tau = 0), until commands resume
    joint limits a measured joint more than `limit_margin` past its MJCF range ->
                 damping, latched until /clear_fault
    clipping     tau to the MJCF torque limits, q targets to the joint range, kp/kd
                 to [0, kp_max] / [0, kd_max]
Nothing is sent until the first /lowstate arrives.

Before running: the robot's sport mode must be released (Unitree app, or the
motion_switcher service), or it keeps fighting the low-level commands. Needs
unitree_ros2's `unitree_go` messages built and sourced, with CycloneDDS on the
robot's interface -- see unitree_ros2's setup.sh.
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

try:
    from unitree_go.msg import LowCmd, LowState
except ImportError as err:  # pragma: no cover - depends on the robot workstation
    raise SystemExit(
        "go2_bridge needs unitree_ros2's unitree_go messages: build unitree_ros2 and "
        f"source its setup before running ({err})"
    )

from ilc_quad.go2_lowcmd import CANONICAL_TO_MOTOR, FOOT_FROM_UNITREE, fill_lowcmd
from ilc_quad.sim_quad_model import (
    CANONICAL_JOINT_NAMES,
    JOINT_CMD_LEN,
    NU,
    QuadModel,
    default_menagerie_root,
    unpack_joint_cmd,
)

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class Go2Bridge(Node):
    def __init__(self):
        super().__init__("go2_bridge")
        p = self.declare_parameter
        p("menagerie_root", default_menagerie_root())
        p("rate_hz", 500.0)
        p("cmd_timeout", 0.05)
        p("damp_kd", 2.0)
        p("limit_margin", 0.05)
        p("kp_max", 100.0)
        p("kd_max", 10.0)
        p("lowcmd_topic", "/lowcmd")
        p("lowstate_topic", "/lowstate")
        get = lambda name: self.get_parameter(name).value

        # limits from the same MJCF the controller plans with
        qm = QuadModel("go2", get("menagerie_root"))
        self.joint_range = qm.joint_range
        self.tau_limit = qm.torque_limit
        self.cmd_timeout = float(get("cmd_timeout"))
        self.damp_kd = float(get("damp_kd"))
        self.limit_margin = float(get("limit_margin"))
        self.kp_max, self.kd_max = float(get("kp_max")), float(get("kd_max"))

        self.cmd = None                 # unpacked (5, 12) joint_cmd
        self.cmd_time = 0.0
        self.q = None                   # last measured joint angles, canonical
        self.fault = None               # latched reason, or None
        self.was_damping = True

        self.lowcmd_pub = self.create_publisher(LowCmd, get("lowcmd_topic"), 10)
        self.create_subscription(LowState, get("lowstate_topic"), self._on_lowstate, 10)
        self.create_subscription(Float64MultiArray, "joint_cmd", self._on_joint_cmd, SENSOR_QOS)
        self.joint_pub = self.create_publisher(JointState, "joint_states", SENSOR_QOS)
        self.contact_pub = self.create_publisher(Float64MultiArray, "foot_contacts", SENSOR_QOS)
        self.create_service(Trigger, "clear_fault", self._on_clear_fault)
        self.lowcmd = LowCmd()
        self.create_timer(1.0 / float(get("rate_hz")), self._send)
        self.get_logger().info(
            f"go2 bridge up: {get('lowstate_topic')} -> joint_states, joint_cmd -> "
            f"{get('lowcmd_topic')} at {get('rate_hz')} Hz; damping until commands arrive")

    def _on_lowstate(self, msg):
        motors = msg.motor_state
        q = np.array([motors[int(i)].q for i in CANONICAL_TO_MOTOR])
        dq = np.array([motors[int(i)].dq for i in CANONICAL_TO_MOTOR])
        tau = np.array([motors[int(i)].tau_est for i in CANONICAL_TO_MOTOR])
        self.q = q

        lo, hi = self.joint_range[:, 0] - self.limit_margin, self.joint_range[:, 1] + self.limit_margin
        bad = np.where((q < lo) | (q > hi))[0]
        if bad.size and self.fault is None:
            names = ", ".join(f"{CANONICAL_JOINT_NAMES[i]}={q[i]:.2f}" for i in bad)
            self.fault = f"joint past its limit: {names}"
            self.get_logger().error(f"damping latched: {self.fault}; /clear_fault to release")

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(CANONICAL_JOINT_NAMES)
        js.position, js.velocity, js.effort = q.tolist(), dq.tolist(), tau.tolist()
        self.joint_pub.publish(js)
        contacts = Float64MultiArray()
        contacts.data = [float(msg.foot_force[int(k)]) for k in FOOT_FROM_UNITREE]
        self.contact_pub.publish(contacts)

    def _on_joint_cmd(self, msg):
        if len(msg.data) != JOINT_CMD_LEN:
            self.get_logger().warn(f"ignoring joint_cmd of length {len(msg.data)}", once=True)
            return
        self.cmd = unpack_joint_cmd(msg.data)
        self.cmd_time = time.monotonic()

    def _on_clear_fault(self, request, response):
        response.success = self.fault is not None
        response.message = f"cleared: {self.fault}" if self.fault else "no fault"
        self.fault = None
        return response

    def _send(self):
        if self.q is None:
            return                      # nothing to the robot before it has spoken
        fresh = self.cmd is not None and time.monotonic() - self.cmd_time < self.cmd_timeout
        damping = self.fault is not None or not fresh
        if damping != self.was_damping:
            self.get_logger().info("damping" if damping else "following joint_cmd")
            self.was_damping = damping
        if damping:
            zeros = np.zeros(NU)
            fill_lowcmd(self.lowcmd, self.q, zeros, zeros, np.full(NU, self.damp_kd), zeros)
        else:
            q_des, dq_des, kp, kd, tau = self.cmd
            fill_lowcmd(
                self.lowcmd,
                np.clip(q_des, self.joint_range[:, 0], self.joint_range[:, 1]),
                dq_des,
                np.clip(kp, 0.0, self.kp_max),
                np.clip(kd, 0.0, self.kd_max),
                np.clip(tau, -self.tau_limit, self.tau_limit),
            )
        self.lowcmd_pub.publish(self.lowcmd)


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = None
    try:
        node = Go2Bridge()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
