#!/usr/bin/env python3
"""Bridge between this package's topics and a real Go1 over unitree_legged_sdk.

Makes the robot look like `sim_node` to a controller, so `ilc_jump_sim.py` drives
hardware and simulation alike. Go1 has no DDS interface (unitree_ros2 is Go2/B2
only); low-level control is UDP to the motor controller board through
unitree_legged_sdk v3.8, via its Python binding `robot_interface`:

    UDP in       LowState      (from the robot, every cycle)
    subscribes   joint_cmd     Float64MultiArray, [q, dq, kp, kd, tau] x 12, canonical
    UDP out      LowCmd        (to the robot), one per cycle at `rate_hz`
    publishes    joint_states  JointState, canonical order; effort = tauEst
                 foot_contacts Float64MultiArray, 4 foot forces FL FR RL RR (raw sensor)
    service      clear_fault   Trigger: release a latched damping fault

The PD in joint_cmd runs on the motor drivers, which is what LowCmd's per-motor
q/dq/Kp/Kd/tau is for; this node only re-orders legs and clips. Motor slots run FR,
FL, RR, RL like Go2's, so go2_lowcmd's leg maps apply unchanged.

Safety, independent of the controller (the same as go2_bridge):
    watchdog     no joint_cmd for `cmd_timeout` s -> damping (kp = 0, kd = damp_kd,
                 tau = 0), until commands resume
    joint limits a measured joint more than `limit_margin` (0.20 rad) past its MJCF range ->
                 damping, latched until /clear_fault
    clipping     tau to the MJCF torque limits, q targets to the joint range, kp/kd
                 to [0, kp_max] / [0, kd_max]
    PowerProtect the SDK's own power limiter, at `power_level` (1..10 = 10..100 %).
                 A jump needs the motors' full power, so the default is 10; lower it
                 for first tests of the stand, not for jumping.
Until the first LowState arrives the SDK's idle command goes out (stop values, zero
gains, zero torque: the motors are passive), as in Unitree's examples -- the board
answers commands, so staying silent would wait forever.

Before running:
  * unitree_legged_sdk v3.8.x built with its Python wrapper; `sdk_path` (or
    PYTHONPATH) pointing at the directory holding robot_interface*.so, e.g.
    unitree_legged_sdk/lib/python/amd64.
  * The workstation wired to the robot on 192.168.123.x (e.g. 192.168.123.162).
  * The robot in low-level mode: from standing, L2+A (lie down), L2+B (damp), then
    L1+L2+Start. Otherwise the sport controller keeps fighting the commands.
  * The robot hanging or lying down: the first commands are damping only, but the
    ILC node stands it up as soon as a trial starts.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from ilc_quad.go2_lowcmd import CANONICAL_TO_MOTOR, FOOT_FROM_UNITREE, POS_STOP_F, VEL_STOP_F
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

LOWLEVEL = 0xFF
MODE_SERVO = 0x0A               # Go1/A1 motor mode for closed-loop servo


class Go1Bridge(Node):
    def __init__(self):
        super().__init__("go1_bridge")
        p = self.declare_parameter
        p("menagerie_root", default_menagerie_root())
        p("sdk_path", "")                 # directory with robot_interface*.so
        p("robot_ip", "192.168.123.10")
        p("local_port", 8080)
        p("robot_port", 8007)
        p("rate_hz", 500.0)
        p("power_level", 10)              # sdk.Safety.PowerProtect factor, 1..10
        p("cmd_timeout", 0.05)
        p("damp_kd", 2.0)
        # 0.20 rad (was 0.05): jumping drives the rear calves through full extension at takeoff -- up to 0.14 rad past
        # -0.89 at 10-16 rad/s in 25% of the CPU robots' jumps (log/dilc/plane/fixbox/audit_dynamics.py) -- which
        # would latch damping mid-takeoff; 0.20 still catches a runaway joint
        p("limit_margin", 0.20)
        p("kp_max", 100.0)
        p("kd_max", 10.0)
        get = lambda name: self.get_parameter(name).value

        if get("sdk_path"):
            sys.path.insert(0, get("sdk_path"))
        try:
            import robot_interface as sdk
        except ImportError as err:  # pragma: no cover - depends on the robot workstation
            raise SystemExit(
                "go1_bridge needs unitree_legged_sdk's Python wrapper (robot_interface): "
                "build unitree_legged_sdk v3.8 and set sdk_path to lib/python/<arch> "
                f"({err})")
        self.sdk = sdk

        # limits from the same MJCF the controller plans with
        qm = QuadModel("go1", get("menagerie_root"))
        self.joint_range = qm.joint_range
        self.tau_limit = qm.torque_limit
        self.cmd_timeout = float(get("cmd_timeout"))
        self.damp_kd = float(get("damp_kd"))
        self.limit_margin = float(get("limit_margin"))
        self.kp_max, self.kd_max = float(get("kp_max")), float(get("kd_max"))
        self.power_level = int(get("power_level"))
        if not 1 <= self.power_level <= 10:
            raise ValueError(f"power_level must be 1..10, got {self.power_level}")

        self.udp = sdk.UDP(LOWLEVEL, int(get("local_port")), get("robot_ip"),
                           int(get("robot_port")))
        self.safe = sdk.Safety(sdk.LeggedType.Go1)
        self.lowcmd, self.lowstate, self.idle = sdk.LowCmd(), sdk.LowState(), sdk.LowCmd()
        for c in (self.lowcmd, self.idle):
            self.udp.InitCmdData(c)
            c.levelFlag = LOWLEVEL

        self.cmd = None                 # unpacked (5, 12) joint_cmd
        self.cmd_time = 0.0
        self.q = None                   # last measured joint angles, canonical
        self.fault = None               # latched reason, or None
        self.was_damping = True

        self.create_subscription(Float64MultiArray, "joint_cmd", self._on_joint_cmd, SENSOR_QOS)
        self.joint_pub = self.create_publisher(JointState, "joint_states", SENSOR_QOS)
        self.contact_pub = self.create_publisher(Float64MultiArray, "foot_contacts", SENSOR_QOS)
        self.create_service(Trigger, "clear_fault", self._on_clear_fault)
        self.create_timer(1.0 / float(get("rate_hz")), self._cycle)
        self.get_logger().info(
            f"go1 bridge up: UDP {get('robot_ip')}:{get('robot_port')} at {get('rate_hz')} Hz, "
            f"power level {self.power_level}/10; damping until commands arrive")

    # -- ROS side -------------------------------------------------------------------------

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

    # -- one UDP cycle: state in, joint_states out, command out ---------------------------

    def _cycle(self):
        self.udp.Recv()
        self.udp.GetRecv(self.lowstate)
        motors = self.lowstate.motorState
        q = np.array([motors[int(i)].q for i in CANONICAL_TO_MOTOR])
        if not np.any(q):               # nothing from the robot yet: passive motors
            self.udp.SetSend(self.idle)
            self.udp.Send()
            return
        dq = np.array([motors[int(i)].dq for i in CANONICAL_TO_MOTOR])
        tau = np.array([motors[int(i)].tauEst for i in CANONICAL_TO_MOTOR])
        self.q = q
        self._check_limits(q)

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(CANONICAL_JOINT_NAMES)
        js.position, js.velocity, js.effort = q.tolist(), dq.tolist(), tau.tolist()
        self.joint_pub.publish(js)
        contacts = Float64MultiArray()
        contacts.data = [float(self.lowstate.footForce[int(k)]) for k in FOOT_FROM_UNITREE]
        self.contact_pub.publish(contacts)

        self._fill_command()
        self.safe.PowerProtect(self.lowcmd, self.lowstate, self.power_level)
        self.udp.SetSend(self.lowcmd)
        self.udp.Send()

    def _check_limits(self, q):
        lo = self.joint_range[:, 0] - self.limit_margin
        hi = self.joint_range[:, 1] + self.limit_margin
        bad = np.where((q < lo) | (q > hi))[0]
        if bad.size and self.fault is None:
            names = ", ".join(f"{CANONICAL_JOINT_NAMES[i]}={q[i]:.2f}" for i in bad)
            self.fault = f"joint past its limit: {names}"
            self.get_logger().error(f"damping latched: {self.fault}; /clear_fault to release")

    def _fill_command(self):
        fresh = self.cmd is not None and time.monotonic() - self.cmd_time < self.cmd_timeout
        damping = self.fault is not None or not fresh
        if damping != self.was_damping:
            self.get_logger().info("damping" if damping else "following joint_cmd")
            self.was_damping = damping
        if damping:
            q_des = np.full(NU, POS_STOP_F)     # no position target: pure damping
            dq_des, kp = np.zeros(NU), np.zeros(NU)
            kd, tau = np.full(NU, self.damp_kd), np.zeros(NU)
        else:
            q_des, dq_des, kp, kd, tau = self.cmd
            q_des = np.clip(q_des, self.joint_range[:, 0], self.joint_range[:, 1])
            kp = np.clip(kp, 0.0, self.kp_max)
            kd = np.clip(kd, 0.0, self.kd_max)
            tau = np.clip(tau, -self.tau_limit, self.tau_limit)
            # a joint with kp = 0 takes no position target (Unitree's stop value), so the
            # driver does not act on a stale q; likewise dq when kd = 0
            q_des = np.where(kp > 0, q_des, POS_STOP_F)
            dq_des = np.where(kd > 0, dq_des, VEL_STOP_F)
        for i, slot in enumerate(CANONICAL_TO_MOTOR):
            m = self.lowcmd.motorCmd[int(slot)]
            m.mode = MODE_SERVO
            m.q, m.dq = float(q_des[i]), float(dq_des[i])
            m.Kp, m.Kd, m.tau = float(kp[i]), float(kd[i]), float(tau[i])


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = None
    try:
        node = Go1Bridge()
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
