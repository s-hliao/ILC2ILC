#!/usr/bin/python3
"""Safely stand a Go1 and apply the simulated initial-stance perturbations.

This node commands the ROS 1 onboard bridge through /joint_cmd_r. It never starts
motion automatically: call /stand/start, then /stand/next for each perturbation.
The start service uses a sports-mode-like staged rise: capture the current folded
pose, move all four legs to a symmetric preload pose, and only then rise to the
simulation stance.  This is still a low-level joint-space test, not Unitree's
factory recovery controller; an operator must physically support the robot.
A joint-state outage over hold_state_timeout, stale mocap, excessive tilt/velocity,
or a /stand/damp request sends pure damping and latches the node until /stand/start
is explicitly called again.
"""

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.VOLATILE,
)

CANONICAL_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")
)
NU = len(CANONICAL_JOINT_NAMES)


def pack_joint_cmd(q, dq, kp, kd, tau):
    """Same field-major 5x12 command format used by sim_quad_model."""
    fields = [np.broadcast_to(np.asarray(v, dtype=float), (NU,)) for v in (q, dq, kp, kd, tau)]
    return np.concatenate(fields).tolist()

# Exact stand_offset cases in dilc_execute.py. Order is planar
# [front thigh, front calf, rear thigh, rear calf].
PERTURBATIONS = {
    "nominal": np.array([0.0, 0.0, 0.0, 0.0]),
    "crouch": np.array([0.1, -0.2, 0.1, -0.2]),
    "tall": np.array([-0.1, 0.2, -0.1, 0.2]),
    "noseup": np.array([0.0, 0.0, 0.1, -0.2]),
    "nosedn": np.array([0.1, -0.2, 0.0, 0.0]),
}

PLANAR_TO_CANONICAL = [
    [CANONICAL_JOINT_NAMES.index(f"{leg}_{joint}_joint") for leg in legs]
    for legs in (("FL", "FR"), ("RL", "RR")) for joint in ("thigh", "calf")
]


def expand(planar):
    full = np.zeros(NU)
    for value, indices in zip(np.asarray(planar, float), PLANAR_TO_CANONICAL):
        full[indices] = value
    return full


def quaternion_tilt(q):
    """Angle between body z and world z; independent of yaw."""
    norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
    if norm < 1e-9:
        return math.inf
    x, y, z, w = q.x/norm, q.y/norm, q.z/norm, q.w/norm
    body_z_world_z = 1.0 - 2.0 * (x*x + y*y)
    return math.acos(max(-1.0, min(1.0, body_z_world_z)))


class Go1StandPerturb(Node):
    def __init__(self):
        super().__init__("go1_stand_perturb")
        p = self.declare_parameter
        p("command_topic", "/joint_cmd_r")
        p("joint_topic", "/joint_states")
        p("pose_topic", "/mocap/pose")
        p("sequence", ["crouch", "tall", "noseup", "nosedn"])
        p("capture_time", 1.0)
        p("preload_time", 2.5)
        p("rise_time", 4.0)
        p("perturb_ramp_time", 2.0)
        p("capture_kp", 20.0)
        p("preload_kp", 35.0)
        p("kp", 60.0)
        p("kd", 4.0)
        p("damp_kd", 2.0)
        # ros1_bridge delivers the 500 Hz Go1 stream in short bursts.  Measured
        # inter-message gaps reach ~0.16 s even though the link averages 450 Hz.
        # The motor-side node still has its independent 50 ms command watchdog.
        p("state_timeout", 0.30)
        # While ramping or holding, the targets do not depend on /joint_states and
        # the onboard watchdog still guards the command link, so one bridge gap
        # must not drop a standing robot.  Only a longer outage damps.
        p("hold_state_timeout", 1.0)
        p("pose_timeout", 0.20)
        p("max_tilt", 0.35)
        p("max_joint_speed", 8.0)
        p("control_rate_hz", 250.0)
        get = lambda name: self.get_parameter(name).value

        self.sequence = list(get("sequence"))
        unknown = [name for name in self.sequence if name not in PERTURBATIONS or name == "nominal"]
        if unknown:
            raise ValueError(f"unknown stance perturbations: {unknown}; choices are {list(PERTURBATIONS)}")
        self.capture_time = float(get("capture_time"))
        self.preload_time = float(get("preload_time"))
        self.rise_time = float(get("rise_time"))
        self.perturb_ramp_time = float(get("perturb_ramp_time"))
        self.capture_kp = float(get("capture_kp"))
        self.preload_kp = float(get("preload_kp"))
        self.kp, self.kd, self.damp_kd = float(get("kp")), float(get("kd")), float(get("damp_kd"))
        self.state_timeout, self.pose_timeout = float(get("state_timeout")), float(get("pose_timeout"))
        self.hold_state_timeout = max(self.state_timeout, float(get("hold_state_timeout")))
        self.max_tilt, self.max_joint_speed = float(get("max_tilt")), float(get("max_joint_speed"))

        # Go1 home pose and per-motor static-gravity torques from the repository's
        # simulation plans. The jump controller uses the same values.
        self.q_home = expand([0.9, -1.8, 0.9, -1.8])
        # Unitree's low-level example uses [0, 1.2, -2.0] as its symmetric
        # preloaded pose.  Unlike that example, this node applies it to all legs
        # and does not execute the example's suspended-robot sine motion.
        self.q_preload = expand([1.2, -2.0, 1.2, -2.0])
        self.tau_stand = expand([-3.67781786, 5.71551808, -0.00173946, 4.26117797])

        self.q = self.dq = None
        self.joint_rx = self.pose_rx = 0.0
        self.tilt = math.inf
        self.mode = "damp"
        self.fault = "not started"
        self.index = -1
        self.active_name = "damp"
        self.ramp_from = self.ramp_to = None
        self.ramp_t0 = 0.0
        self.ramp_duration = 0.0
        self.kp_from = self.kp_to = 0.0
        self.tau_from = self.tau_to = np.zeros(NU)
        self.next_stage = None

        self.pub = self.create_publisher(Float64MultiArray, str(get("command_topic")), 1)
        self.create_subscription(JointState, str(get("joint_topic")), self.on_joints, SENSOR_QOS)
        self.create_subscription(PoseStamped, str(get("pose_topic")), self.on_pose, SENSOR_QOS)
        self.create_service(Trigger, "/stand/start", self.on_start)
        self.create_service(Trigger, "/stand/next", self.on_next)
        self.create_service(Trigger, "/stand/nominal", self.on_nominal)
        self.create_service(Trigger, "/stand/damp", self.on_damp)
        self.create_service(Trigger, "/stand/status", self.on_status)
        self.create_timer(1.0 / float(get("control_rate_hz")), self.tick)
        self.get_logger().warn(
            "DAMPED: support the robot, verify low-level mode and mocap, then call /stand/start")

    def on_joints(self, msg):
        if len(msg.position) != NU or len(msg.velocity) != NU:
            return
        if msg.name and list(msg.name) != list(CANONICAL_JOINT_NAMES):
            self.trip("joint_states order is not canonical")
            return
        self.q = np.asarray(msg.position, float)
        self.dq = np.asarray(msg.velocity, float)
        self.joint_rx = time.monotonic()

    def on_pose(self, msg):
        self.tilt = quaternion_tilt(msg.pose.orientation)
        self.pose_rx = time.monotonic()

    def sensors_ok(self, state_timeout=None):
        now = time.monotonic()
        state_timeout = self.state_timeout if state_timeout is None else state_timeout
        if self.q is None or now - self.joint_rx > state_timeout:
            return False, "joint state missing or stale"
        if now - self.pose_rx > self.pose_timeout:
            return False, "mocap pose missing or stale"
        if self.tilt > self.max_tilt:
            return False, f"trunk tilt {math.degrees(self.tilt):.1f} deg exceeds limit"
        if np.max(np.abs(self.dq)) > self.max_joint_speed:
            return False, f"joint speed {np.max(np.abs(self.dq)):.1f} rad/s exceeds limit"
        return True, "ready"

    def begin_ramp(self, name, target, duration, kp_from, kp_to, tau_from=None, tau_to=None,
                   next_stage=None):
        self.ramp_from = self.q.copy()
        self.ramp_to = np.asarray(target, float).copy()
        self.ramp_t0 = time.monotonic()
        self.ramp_duration = max(0.1, duration)
        self.kp_from, self.kp_to = float(kp_from), float(kp_to)
        self.tau_from = np.zeros(NU) if tau_from is None else np.asarray(tau_from, float).copy()
        self.tau_to = np.zeros(NU) if tau_to is None else np.asarray(tau_to, float).copy()
        self.next_stage = next_stage
        self.active_name = name
        self.mode = "ramp"
        self.fault = ""
        self.get_logger().warn(
            f"stand stage {name}: smooth ramp over {self.ramp_duration:.1f} s "
            f"(Kp {self.kp_from:.0f}->{self.kp_to:.0f})")

    def begin_preload(self):
        self.begin_ramp("preload", self.q_preload, self.preload_time,
                        self.capture_kp, self.preload_kp, next_stage=self.begin_rise)

    def begin_rise(self):
        self.begin_ramp("rise", self.q_home, self.rise_time,
                        self.preload_kp, self.kp, tau_to=self.tau_stand,
                        next_stage=self.finish_rise)

    def finish_rise(self):
        self.active_name = "nominal"
        self.mode = "hold"
        self.ramp_to = self.q_home.copy()
        self.kp_to = self.kp
        self.tau_to = self.tau_stand.copy()
        self.next_stage = None
        self.get_logger().warn(
            "holding nominal; keep supporting until stable, then use /stand/next")

    def on_start(self, request, response):
        ok, reason = self.sensors_ok()
        if not ok:
            response.success, response.message = False, f"staying damped: {reason}"
            return response
        self.index = -1
        # First stiffen at the measured pose.  Starting a position trajectory
        # immediately from a collapsed/asymmetric pose caused the first hardware
        # test to stall halfway through the rise.
        self.begin_ramp("capture", self.q, self.capture_time, 0.0, self.capture_kp,
                        next_stage=self.begin_preload)
        response.success, response.message = True, "started capture -> preload -> rise sequence"
        return response

    def on_next(self, request, response):
        if self.mode not in ("hold",):
            response.success, response.message = False, f"not holding safely (mode={self.mode})"
            return response
        ok, reason = self.sensors_ok()
        if not ok:
            # Refuse without damping; tick() still damps on a real outage.
            response.success, response.message = False, f"still holding, not advancing: {reason}"
            return response
        if self.index + 1 >= len(self.sequence):
            response.success, response.message = False, "end of perturbation sequence; call /stand/nominal"
            return response
        self.index += 1
        name = self.sequence[self.index]
        self.begin_ramp(name, self.q_home + expand(PERTURBATIONS[name]),
                        self.perturb_ramp_time, self.kp, self.kp,
                        self.tau_stand, self.tau_stand)
        response.success, response.message = True, f"ramping to {name}"
        return response

    def on_nominal(self, request, response):
        ok, reason = self.sensors_ok()
        if self.mode == "damp" or not ok:
            response.success, response.message = False, f"call /stand/start first ({reason})"
            return response
        self.begin_ramp("nominal", self.q_home, self.perturb_ramp_time,
                        self.kp, self.kp, self.tau_stand, self.tau_stand)
        response.success, response.message = True, "returning to nominal stand"
        return response

    def on_damp(self, request, response):
        self.trip("operator requested damping")
        response.success, response.message = True, "damping latched"
        return response

    def on_status(self, request, response):
        ok, reason = self.sensors_ok()
        response.success = self.mode != "damp" and ok
        response.message = (f"mode={self.mode}, target={self.active_name}, tilt={math.degrees(self.tilt):.1f} deg, "
                            f"sensors={reason}, fault={self.fault or 'none'}")
        return response

    def trip(self, reason):
        if self.mode != "damp" or self.fault != reason:
            self.get_logger().error(f"DAMPING LATCHED: {reason}")
        self.mode, self.fault, self.active_name = "damp", reason, "damp"

    def publish_damp(self):
        msg = Float64MultiArray()
        q = self.q if self.q is not None else np.zeros(NU)
        msg.data = pack_joint_cmd(q, 0.0, 0.0, self.damp_kd, 0.0)
        self.pub.publish(msg)

    def tick(self):
        if self.mode == "damp":
            self.publish_damp()
            return
        ok, reason = self.sensors_ok(self.hold_state_timeout)
        if not ok:
            self.trip(reason)
            self.publish_damp()
            return
        if self.mode == "ramp":
            a = min((time.monotonic() - self.ramp_t0) / self.ramp_duration, 1.0)
            # Cubic smoothstep gives zero target velocity at both ends.
            s = a*a*(3.0 - 2.0*a)
            q_des = (1.0 - s) * self.ramp_from + s * self.ramp_to
            kp = (1.0 - s) * self.kp_from + s * self.kp_to
            tau = (1.0 - s) * self.tau_from + s * self.tau_to
            if a >= 1.0:
                continuation = self.next_stage
                if continuation is not None:
                    continuation()
                else:
                    self.mode = "hold"
                    self.get_logger().info(
                        f"holding {self.active_name}; call /stand/next, /stand/nominal, or /stand/damp")
        else:
            q_des = self.ramp_to
            kp = self.kp_to
            tau = self.tau_to
        msg = Float64MultiArray()
        msg.data = pack_joint_cmd(q_des, 0.0, kp, self.kd, tau)
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Go1StandPerturb()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Keep sending damping briefly so the onboard watchdog takes over cleanly.
        if rclpy.ok():
            for _ in range(25):
                node.publish_damp()
                rclpy.spin_once(node, timeout_sec=0.004)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
