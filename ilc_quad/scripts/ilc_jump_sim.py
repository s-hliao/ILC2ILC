#!/usr/bin/env python3
"""ILC jumping node: flies the jump, learns the contact forces trial to trial.

Runs the paper's pipeline (`ilc_gen.JumpILC`) against a live robot: the full-body TO
gives the reference and trial-1 forces, each trial is flown with the force-based
low-level controller, and between trials the 3-stage ILC update moves the forces.

    backend:=sim   drives `sim_node` (MuJoCo Go2, optional box). Trials reset and start
                   on their own.
    backend:=go1   drives a real Go1 through `go1_bridge` (unitree_legged_sdk), and
    backend:=go2   a real Go2 through `go2_bridge` (unitree_ros2), with the trunk
                   pose from your OptiTrack node. Every trial waits for an
                   operator: `ros2 service call /start_trial std_srvs/srv/Trigger`.

Both backends speak the same topics, so this node does not know which it is on
beyond how trials start and where the trunk pose comes from:

    subscribes  joint_states   JointState, canonical order (sim_node / go2_bridge)
                <pose_topic>   Odometry or PoseStamped of the trunk: `base_odom` from
                               sim_node, or the mocap topic on the real robot
    publishes   joint_cmd      Float64MultiArray, [q, dq, kp, kd, tau] x 12 -- executed
                               as tau + kp (q - q_meas) + kd (dq - dq_meas) at the motors
    services    start_trial    Trigger: fly the next trial (go2; sim starts by itself)
                damp           Trigger: stop, zero stiffness, joint damping only
    calls       reset_trial    sim only, before every trial

Low-level controller during the jump: feedforward torque = the TO's joint torque
+ J(q)^T R(theta)^T (U - u_TO) at the measured configuration, and a joint PD tracking
the TO's joint trajectory. The ILC learns contact forces as in the paper (Appendix
V.B), whose feedforward is J^T R^T f alone; the TO torque is added because J^T f is
only the static part of what the full-body plan needs (nothing for the swinging
front leg, a third of the rear knee's at takeoff), so without it trial 1 does not
leave the ground. Hip abduction is held at 0.

A trial: stand (ramp to the home pose, then settle) -> jump (N samples of the TO
grid) -> land (hold the home pose) -> update (ILC, in a worker thread while the
robot keeps standing). The recorded trial is resampled onto the TO grid and
expressed in the TO's frame: CoM relative to where it stood at the jump's start,
along the heading it faced then.

Safety on the real robot, beyond go2_bridge's own watchdog and joint-limit latch:
the node damps (and waits for `start_trial`) if the trunk pose or joint states go
stale, or if the trunk tilts past `max_tilt` outside the jump itself.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from ilc_quad.ilc_gen import JumpILC, describe_result
from ilc_quad.sim_quad_model import (
    CANONICAL_JOINT_NAMES,
    NU,
    QuadModel,
    default_menagerie_root,
    pack_joint_cmd,
)

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.VOLATILE,
)

# canonical (12,) indices of the planar joints [F thigh, F calf, R thigh, R calf]:
# both legs of a pair per planar joint, hip abduction separately
PLANAR_TO_CANONICAL = [
    [CANONICAL_JOINT_NAMES.index(f"{leg}_{j}_joint") for leg in legs]
    for legs in (("FL", "FR"), ("RL", "RR")) for j in ("thigh", "calf")
]
HIP_IDX = [CANONICAL_JOINT_NAMES.index(f"{leg}_hip_joint") for leg in ("FL", "FR", "RL", "RR")]


def expand(planar) -> np.ndarray:
    """Planar (4,) joint quantity -> canonical (12,), both legs of a pair alike, hips 0."""
    full = np.zeros(NU)
    for i, idx in enumerate(PLANAR_TO_CANONICAL):
        full[idx] = planar[i]
    return full


def collapse(full) -> np.ndarray:
    """Canonical (12,) -> planar (4,), the mean over each leg pair."""
    return np.array([np.mean(np.asarray(full)[idx]) for idx in PLANAR_TO_CANONICAL])


def quat_rotate(q, v):
    """Rotate v by the unit quaternion q = (w, x, y, z)."""
    w, u = q[0], np.asarray(q[1:])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def planar_pitch(q):
    """Nose-up pitch (the planar model's theta) from a (w, x, y, z) quaternion."""
    w, x, y, z = q
    return -math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def tilt_of(q):
    """Angle between the trunk's z axis and the world's."""
    zb = quat_rotate(q, np.array([0.0, 0.0, 1.0]))
    return math.acos(max(-1.0, min(1.0, zb[2])))


def stamp_sec(stamp) -> float:
    return stamp.sec + 1e-9 * stamp.nanosec


class IlcJumpNode(Node):
    def __init__(self):
        super().__init__("ilc_jump")

        p = self.declare_parameter
        p("backend", "sim")                       # sim | go1 | go2 (the last two: hardware)
        p("robot", "go2")
        p("menagerie_root", default_menagerie_root())
        p("control_rate_hz", 500.0)               # rate of joint_states; one command each
        # task: CoM displacement standing -> landing, and the box (height <= 0: none),
        # x_front measured from the standing CoM along the robot's heading
        p("jump_dx", 0.50)
        p("jump_dz", 0.20)
        p("box_x_front", 0.25)
        p("box_height", 0.20)
        p("phases", [20, 20, 25])                 # Ndc, Nsc, Nfl on the TO grid
        p("dt", 0.01)                             # TO grid
        p("margin", 0.8)                          # share of each hard limit the TO may use
        p("swing_qd", 10.0)                       # TO: soft speed limit on swing legs, rad/s
        p("w_swing", 1.0)                         # TO: weight on swing speed above it
        p("qu", 3e-5)                             # ILC step penalty (Qu); larger = gentler
        # Qu in Stage III (<= 0: the same as qu). Stage III weighs only the landing state, so
        # the Stage I-II step overshoots there and the landing drifts off over trials; 10x
        # the step penalty holds it (MuJoCo sweep over the paper's tasks, 2026-09)
        p("qu_stage3", 3e-4)
        # ILC error weights Qe on [x, z, theta, vx, vz, omega]. The paper's diag{1,3,3,...}
        # prices a degree of landing pitch like 9 cm of distance; x weighted as z did better
        # on Go1's 40/60 cm jumps and Go2's flat ones (Go2's (60,10) box preferred the paper's)
        p("qe", [3.0, 3.0, 3.0, 0.01, 0.01, 0.01])
        p("n_stage1", 5)
        p("n_stage2", 5)
        p("max_trials", 25)
        p("pos_tol", 0.01)
        p("theta_tol_deg", 1.0)
        # files: the TO is loaded from reference_file if it exists, else solved and saved
        # there; log_dir gets one npz per trial; resume_file restarts learning from a log
        p("reference_file", "")
        p("log_dir", "")
        p("resume_file", "")
        # low-level gains (per motor) and timing
        p("stand_kp", 60.0)
        p("stand_kd", 5.0)
        p("jump_kp", 30.0)
        p("jump_kd", 1.0)
        # legs in contact during the jump: force control -- the ILC's forces act through
        # J^T only if no stiff joint PD fights them (see _tick_jump)
        p("contact_kp", 0.0)
        p("contact_kd", 1.0)
        # jump feedforward: to_torque = TO joint torque + J^T R^T (U - u_TO);
        # force = J^T R^T U only, the paper's force-based law (Appendix V.B)
        p("feedforward", "to_torque")
        p("land_kp", 60.0)
        p("land_kd", 5.0)
        p("damp_kd", 2.0)
        p("stand_time", 1.0)                      # ramp to the home pose
        p("settle_time", 0.5)                     # then hold before jumping
        p("land_time", 1.5)                       # hold after the TO horizon
        # trunk pose: sim_node's base_odom, or the mocap topic on the real robot
        p("pose_topic", "")
        p("pose_type", "")                        # odometry | pose
        p("mocap_offset", [0.0, 0.0, 0.0])        # tracked point in the base frame
        p("auto_start", True)                     # sim only; go2 always waits
        p("exit_when_done", False)                # sim runs: exit once learning stops
        # safety (go2 especially)
        p("max_tilt", 0.6)                        # rad, outside the jump itself
        p("state_timeout", 0.05)                  # s without joint_states or pose

        get = lambda name: self.get_parameter(name).value
        self.backend = get("backend")
        if self.backend not in ("sim", "go1", "go2"):
            raise ValueError(f"backend must be sim, go1 or go2, got {self.backend!r}")
        if self.backend != "sim" and get("robot") != self.backend:
            raise ValueError(f"backend {self.backend} flies robot {self.backend}, "
                             f"but robot is {get('robot')!r}: the plan would be for another robot")
        self.is_sim = self.backend == "sim"
        self.auto_start = self.is_sim and bool(get("auto_start"))
        self.exit_when_done = self.is_sim and bool(get("exit_when_done"))

        # ---- the ILC core: TO reference (solved or loaded) and the learning state ----
        box = None
        if float(get("box_height")) > 0:
            box = dict(x_front=float(get("box_x_front")), height=float(get("box_height")))
        self.qm = QuadModel(get("robot"), get("menagerie_root"))
        ref_file = get("reference_file")
        load = ref_file if ref_file and os.path.exists(ref_file) else None
        make_ilc = lambda reference: JumpILC(
            self.qm, jump=(float(get("jump_dx")), float(get("jump_dz"))), box=box,
            phases=tuple(int(n) for n in get("phases")), dt=float(get("dt")),
            n_stage1=int(get("n_stage1")), n_stage2=int(get("n_stage2")),
            pos_tol=float(get("pos_tol")), theta_tol=math.radians(float(get("theta_tol_deg"))),
            margin=float(get("margin")), swing_qd=float(get("swing_qd")),
            w_swing=float(get("w_swing")), Qu_diag=float(get("qu")),
            Qe_diag=[float(v) for v in get("qe")],
            Qu_stage3=float(get("qu_stage3")) if float(get("qu_stage3")) > 0 else None,
            reference=reference)
        self.ilc = None
        if load:
            try:
                self.ilc = make_ilc(load)
                self.get_logger().info(f"loaded the TO reference from {load}")
            except ValueError as err:     # made for another task or TO setting
                self.get_logger().warn(f"{err}; solving a new one")
                load = None
        if self.ilc is None:
            self.get_logger().info("solving the full-body TO (tens of seconds)...")
            self.ilc = make_ilc(None)
        self.get_logger().info(self.ilc.describe_reference())
        if not self.ilc.to_info["success"]:
            self.get_logger().warn("the TO did not converge; its last iterate is the reference")
        if ref_file and load is None:
            self.ilc.save_reference(ref_file)
            self.get_logger().info(f"reference saved to {ref_file}")
        resume = get("resume_file")
        if resume:
            data = np.load(resume)
            self.ilc.U = data["U_next"].copy()
            self.ilc.trial = int(data["trial"])
            self.get_logger().info(f"resumed from {resume}: next trial {self.ilc.trial + 1}")
        self.max_trials = int(get("max_trials"))
        self.log_dir = get("log_dir")
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

        # ---- timing on the TO grid ----
        self.tick = 1.0 / float(get("control_rate_hz"))
        self.T_jump = self.ilc.N * self.ilc.dt

        fb = self.ilc.fb
        self.fb = fb
        self.tau_limit = self.qm.torque_limit
        self.q_stand = expand(fb.q_home)
        # standing feedforward: the weight shared by the two leg pairs
        f_static = np.array([0.0, 0.5, 0.0, 0.5]) * fb.total_mass * fb.g
        self.tau_stand = expand(fb.torque_map(fb.q_home, 0.0) @ f_static)
        self.gains = {k: (float(get(f"{k}_kp")), float(get(f"{k}_kd")))
                      for k in ("stand", "jump", "land", "contact")}
        self.damp_kd = float(get("damp_kd"))
        self.stand_time, self.settle_time = float(get("stand_time")), float(get("settle_time"))
        self.land_time = float(get("land_time"))
        self.max_tilt = float(get("max_tilt"))
        self.feedforward = get("feedforward")
        if self.feedforward not in ("to_torque", "force"):
            raise ValueError(f"feedforward must be to_torque or force, got {self.feedforward!r}")
        self.state_timeout = float(get("state_timeout"))
        self.mocap_offset = np.asarray(get("mocap_offset"), float)

        # ---- I/O ----
        pose_topic = get("pose_topic") or ("base_odom" if self.is_sim else "")
        pose_type = get("pose_type") or "odometry"
        if not pose_topic:
            raise ValueError(f"backend {self.backend} needs pose_topic: the OptiTrack trunk pose")
        msg_type = {"odometry": Odometry, "pose": PoseStamped}[pose_type]
        self.pose = None                      # (receipt time, position (3,), quat (w,x,y,z))
        self.create_subscription(msg_type, pose_topic, self._on_pose, SENSOR_QOS)
        self.create_subscription(JointState, "joint_states", self._on_joint_states, SENSOR_QOS)
        self.contacts = np.full(4, np.nan)    # latest foot normal forces FL FR RL RR (logged)
        self.create_subscription(Float64MultiArray, "foot_contacts", self._on_contacts, SENSOR_QOS)
        self.cmd_pub = self.create_publisher(Float64MultiArray, "joint_cmd", SENSOR_QOS)
        self.create_service(Trigger, "start_trial", self._on_start_trial)
        self.create_service(Trigger, "damp", self._on_damp)
        self.reset_client = self.create_client(Trigger, "reset_trial") if self.is_sim else None

        self.phase = "idle"                   # idle | reset | stand | jump | land | update | done | damp
        self.phase_t0 = None                  # state time the phase began
        self.last_js_wall = None
        self.stood = False                    # whether idle holds a stand or sends nothing
        self.fell = False                     # this trial's landing failed: damp until reset
        self.worker = None
        self.get_logger().info(
            f"backend {self.backend}: pose from {pose_topic} ({pose_type}), "
            f"{self.ilc.N} samples x {self.ilc.dt*1e3:.0f} ms, "
            f"{'trials start on their own' if self.auto_start else 'call /start_trial for each trial'}")

    # -- inputs ---------------------------------------------------------------------------

    def _on_pose(self, msg):
        pose = msg.pose.pose if isinstance(msg, Odometry) else msg.pose
        quat = np.array([pose.orientation.w, pose.orientation.x, pose.orientation.y,
                         pose.orientation.z])
        pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        pos = pos - quat_rotate(quat, self.mocap_offset)       # tracked point -> base origin
        self.pose = (time.monotonic(), pos, quat)

    def _on_start_trial(self, request, response):
        if self.phase not in ("idle", "damp", "done"):
            response.success, response.message = False, f"busy ({self.phase})"
        elif self.ilc.trial >= self.max_trials:
            response.success, response.message = False, f"max_trials ({self.max_trials}) reached"
        else:
            self._begin_trial()
            response.success, response.message = True, f"trial {self.ilc.trial + 1} starting"
        return response

    def _on_damp(self, request, response):
        self._to_damp("damp requested")
        response.success, response.message = True, "damping"
        return response

    def _on_contacts(self, msg):
        if len(msg.data) == 4:
            self.contacts = np.asarray(msg.data, float)

    def _on_joint_states(self, msg: JointState):
        if len(msg.position) != NU or len(msg.velocity) != NU:
            return
        if msg.name and list(msg.name) != list(CANONICAL_JOINT_NAMES):
            self.get_logger().error("joint_states is not in canonical order", once=True)
            return
        now = stamp_sec(msg.header.stamp)
        q, dq = np.asarray(msg.position), np.asarray(msg.velocity)
        # applied torque as the robot reports it (sim: post-clamp; Go2: tau_est), logged
        self.effort = np.asarray(msg.effort) if len(msg.effort) == NU else np.full(NU, np.nan)
        self.last_js_wall = time.monotonic()
        if self.phase_t0 is None:
            self.phase_t0 = now

        if not self._inputs_fresh():
            if self.phase in ("stand", "jump", "land", "update"):
                self._to_damp("joint states or trunk pose went stale")
            return
        if self.phase in ("stand", "land", "update", "idle") and self.stood and not self.fell:
            tilt = tilt_of(self.pose[2])
            if tilt > self.max_tilt and self.phase in ("land", "update"):
                # a failed landing: go limp, but the jump itself was recorded, and a bad
                # early trial is what the ILC learns from -- the update still runs
                self.fell = True
                self.get_logger().warn(
                    f"landing failed (trunk tilted {math.degrees(tilt):.0f} deg): damping")
            elif tilt > self.max_tilt:
                self._to_damp(f"trunk tilted {math.degrees(tilt):.0f} deg")
                return

        try:
            cmd = getattr(self, f"_tick_{self.phase}")(now, q, dq)
        except Exception as err:          # a bug mid-jump must not leave the last command held
            self.get_logger().error(f"{self.phase} tick failed: {err!r}")
            self._to_damp("controller error")
            cmd = self._tick_damp(now, q, dq)
        if cmd is not None:
            msg_out = Float64MultiArray()
            msg_out.data = pack_joint_cmd(*cmd)
            self.cmd_pub.publish(msg_out)

    def _inputs_fresh(self):
        if self.pose is None:
            return False
        return time.monotonic() - self.pose[0] < self.state_timeout or self.is_sim

    # -- phases ---------------------------------------------------------------------------

    def _set_phase(self, phase, now=None):
        self.phase = phase
        self.phase_t0 = now

    def _begin_trial(self):
        if self.is_sim:
            if not self.reset_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().error("reset_trial service not available")
                return
            self._set_phase("reset")
            future = self.reset_client.call_async(Trigger.Request())
            future.add_done_callback(lambda f: self._set_phase("stand"))
        else:
            self._set_phase("stand")
        self.stand_from = None
        self.get_logger().info(f"trial {self.ilc.trial + 1} (stage {self.ilc.stage()}): standing up")

    def _to_damp(self, reason):
        if self.phase != "damp":
            self.get_logger().warn(f"damping: {reason}")
        self._set_phase("damp")
        self.stood = False

    def _hold(self, kind):
        kp, kd = self.gains[kind]
        return self.q_stand, np.zeros(NU), kp, kd, self.tau_stand

    def _tick_idle(self, now, q, dq):
        if self.auto_start and self.ilc.trial < self.max_trials and self.worker is None:
            self._begin_trial()
            return None
        return self._hold("stand") if self.stood else None

    def _tick_reset(self, now, q, dq):
        return None

    def _tick_damp(self, now, q, dq):
        return q, np.zeros(NU), 0.0, self.damp_kd, 0.0

    def _tick_done(self, now, q, dq):
        return self._hold("stand") if self.stood else self._tick_damp(now, q, dq)

    def _tick_stand(self, now, q, dq):
        if self.stand_from is None:          # first tick after reset / start: ramp from here
            self.stand_from = q.copy()
            self.phase_t0 = now
        t = now - self.phase_t0
        a = min(t / self.stand_time, 1.0) if self.stand_time > 0 else 1.0
        kp, kd = self.gains["stand"]
        self.stood = True
        if t >= self.stand_time + self.settle_time:
            self._start_jump(now)
            return self._tick_jump(now, q, dq)
        q_des = (1 - a) * self.stand_from + a * self.q_stand
        return q_des, np.zeros(NU), a * kp, kd, a * self.tau_stand

    def _start_jump(self, now):
        self._set_phase("jump", now)
        _, pos, quat = self.pose
        self.frame = dict(origin=pos.copy(), yaw=yaw_of(quat))
        self.rec = dict(t=[], pos=[], quat=[], q=[], dq=[], tau_pd=[], cmd_tau=[], q_des=[],
                        dq_des=[], tau_total=[], effort=[], contacts=[])
        self.U_flown = self.ilc.U.copy()
        self.fell = False
        self.get_logger().info(f"trial {self.ilc.trial + 1}: jump")

    def _tick_jump(self, now, q, dq):
        # on the control tick grid: state stamps (ns-rounded on ROS, accumulated floats in
        # sim) put a tick that falls on a TO sample boundary a hair to either side of it,
        # and the force sample it picks -- held for the whole sample -- then shifts a tick
        # early or late from trial to trial, which reads as trial-to-trial noise
        t = round((now - self.phase_t0) / self.tick) * self.tick
        if t >= self.T_jump:
            self._set_phase("land", now)
            return self._tick_land(now, q, dq)
        ilc, dt = self.ilc, self.ilc.dt
        k = min(int(t / dt + 1e-6), ilc.N - 1)
        a = t / dt - k
        q_des = expand((1 - a) * ilc.to_info["q_ref"][k] + a * ilc.to_info["q_ref"][k + 1])
        dq_des = expand((1 - a) * ilc.to_info["qd_ref"][k] + a * ilc.to_info["qd_ref"][k + 1])

        # feedforward: the TO's joint torque, plus the ILC's force correction through
        # J(q)^T R(theta)^T at the measured configuration; both held over each TO sample
        # like the SRB model assumes. J^T f alone is only the static part of what the plan
        # needs -- it has no torque for swinging the front leg after liftoff, and a third of
        # the rear knee's at takeoff -- so trial 1 flies the TO's torque exactly.
        # (feedforward:=force drops the TO torque and flies J^T R^T U alone, as in the paper.)
        theta = planar_pitch(self.pose[2])
        tau_to = expand(ilc.to_info["tau"][k] / 2.0)          # pair total -> per motor
        tau_ilc = np.zeros(NU)
        if k < ilc.Nc:                      # by sample, not time: k rounds to the grid
            T = self.fb.torque_map(collapse(q), theta)
            tau_ilc = expand(T @ self.U_flown[k])
            tau_to = tau_to + tau_ilc - expand(T @ ilc.u_ref[k])
        if self.feedforward == "force":
            tau_to = tau_ilc
        tau_ff = np.clip(tau_to, -self.tau_limit, self.tau_limit)

        # joint PD on swing and flight legs; a leg in contact is force-controlled (the
        # feedforward above) with damping only. A stiff PD on a planted leg pins the trunk
        # to the planned joint angles and cancels any force the ILC adds -- the learned
        # forces then do not act as its SRB model predicts, and it diverges.
        kp, kd = np.full(NU, self.gains["jump"][0]), np.full(NU, self.gains["jump"][1])
        if k < ilc.Nc:
            for pair, joints in ((0, PLANAR_TO_CANONICAL[0:2]), (1, PLANAR_TO_CANONICAL[2:4])):
                if not ilc.swing_mask[k, 2 * pair]:
                    for idx in joints:
                        kp[idx], kd[idx] = self.gains["contact"]
        kp[HIP_IDX], kd[HIP_IDX] = self.gains["stand"]
        # everything the motor carries besides J^T f of the ILC's forces: the ILC's torque
        # limits apply to  T u + this  (QuadILCStageSolver.solve's tau_pd_k)
        tau_pd = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff - tau_ilc

        _, pos, quat = self.pose
        self.rec["t"].append(t)
        self.rec["pos"].append(pos.copy())
        self.rec["quat"].append(quat.copy())
        self.rec["q"].append(q.copy())
        self.rec["dq"].append(dq.copy())
        self.rec["tau_pd"].append(tau_pd)
        self.rec["cmd_tau"].append(tau_ff)
        self.rec["q_des"].append(q_des)
        self.rec["dq_des"].append(dq_des)
        self.rec["tau_total"].append(kp * (q_des - q) + kd * (dq_des - dq) + tau_ff)
        self.rec["effort"].append(self.effort.copy())
        self.rec["contacts"].append(self.contacts.copy())
        return q_des, dq_des, kp, kd, tau_ff

    def _tick_land(self, now, q, dq):
        if self.fell or now - self.phase_t0 >= self.land_time:
            self._set_phase("update", now)
            self.worker = threading.Thread(target=self._update_worker, daemon=True)
            self.worker.start()
        return self._tick_damp(now, q, dq) if self.fell else self._hold("land")

    def _tick_update(self, now, q, dq):
        if self.worker is not None and not self.worker.is_alive():
            self.worker = None
            r = self.last_result
            done = r["converged"] or r.get("failed") or self.ilc.trial >= self.max_trials
            self._set_phase("done" if done else "idle", now)
            if done:
                self.get_logger().info(
                    "target reached" if r["converged"] else "stopped: the ILC update failed"
                    if r.get("failed") else f"max_trials ({self.max_trials}) reached")
                if self.exit_when_done:
                    raise SystemExit(0)
            elif not self.auto_start:
                self.get_logger().info(f"ready: call /start_trial for trial {self.ilc.trial + 1}")
            if self.fell:                     # idle then sends nothing: the bridge damps
                self.stood = False
        return self._tick_damp(now, q, dq) if self.fell else self._hold("land")

    # -- learning -------------------------------------------------------------------------

    def _trial_log(self):
        """The recorded trial on the TO grid, in the TO's frame (see JumpILC.update)."""
        ilc, fb = self.ilc, self.fb
        rec = {k: np.asarray(v) for k, v in self.rec.items()}
        t = rec["t"]
        c, s = math.cos(self.frame["yaw"]), math.sin(self.frame["yaw"])
        rel = rec["pos"] - self.frame["origin"]
        fwd = c * rel[:, 0] + s * rel[:, 1]                 # along the heading at jump start
        theta = np.array([planar_pitch(qq) for qq in rec["quat"]])
        q_planar = np.array([collapse(qq) for qq in rec["q"]])
        dq_planar = np.array([collapse(v) for v in rec["dq"]])
        states = [np.concatenate([[x, z, th], qp])
                  for x, z, th, qp in zip(fwd, rel[:, 2], theta, q_planar)]
        com = np.array([fb.com(st).full().ravel() for st in states])
        shift = ilc.x_ref[0, :2] - com[0]                   # start where the reference starts
        com = com + shift
        feet = np.array([fb.feet(st).full().reshape(2, 2) for st in states]) + shift
        comd = np.gradient(com, t, axis=0)
        thetad = np.gradient(theta, t)

        grid = np.arange(ilc.N + 1) * ilc.dt
        grid[-1] = min(grid[-1], t[-1])                     # last tick may fall just short
        at = lambda y: np.interp(grid, t, y)
        X = np.column_stack([at(com[:, 0]), at(com[:, 1]), at(theta),
                             at(comd[:, 0]), at(comd[:, 1]), at(thetad)])
        Nc = ilc.Nc
        per_motor_pd = np.array([collapse(v) for v in rec["tau_pd"]])
        log = dict(
            X=X,
            q=np.column_stack([at(q_planar[:, i]) for i in range(4)])[:Nc],
            theta=X[:Nc, 2],
            qdot=np.column_stack([at(dq_planar[:, i]) for i in range(4)])[:Nc],
            tau_pd=np.column_stack([at(per_motor_pd[:, i]) for i in range(4)])[:Nc],
        )
        return log, rec, self._measured_clearance(t, feet)

    def _measured_clearance(self, t, feet):
        """
        Smallest foot height above the terrain in flight (samples Nc+1 .. N-1, like
        QuadILCStageSolver.min_clearance) from forward kinematics of the measured joints
        and trunk, rather than the TO's leg motion carried on the body.
        """
        ilc = self.ilc
        flight = (t > (ilc.Nc + 1) * ilc.dt - 1e-9) & (t < (ilc.N - 1) * ilc.dt + 1e-9)
        worst = np.inf
        box = ilc.box
        for xz in feet[flight].reshape(-1, 2):
            terrain = box["height"] if box is not None and xz[0] >= box["x_front"] else 0.0
            worst = min(worst, xz[1] - self.fb.foot_radius - terrain)
        return float(worst)

    def _update_worker(self):
        try:
            log, rec, clearance = self._trial_log()
            trial = self.ilc.trial + 1
            result = self.ilc.update(log)
            result["clearance"] = clearance           # measured, not the TO's legs
            self.last_result = result
            self.get_logger().info(describe_result(result))
            if self.log_dir:
                path = os.path.join(self.log_dir, f"trial_{trial:03d}.npz")
                np.savez(path, U_flown=self.U_flown, U_next=self.ilc.U, trial=self.ilc.trial,
                         x_ref=self.ilc.x_ref, result=json.dumps(result, default=float),
                         **{f"log_{k}": v for k, v in log.items()},
                         **{f"rec_{k}": v for k, v in rec.items()})
        except Exception as err:          # never leave the robot in 'update'
            self.get_logger().error(f"ILC update failed: {err!r}")
            self.last_result = dict(converged=False, failed=True)


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = None
    try:
        node = IlcJumpNode()
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
