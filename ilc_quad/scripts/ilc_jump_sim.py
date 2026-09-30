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
grid) -> land (from touchdown: the balance controller, a trunk PD on pitch, height and
fore-aft position through J^T with every foot kept pushing down; or
landing_controller:=pd, a joint PD holding the home pose) -> update (ILC, in a worker thread while the
robot keeps standing). The recorded trial is resampled onto the TO grid and
expressed in the TO's frame: CoM relative to where it stood at the jump's start,
along the heading it faced then.

Safety on the real robot, beyond go2_bridge's own watchdog and joint-limit latch:
the node damps (and waits for `start_trial`) if the trunk pose or joint states go
stale, or if the trunk tilts past `max_tilt` outside the jump itself.
"""

from __future__ import annotations

import math
import os
import threading
import time

import casadi as ca
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
from ilc_quad.trial_log import TrialRecorder
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
        p("phases", [30, 30, 30])                 # Ndc, Nsc, Nfl on the TO grid
        p("dt", 0.01)                             # TO grid
        p("margin", 0.85)                         # share of each hard limit the TO may use
        p("swing_qd", 10.0)                       # TO: soft speed limit on swing legs, rad/s
        p("w_swing", 1.0)                         # TO: weight on swing speed above it
        p("box_clearance", 0.04)                  # TO: moving feet clear the box's front edge
        p("box_setback", 0.03)                    #     by this much, when this close to it
        p("to_guess", "srb")                      # TO: start from an SRB plan (srb) or a line
        p("landing_pitch_rate_deg", 90.0)         # TO: |pitch rate| at touchdown; <= 0: free
        p("flight_pitch_rate_deg", 0.0)           # TO: |pitch rate| takeoff to touchdown; <= 0: free
        p("flight_min_pitch_deg", 0.0)            # TO: lowest pitch in flight (nose up +); <= -90: free
        p("flight_spin_change_deg", 20.0)         # TO: pitch-rate spread allowed in flight; <= 0: free
        p("leg_clearance", 0.10)                  # TO: m between the front and rear legs; <= 0: off
        p("landing_clearance", 0.06)              # TO: m feet stay above the landing late in flight; <= 0: off
        p("mu", 0.6)                              # ground friction the plan, ILC and landing assume
        p("to_steps", 1)                          # TO: >1 = box plans by continuation in height
        p("qu", 1e-4)                             # ILC step penalty (Qu); larger = gentler
        # Qu in Stage III (<= 0: the same as qu). Stage III is safeguarded (JumpILC.update: a
        # step that lands worse is undone and retried shorter), so it can step freely; at
        # the old 3e-4 its steps were too short to lift a 1-2 cm landing shortfall
        p("qu_stage3", 1e-5)
        # ILC error weights Qe on [x, z, theta, vx, vz, omega]. The paper's diag{1,3,3,...}
        # prices a degree of landing pitch like 9 cm of distance, and left Go1's 60 cm jump
        # 1.4 cm short; x weighted as z passed the whole Go1 validation suite (2026-09-30)
        p("qe", [3.0, 3.0, 3.0, 0.01, 0.01, 0.01])
        # Stage III safeguard (JumpILC.update): a trial landing worse than the best, or
        # falling, is rolled back to the best trial with a shorter step. False: the paper's
        # law, stepping from every trial -- falls included, their jump window is still data
        p("stage3_safeguard", True)
        # Stage III plans with the landing sensitivities corrected from the trials flown
        # (JumpILC secant: Broyden updates of the SRB model's G)
        p("stage3_secant", False)
        # ILC step variants (JumpILC; the defaults are the law above)
        # learning gain: U <- U + gain du*. 2: the robot realizes only part of each model step,
        # and at 1 falls from a reality gap took ~2x the trials to stop (sweep, 2026-09-30)
        p("ilc_gain", 2.0)
        p("ilc_gain_stage3", 0.0)                 # gain in Stage III; <= 0: ilc_gain
        p("ilc_forget", 1.0)                      # leak toward u_ref each step; 1: none
        p("ilc_tau_scale", 1.0)                   # QP's torque limit x this (weak motors)
        p("ilc_fmin", 5.0)                        # QP's least normal force per stance pair, N
        p("ilc_slack_weight", 1e2)                # QP's penalty on softened torque rows
        # QP's motor rows: go1 = the Go1 torque-speed envelope (the legacy A1 MDC constants
        # never bind, so the ILC asked speed-limited knees for force they could not make)
        p("ilc_mdc", "go1")
        p("ilc_mdc_speed_scale", 1.0)             # go1: no-load speeds x this
        p("ilc_lever_arms", "plan")               # SRB lever arms in contact: plan | measured
        p("stage3_theta_mode", "track")           # Stage III pitch rows: track | hold | free
        p("landing_pitch_target_deg", 0.0)        # Stage III target landing pitch (nose up +)
        p("fall_pitch_bias_deg", 0.0)             # a fall's flight tail seen this much more nose-down
        p("converge_any_stage", False)            # converged may be declared in Stages I-II too
        p("converged_requires_no_fall", False)    # a trial that fell never converges
        # Stage III safeguard variants (stage3_safeguard true only)
        p("stage3_backoff", "qu")                 # on rejection: qu (step weight x growth) | alpha (step x 0.5^n)
        p("safeguard_growth", 4.0)                # qu: step weight growth per rejection
        p("stage3_qu_scale_max", 0.0)             # qu: cap on that scale; <= 0: none
        p("stage3_accept_rows", "xzth")           # safeguard cost on landing x, z, pitch | x, z
        p("stage3_accept_tol", 0.0)               # best if cost <= best's x (1 + this)
        p("stage3_best_refresh", False)           # the best's cost: mean over re-flights of its forces
        p("stage3_fall_policy", "rollback")       # a Stage III fall: rollback to best | step from it
        p("n_stage1", 2)
        p("n_stage2", 3)
        p("max_trials", 25)
        p("pos_tol", 0.01)
        p("theta_tol_deg", 1.0)
        # files: the TO is loaded from reference_file if it exists, else solved and saved
        # there. log_dir/run_name/ gets the run (trial_log.TrialRecorder: meta.json,
        # reference.npz, one self-contained trial_NNN.npz per trial; run_name "" = a
        # timestamp). A trial file, or a run folder (its last trial), can seed a run:
        #   resume_from    continue learning the same task from that trial
        #   transfer_from  start this task from another task's learned trial, by
        #                  transfer_mode "retarget" (this task's own TO reference, plus the
        #                  correction learned on top of the other's) or "paper" (keep the
        #                  other task's reference and forces, aim at this goal, Stage III
        #                  only, one Stage III step before the first trial -- Sec. II-D3)
        p("reference_file", "")
        p("log_dir", "")
        p("run_name", "")
        p("resume_from", "")
        p("resume_file", "")                      # older name for resume_from
        p("transfer_from", "")
        p("transfer_mode", "retarget")
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
        # landing: balance = trunk PD (pitch, height, fore-aft) -> pair forces -> J^T, from
        # touchdown on; pd = hold the home pose with land_kp/kd alone (the old landing)
        p("landing_controller", "balance")
        p("bal_pitch_w", 35.0)                    # rad/s: pitch loop natural frequency
        p("bal_z_w", 8.0)                         # rad/s: CoM height loop
        p("bal_x_w", 2.0)                         # rad/s: re-centre the CoM over the feet
        p("bal_brake", 60.0)                      # 1/s: forward-velocity damping at landing
        p("bal_zeta", 1.0)                        # damping ratio of all three loops
        p("bal_fmin", 30.0)                       # N per pair: every foot keeps pushing down
        p("bal_hold_tau", 0.3)                    # s: trunk target moves to standing, then holds
        p("bal_hold_speed", 0.2)                  # m/s: below this forward speed the legs hold
        p("bal_crouch", 0.0)                      # m: the landing lets the trunk sink this far
        p("bal_fz_max", 3.5)                      # body weights: cap on the total landing push
        p("bal_w_lever", 0.3)                     # m: QP weighs a moment error like a force at this lever
        p("bal_w_z", 0.3)                         # QP weight of the vertical force (1: fore-aft)
        p("bal_fx_guard", True)                   # net fore-aft push only toward the CoM's place
        p("bal_z_guard", False)                   # no more than the weight while above height, rising
        p("bal_rise", 0.5)                        # s: then rises back to standing, once holding
        p("bal_kp", 40.0)                         # joint PD holding the feet on their anchors
        p("bal_kd", 3.0)
        p("bal_kd_absorb", 0.5)                   # joint damping until the legs hold: let them give
        p("bal_tau_frac", 0.85)                   # share of each motor's limit the landing QP may use
        # a pair whose feet have read unloaded for bal_lift_ticks ticks gets no force in the
        # landing QP: without it the QP books bal_fmin (and its friction) on lifted rear feet,
        # and the fore-aft guard is met on paper while the loaded front feet push forward
        p("bal_contact_aware", False)
        p("bal_lift_ticks", 5)
        # foot_contacts reading, above the foot's own reading early in flight, that ends the
        # flight: Go1's footForce is a raw sensor value with a per-foot offset of tens of
        # counts and drift, so each trial tares it while the feet are surely unloaded (the
        # first half of the flight) and asks for this much, or 5 sigma of that noise, more
        p("touchdown_force", 20.0)
        # last samples of flight: legs aim the feet where the plan has them relative to the
        # trunk's *planned* pitch, so a pitch error is taken up by the legs and all four feet
        # touch down together; 0: fly the planned joint angles to the end
        p("level_feet_samples", 10)
        # level-feet legs aim each foot's fore-aft offset at the plan's landing lever (sample
        # N) rather than its sample-by-sample path: an early touchdown still lands the front
        # feet ahead of the CoM, where the landing has a lever to brake the pitch over
        p("level_feet_hold_x", False)
        # milder: only the front feet, and only never behind their landing lever (the box
        # plans still swing them forward in the last samples; the 60 cm plans are there early
        # and are left alone -- holding both pairs made those fall)
        p("level_feet_front_lever", True)
        # flight: all thighs offset by -kp (pitch - planned) - kd (pitch rate - planned),
        # the legs as a reaction wheel holding the trunk on the plan's pitch; 0: off
        p("flight_att_kp", 0.0)                   # rad of thigh per rad of pitch error
        p("flight_att_kd", 0.0)                   # s
        p("flight_att_max", 0.35)                 # rad: largest thigh offset
        p("late_liftoff", False)                  # rear legs start their swing only once off the ground
        p("vel_filter_hz", 30.0)                  # low-pass on the differentiated trunk pose
        # mocap: capture -> receipt delay, s (Motive reports it). The ILC's trial log dates
        # each pose frame this much before it arrived; uncompensated, the jump's ~1.5 m/s
        # puts the measured landing 1.5 mm short per ms, and the ILC learns to overshoot
        p("pose_latency", 0.0)
        # ILC trial log: velocities by a line fit over +- this many s of pose frames (a
        # central difference of held/noisy mocap frames is noise; exact for constant
        # acceleration), and the flight's CoM by a ballistic fit
        p("log_vel_window", 0.01)
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
        jump = (float(get("jump_dx")), float(get("jump_dz")))
        learning = dict(
            pos_tol=float(get("pos_tol")), theta_tol=math.radians(float(get("theta_tol_deg"))),
            Qu_diag=float(get("qu")), Qe_diag=[float(v) for v in get("qe")],
            Qu_stage3=float(get("qu_stage3")) if float(get("qu_stage3")) > 0 else None,
            safeguard=bool(get("stage3_safeguard")), secant=bool(get("stage3_secant")),
            gain=float(get("ilc_gain")), gain_stage3=float(get("ilc_gain_stage3")),
            forget=float(get("ilc_forget")), tau_scale=float(get("ilc_tau_scale")),
            fmin=float(get("ilc_fmin")), slack_weight=float(get("ilc_slack_weight")),
            mdc=get("ilc_mdc"), mdc_speed_scale=float(get("ilc_mdc_speed_scale")),
            stage3_theta_mode=get("stage3_theta_mode"),
            landing_pitch_target=math.radians(float(get("landing_pitch_target_deg"))),
            fall_pitch_bias=math.radians(float(get("fall_pitch_bias_deg"))),
            converge_any_stage=bool(get("converge_any_stage")),
            converged_requires_no_fall=bool(get("converged_requires_no_fall")),
            stage3_backoff=get("stage3_backoff"), safeguard_growth=float(get("safeguard_growth")),
            stage3_qu_scale_max=(float(get("stage3_qu_scale_max"))
                                 if float(get("stage3_qu_scale_max")) > 0 else None),
            stage3_accept_rows=get("stage3_accept_rows"),
            stage3_accept_tol=float(get("stage3_accept_tol")),
            stage3_best_refresh=bool(get("stage3_best_refresh")),
            stage3_fall_policy=get("stage3_fall_policy"))
        self.lever_arms = get("ilc_lever_arms")
        if self.lever_arms not in ("plan", "measured"):
            raise ValueError(f"ilc_lever_arms must be plan or measured, got {self.lever_arms!r}")
        make_ilc = lambda reference: JumpILC(
            self.qm, jump=jump, box=box,
            phases=tuple(int(n) for n in get("phases")), dt=float(get("dt")),
            n_stage1=int(get("n_stage1")), n_stage2=int(get("n_stage2")),
            margin=float(get("margin")), swing_qd=float(get("swing_qd")),
            w_swing=float(get("w_swing")), box_clearance=float(get("box_clearance")),
            box_setback=float(get("box_setback")), to_steps=int(get("to_steps")),
            to_guess=get("to_guess"),
            landing_pitch_rate=(math.radians(float(get("landing_pitch_rate_deg")))
                                if float(get("landing_pitch_rate_deg")) > 0 else None),
            flight_pitch_rate=(math.radians(float(get("flight_pitch_rate_deg")))
                               if float(get("flight_pitch_rate_deg")) > 0 else None),
            flight_min_pitch=(math.radians(float(get("flight_min_pitch_deg")))
                              if float(get("flight_min_pitch_deg")) > -90 else None),
            flight_spin_change=(math.radians(float(get("flight_spin_change_deg")))
                                if float(get("flight_spin_change_deg")) > 0 else None),
            leg_clearance=(float(get("leg_clearance"))
                           if float(get("leg_clearance")) > 0 else None),
            landing_clearance=(float(get("landing_clearance"))
                               if float(get("landing_clearance")) > 0 else None),
            mu=float(get("mu")), reference=reference, **learning)

        parent = None
        transfer, mode = get("transfer_from"), get("transfer_mode")
        if mode not in ("retarget", "paper"):
            raise ValueError(f"transfer_mode must be retarget or paper, got {mode!r}")
        ref_file = get("reference_file")
        if transfer and mode == "paper":
            # nothing to plan: the source's reference is flown, aimed at this goal
            src = TrialRecorder.load(transfer)
            self.ilc = JumpILC.from_trial(self.qm, src, jump=jump, box=box, **learning)
            info = self.ilc.prime(src.log)
            parent = dict(transfer_from=src["path"], mode=mode, source_trial=src["trial"])
            self.get_logger().info(
                f"transfer (paper) from {src['path']}: its reference and forces, aimed at "
                f"{np.round(self.ilc.goal, 3)}, Stage III only; first-step QP "
                f"{'ok' if info['success'] else 'FAILED'}")
        else:
            load = ref_file if ref_file and os.path.exists(ref_file) else None
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
            if ref_file and load is None:
                self.ilc.save_reference(ref_file)
                self.get_logger().info(f"reference saved to {ref_file}")
            if transfer:
                src = TrialRecorder.load(transfer)
                self.ilc.transfer_from(src)
                parent = dict(transfer_from=src["path"], mode=mode, source_trial=src["trial"])
                self.get_logger().info(
                    f"transfer (retarget) from {src['path']}: this task's reference plus "
                    f"the correction learned there")
        self.get_logger().info(self.ilc.describe_reference())
        if not self.ilc.to_info["success"]:
            self.get_logger().warn("the TO did not converge; its last iterate is the reference")
        resume = get("resume_from") or get("resume_file")
        if resume:
            src = TrialRecorder.load(resume)
            self.ilc.restore(src)
            parent = dict(resume_from=src["path"], source_trial=src["trial"])
            self.get_logger().info(f"resumed from {src['path']}: next trial {self.ilc.trial + 1}")
        self.max_trials = int(get("max_trials"))
        self.recorder = None
        if get("log_dir"):
            params = {name: self.get_parameter(name).value for name in self._parameters
                      if name != "use_sim_time"}
            self.recorder = TrialRecorder(get("log_dir"), get("run_name"),
                                          meta=dict(params=params, config=self.ilc.config,
                                                    parent=parent))
            self.recorder.save_reference(self.ilc)
            self.get_logger().info(f"recording the run to {self.recorder.run_dir}")

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
        self.landing_controller = get("landing_controller")
        if self.landing_controller not in ("balance", "pd"):
            raise ValueError(f"landing_controller must be balance or pd, got {self.landing_controller!r}")
        zeta = float(get("bal_zeta"))
        self.bal_gains = {k: (float(get(f"bal_{k}_w")) ** 2, 2 * zeta * float(get(f"bal_{k}_w")))
                          for k in ("pitch", "z", "x")}
        self.bal_fmin = float(get("bal_fmin"))
        self.bal_brake = float(get("bal_brake"))
        self.bal_hold_tau = float(get("bal_hold_tau"))
        self.bal_hold_speed = float(get("bal_hold_speed"))
        self.bal_crouch = float(get("bal_crouch"))
        self.bal_fz_max = float(get("bal_fz_max")) * self.fb.total_mass * self.fb.g
        self.bal_w_lever, self.bal_w_z = float(get("bal_w_lever")), float(get("bal_w_z"))
        self.bal_fx_guard = bool(get("bal_fx_guard"))
        self.bal_z_guard = bool(get("bal_z_guard"))
        self.bal_rise = float(get("bal_rise"))
        # the landing force QP: 4 forces, friction cones |fx| <= mu fz as 4 rows
        mu = self.ilc.mdc_params["mu"]
        self.bal_cone = np.array([[1.0, -mu, 0.0, 0.0], [-1.0, -mu, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, -mu], [0.0, 0.0, -1.0, -mu]])
        self.bal_qp = ca.conic("bal", "qrqp", dict(h=ca.Sparsity.dense(4, 4),
                                                   a=ca.Sparsity.dense(10, 4)),
                               dict(print_iter=False, print_header=False, print_info=False,
                                    error_on_fail=False))
        self.bal_pd = (float(get("bal_kp")), float(get("bal_kd")))
        self.bal_kd_absorb = float(get("bal_kd_absorb"))
        self.bal_tau_frac = float(get("bal_tau_frac"))
        self.bal_contact_aware = bool(get("bal_contact_aware"))
        self.bal_lift_ticks = int(get("bal_lift_ticks"))
        self.level_feet_hold_x = bool(get("level_feet_hold_x"))
        self.level_feet_front_lever = bool(get("level_feet_front_lever"))
        self.touchdown_force = float(get("touchdown_force"))
        self.ff_tare, self.ff_base, self.ff_thresh = [], None, None
        self.vel_filter_hz = float(get("vel_filter_hz"))
        self.pose_latency = float(get("pose_latency"))
        self.log_vel_window = float(get("log_vel_window"))
        self.level_feet_samples = int(get("level_feet_samples"))
        self.flight_att = (float(get("flight_att_kp")), float(get("flight_att_kd")),
                           float(get("flight_att_max")))
        self.late_liftoff = bool(get("late_liftoff"))
        self.rear_off = False
        # the plan's feet relative to its base, world axes, for the level touchdown
        s_ref = self.ilc.to_info["s"]
        self.feet_off_ref = np.array([fb.feet(sk).full().ravel().reshape(2, 2) - sk[:2]
                                      for sk in s_ref])
        self.vel = None                       # filtered (time, pos, theta, [vx, vz, theta_d])
        self.touched_down = False
        self.anchor = None                    # landing: where each pair's foot is held
        self.pair_up = [0, 0]                 # ticks each pair's feet have read unloaded
        self.surface_z = None
        self.last_q = None
        self.bal_force = None
        self.t_touchdown = None
        self.measure_until = None
        # the standing CoM over its feet: where balance holds the trunk
        s_home = fb.standing_state()
        feet_home = fb.feet(s_home).full().ravel()
        self.com_over_feet = (fb.com(s_home).full().ravel()
                              - 0.5 * (feet_home[0:2] + feet_home[2:4]))
        self.base_over_feet = s_home[:2] - 0.5 * (feet_home[0:2] + feet_home[2:4])
        self.hold_pose = None                 # landing: trunk pose the legs hold [x, z, pitch]
        self.crouch = 0.0                     # landing: how far below standing height to aim
        self.holding = False

        # ---- I/O ----
        pose_topic = get("pose_topic") or ("base_odom" if self.is_sim else "")
        pose_type = get("pose_type") or "odometry"
        if not pose_topic:
            raise ValueError(f"backend {self.backend} needs pose_topic: the OptiTrack trunk pose")
        msg_type = {"odometry": Odometry, "pose": PoseStamped}[pose_type]
        self.pose = None                      # (receipt time, position (3,), quat (w,x,y,z))
        self.pose_seq = 0                     # pose frames received; a new one each frame
        self.pose_fresh = False               # this tick is the first to see the latest frame
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
        self.pose_seq += 1

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

        self._update_velocity(now)
        self.last_q = q
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

    def _update_velocity(self, now):
        """Trunk velocity [v_fwd, v_z, pitch rate] by differentiating the pose over each new
        frame, low-passed (first order, vel_filter_hz). Mocap frames come slower than the
        control ticks (OptiTrack: 120-360 Hz); differencing every tick would read a held
        frame as standing still and the next as a jump."""
        _, pos, quat = self.pose
        theta = planar_pitch(quat)
        self.pose_fresh = self.vel is None or self.pose_seq != self.vel[4]
        if not self.pose_fresh:
            return
        if self.vel is None or now <= self.vel[0]:
            self.vel = (now, pos.copy(), theta, np.zeros(3), self.pose_seq)
            return
        t0, p0, th0, v0, _ = self.vel
        h = now - t0
        yaw = self.frame["yaw"] if getattr(self, "frame", None) else yaw_of(quat)
        d = (pos - p0) / h
        raw = np.array([math.cos(yaw) * d[0] + math.sin(yaw) * d[1], d[2], (theta - th0) / h])
        a = 1.0 - math.exp(-2 * math.pi * self.vel_filter_hz * h)
        self.vel = (now, pos.copy(), theta, v0 + a * (raw - v0), self.pose_seq)

    def _balance(self, q, dq):
        """
        Landing balance. The jump lands moving forward at ~1.5 m/s; stopped dead at the
        feet, that momentum pivots the trunk over the front feet. Braking at the feet
        pitches it nose-down too (the backward push acts below the CoM), so how hard the
        robot may brake is set by how much front-rear load shift the pitch can take.

        Pair forces f = [fFx, fFz, fRx, fRz] (world, ground on robot), applied through
        J^T, from a small QP each tick: every pair pushes down at least bal_fmin (the rear
        feet stay on the ground) inside its friction cone, and f matches, in this order of
        weight,
          - the pitch moment of a PD levelling the trunk (bal_pitch_w),
          - braking: forward velocity damped at bal_brake (1/s), plus a weak spring
            (bal_x_w) re-centring the CoM over the feet once it has stopped,
          - the vertical force of a PD on the CoM height over the feet (bal_z_w),
        with every motor's torque inside its limit.
        The moment row includes the braking forces' own lever (they act below the CoM), so
        the QP loads the front legs to cancel the nose-down moment braking makes. It has
        to brake early: the landing's ~19 N s of forward momentum, stopped at the feet,
        is ~5 N s of nose-down angular impulse, and the front legs' lever to cancel it
        shrinks as the CoM travels forward -- braked gently, the CoM reaches the front
        feet first and the trunk pivots over them. The vertical force gives way first:
        the front legs push harder than the height loop asks.

        The joint PD targets keep the feet on the landing surface (the standing height
        at the jump's start plus the box): each pair's foot is anchored where it lands, a
        pair still in the air is aimed straight down onto the surface, by IK with the
        trunk at a held pose: the measured one while the trunk still moves faster than
        bal_hold_speed (the legs give with the impact), then moving over bal_hold_tau to
        level standing over the anchored feet, where the legs hold the robot. Both the
        height loop and that pose aim bal_crouch below standing height at touchdown, so
        the legs compress to absorb the landing, and rise back over bal_rise once holding.
        """
        fb, ilc = self.fb, self.ilc
        vx, vz, theta_d = self.vel[3]
        s = self._planar_state(q)
        theta, qp = s[2], s[3:]
        com = fb.com(s).full().ravel()
        feet = fb.feet(s).full().ravel().reshape(2, 2)
        q_des = expand(self._anchored_joints(s, feet))
        err = com - 0.5 * (feet[0] + feet[1]) - self.com_over_feet
        err[1] += self.crouch                                   # absorb: aim lower at first
        m, inertia, g = fb.total_mass, ilc.nominal.I, fb.g
        (kth, dth), (kz, dz), (kx, _) = (self.bal_gains[k] for k in ("pitch", "z", "x"))
        mu, fmin, fmax = ilc.mdc_params["mu"], self.bal_fmin, ilc.mdc_params["fmax"]
        Fz = float(np.clip(m * (g - kz * err[1] - dz * vz), 2 * fmin, 2 * fmax))
        Fx_des = m * (-self.bal_brake * vx - kx * err[0])
        M_des = inertia * (-kth * theta - dth * theta_d)       # nose-up moment
        r = feet - com                                          # levers, front and rear
        rows = np.array([[-r[0, 1], r[0, 0], -r[1, 1], r[1, 0]],   # moment r_x f_z - r_z f_x
                         [1.0, 0.0, 1.0, 0.0],                     # net fore-aft force
                         [0.0, 1.0, 0.0, 1.0]])                    # net vertical force
        w = np.array([1.0 / self.bal_w_lever ** 2, 1.0, self.bal_w_z])  # moment in N over a lever
        target = np.array([M_des, Fx_des, Fz])
        H = 2.0 * (rows.T * w) @ rows + 1e-6 * np.eye(4)
        gvec = -2.0 * (rows.T * w) @ target
        lbx = [-mu * fmax, fmin, -mu * fmax, fmin]
        ubx = [mu * fmax, fmax, mu * fmax, fmax]
        if self.bal_contact_aware:
            fd = self._feet_down()
            for i in range(2):
                self.pair_up[i] = 0 if (fd[2 * i] or fd[2 * i + 1]) else self.pair_up[i] + 1
                if self.pair_up[i] >= self.bal_lift_ticks:      # lifted: it pushes on nothing
                    lbx[2 * i:2 * i + 2] = [0.0, 0.0]
                    ubx[2 * i:2 * i + 2] = [0.0, 0.0]
        # rows: friction cones (<= 0), then per-motor torque T f within 85% of the limit
        # (the rest for the joint PD) -- demanded past it, the front knees fold
        # and the total push under bal_fz_max: the legs give instead of meeting the impact
        # rigidly (braking hard asks for all the friction, i.e. all the vertical force, the
        # legs can make)
        T = fb.torque_map(qp, theta)
        tmax = self.bal_tau_frac * collapse(self.tau_limit)
        # and the net fore-aft push never drives the CoM further off: forward only while
        # it is behind its place over the feet and not moving away, backward likewise.
        # Near the front feet, vertical forces have no lever left to raise the nose, and
        # the cheapest nose-up moment is a forward push at the front feet -- which carries
        # the CoM past them
        away = err[0] > 0 or vx > 0
        fx_lo = 0.0 if (self.bal_fx_guard and not away) else -np.inf
        fx_hi = 0.0 if (self.bal_fx_guard and away) else np.inf
        # likewise vertically: above its height and still rising, the trunk gets no more than
        # its weight -- the pitch loop, pushing the nose up with the front legs, otherwise
        # pops the robot up onto its front feet with the rear ones hanging
        fz_hi = self.bal_fz_max
        if self.bal_z_guard and err[1] > 0 and vz > 0:
            fz_hi = min(fz_hi, m * g)
        A = np.vstack([self.bal_cone, T, [[0.0, 1.0, 0.0, 1.0]], [[1.0, 0.0, 1.0, 0.0]]])
        lba = np.concatenate([np.full(4, -np.inf), -tmax, [-np.inf], [fx_lo]])
        uba = np.concatenate([np.zeros(4), tmax, [fz_hi], [fx_hi]])
        sol = self.bal_qp(h=H, g=gvec, a=A, lba=lba, uba=uba, lbx=lbx, ubx=ubx)
        f = np.array(sol["x"]).ravel()
        if not self._forces_ok(f, lbx, ubx, mu):
            # torque limits, the push cap or the fore-aft guard can be jointly infeasible
            # at an awkward touchdown pose -- and an infeasible QP returns garbage (a rear
            # pair pulling on the ground at -240 N, the front past fmax). Drop those rows:
            # bounds and friction cones alone are always feasible
            free = np.arange(len(lba)) >= 4
            lba2, uba2 = lba.copy(), uba.copy()
            lba2[free], uba2[free] = -np.inf, np.inf
            f = np.array(self.bal_qp(h=H, g=gvec, a=A, lba=lba2, uba=uba2,
                                     lbx=lbx, ubx=ubx)["x"]).ravel()
        if not self._forces_ok(f, lbx, ubx, mu):                # never expected: project
            f = np.where(np.isfinite(f), f, 0.0)
            f[1::2] = np.clip(f[1::2], fmin, fmax)
            f[0::2] = np.clip(f[0::2], -mu * f[1::2], mu * f[1::2])
        self.bal_force = f
        tau_ff = np.clip(expand(fb.torque_map(qp, theta) @ f), -self.tau_limit, self.tau_limit)
        # while braking, the joint targets follow the trunk, and damping them toward zero
        # speed would stiffen the legs against the impact (kd 3 at ~10 rad/s of knee
        # compression is past the motors' limit): light damping until the legs hold
        kd_leg = self.bal_pd[1] if self.holding else self.bal_kd_absorb
        kp, kd = np.full(NU, self.bal_pd[0]), np.full(NU, kd_leg)
        kp[HIP_IDX], kd[HIP_IDX] = self.gains["stand"]
        return q_des, np.zeros(NU), kp, kd, tau_ff

    @staticmethod
    def _forces_ok(f, lbx, ubx, mu, tol=1e-3):
        """Whether pair forces are finite, inside their bounds and friction cones."""
        return (bool(np.all(np.isfinite(f)))
                and bool(np.all(f >= np.asarray(lbx) - tol) and np.all(f <= np.asarray(ubx) + tol))
                and bool(np.all(np.abs(f[0::2]) <= mu * f[1::2] + tol)))

    def _feet_down(self):
        """Per foot (FL FR RL RR): loaded, by the reading above its in-flight tare when
        this trial has one (see touchdown_force), else by the raw reading."""
        c = self.contacts
        if not np.isfinite(c).all():
            return np.zeros(4, bool)
        if self.ff_base is None:
            return c > self.touchdown_force
        return c - self.ff_base > self.ff_thresh

    def _anchored_joints(self, s, feet):
        """Planar joint targets that put each pair's foot on its landing anchor (see
        _balance), by damped least-squares IK with the trunk where it is measured."""
        known = np.isfinite(self.contacts).all()
        fd = self._feet_down()
        down = [bool(fd[2 * i] or fd[2 * i + 1]) for i in range(2)]
        if self.anchor is None:
            self.anchor = feet.copy()
            self.anchor[:, 1] = self.surface_z
            self.anchored = [bool(down[i]) or not known for i in range(2)]
            self.q_anchor = s[3:].copy()
            self.hold_pose = s[:3].copy()
            self.holding = False
            self.crouch = self.bal_crouch
        for i in range(2):
            if not self.anchored[i]:
                self.anchor[i, 0] = feet[i, 0]            # descending: straight down
                self.anchored[i] = bool(down[i])
        # the trunk pose the legs hold. While the landing's momentum is still being braked
        # (forward speed above bal_hold_speed) it is the measured pose: the legs give with
        # the trunk, and the braking stays with the force QP -- held back by the legs
        # instead, the trunk pitches over the front feet. Below that speed it moves
        # (bal_hold_tau) to level standing over the anchored feet and stays: IK at the
        # measured pose throughout would move the targets with the trunk, and nothing
        # would stop it creeping forward over the feet
        if not self.holding and abs(self.vel[3][0]) > self.bal_hold_speed:
            self.hold_pose = s[:3].copy()
        else:
            self.holding = True
            if self.bal_rise > 0:                       # stand back up
                self.crouch *= math.exp(-self.tick / self.bal_rise)
            goal = np.concatenate([0.5 * (self.anchor[0] + self.anchor[1]) + self.base_over_feet,
                                   [0.0]])
            goal[1] -= self.crouch
            a = 1.0 - math.exp(-self.tick / self.bal_hold_tau) if self.bal_hold_tau > 0 else 1.0
            self.hold_pose = self.hold_pose + a * (goal - self.hold_pose)
        s_hold = np.concatenate([self.hold_pose, s[3:]])
        self.q_anchor = self._ik(s_hold, self.anchor.ravel(), self.q_anchor)
        return self.q_anchor

    def _planar_state(self, q):
        """Measured planar state [fwd, z, pitch, q] (fwd along the jump's heading)."""
        _, pos, quat = self.pose
        yaw = self.frame["yaw"] if getattr(self, "frame", None) else yaw_of(quat)
        fwd = math.cos(yaw) * pos[0] + math.sin(yaw) * pos[1]
        return np.concatenate([[fwd, pos[2], planar_pitch(quat)], collapse(q)])

    def _ik(self, s, target, q0, iters=6):
        """Planar joints putting the feet at target [xF, zF, xR, zR] with the trunk at s,
        damped least squares from q0, inside the joint range."""
        fb = self.fb
        lo, hi = fb.joint_range[:, 0], fb.joint_range[:, 1]
        q, s_ik = np.array(q0, float), np.array(s, float)
        for _ in range(iters):
            s_ik[3:] = q
            e = target - fb.feet(s_ik).full().ravel()
            if np.abs(e).max() < 1e-4:
                break
            J = fb.Jc(s_ik).full()[:, 3:]
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(4), e), lo, hi)
        return q

    def _land_cmd(self, q, dq):
        if self.landing_controller == "balance":
            return self._balance(q, dq)
        return self._hold("land")

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
        if self.exit_when_done and self.ilc.trial >= self.max_trials:
            # e.g. resumed past max_trials, which counts from trial 1
            self.get_logger().info(f"max_trials ({self.max_trials}) reached")
            raise SystemExit(0)
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
                        dq_des=[], tau_total=[], effort=[], contacts=[], bal_f=[],
                        pose_new=[])
        self.U_flown = self.ilc.U.copy()
        self.fell = False
        self.touched_down = False
        self.t_touchdown = None
        self.anchor = None
        self.pair_up = [0, 0]
        self.rear_off = False
        self.ff_tare, self.ff_base, self.ff_thresh = [], None, None
        # the landing surface for the balance controller's foot anchors: the feet's
        # height standing now, raised by the box the plan lands on
        feet = self.fb.feet(self._planar_state(self.last_q)).full().ravel()
        box = self.ilc.box
        self.surface_z = 0.5 * (feet[1] + feet[3]) + (box["height"] if box is not None else 0.0)
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
        # touchdown before the plan's last sample: the landing controller takes over there
        # and then (late in flight only, so push-off contact never trips it). The foot
        # sensors are tared over the first half of the flight
        k_detect = ilc.Nc + (ilc.N - ilc.Nc) // 2
        if ilc.Nc + 2 <= k < k_detect and np.isfinite(self.contacts).all():
            self.ff_tare.append(self.contacts.copy())
        if k >= k_detect and self.ff_base is None and len(self.ff_tare) >= 5:
            tare = np.array(self.ff_tare)
            self.ff_base = np.median(tare, axis=0)
            self.ff_thresh = np.maximum(self.touchdown_force, 5.0 * tare.std(axis=0))
        if self.landing_controller == "balance" and k >= k_detect:
            if self.touched_down or self._feet_down().any():
                if not self.touched_down:
                    self.t_touchdown = t
                self.touched_down = True
                q_des, dq_des, kp, kd, tau_ff = self._balance(q, dq)
                self._record(t, q, dq, q_des, dq_des, kp, kd, tau_ff,
                             kp * (q_des - q) + kd * (dq_des - dq) + tau_ff)
                return q_des, dq_des, kp, kd, tau_ff
        q_des = expand((1 - a) * ilc.to_info["q_ref"][k] + a * ilc.to_info["q_ref"][k + 1])
        dq_des = expand((1 - a) * ilc.to_info["qd_ref"][k] + a * ilc.to_info["qd_ref"][k + 1])
        if self.level_feet_samples > 0 and k >= ilc.N - self.level_feet_samples:
            # feet where the plan puts them off the base (world axes), whatever the trunk's
            # pitch: the legs take up the pitch error and all feet meet the surface together
            s_meas = self._planar_state(q)
            target = s_meas[:2] + ((1 - a) * self.feet_off_ref[k] + a * self.feet_off_ref[k + 1])
            if self.level_feet_hold_x:
                target[:, 0] = s_meas[0] + self.feet_off_ref[ilc.N][:, 0]
            elif self.level_feet_front_lever:
                target[0, 0] = max(target[0, 0], s_meas[0] + self.feet_off_ref[ilc.N][0, 0])
            q_level = self._ik(s_meas, target.ravel(), collapse(q_des))
            q_des = expand(q_level)
        if k >= ilc.Nc and self.flight_att[0] + self.flight_att[1] > 0:
            # flight attitude: the feet carry no force, so nothing the ILC learns can steer
            # the trunk here -- yet the box plans' quick leg tuck after takeoff, flown with
            # lag, spins the trunk ~40 deg/s further nose-down than planned. Swing all thighs
            # against the pitch error: with the legs' angular momentum traded against the
            # trunk's, raising q_thigh raises the nose
            kp_a, kd_a, max_a = self.flight_att
            th_ref = (1 - a) * ilc.x_ref[k, 2] + a * ilc.x_ref[k + 1, 2]
            w_ref = (1 - a) * ilc.x_ref[k, 5] + a * ilc.x_ref[k + 1, 5]
            e_th, e_w = planar_pitch(self.pose[2]) - th_ref, self.vel[3][2] - w_ref
            delta = float(np.clip(-kp_a * e_th - kd_a * e_w, -max_a, max_a))
            q_des = q_des + expand(np.array([delta, 0.0, delta, 0.0]))

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
        if self.late_liftoff and ilc.Nc <= k < ilc.Nc + 5 and not self.rear_off:
            # rear feet still on the ground past the planned takeoff: keep the rear legs
            # force-controlled (no swing PD or swing torque) until they come off, rather than
            # tucking them against the ground
            c = self.contacts
            if np.isfinite(c).all() and max(c[2], c[3]) > self.touchdown_force:
                for idx in PLANAR_TO_CANONICAL[2] + PLANAR_TO_CANONICAL[3]:
                    kp[idx], kd[idx] = self.gains["contact"]
                    tau_ff[idx] = 0.0
            else:
                self.rear_off = True
        # everything the motor carries besides J^T f of the ILC's forces: the ILC's torque
        # limits apply to  T u + this  (QuadILCStageSolver.solve's tau_pd_k)
        tau_pd = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff - tau_ilc

        self._record(t, q, dq, q_des, dq_des, kp, kd, tau_ff, tau_pd)
        return q_des, dq_des, kp, kd, tau_ff

    def _record(self, t, q, dq, q_des, dq_des, kp, kd, tau_ff, tau_pd):
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
        self.rec["pose_new"].append(self.pose_fresh)          # first tick of a mocap frame
        # the landing controller's pair forces this tick (NaN before it takes over)
        self.rec["bal_f"].append(self.bal_force.copy() if self.bal_force is not None
                                 else np.full(4, np.nan))
        self.bal_force = None

    def _tick_land(self, now, q, dq):
        if self.fell or now - self.phase_t0 >= self.land_time:
            self._set_phase("update", now)
            # the recording is final from here: the update thread reads it
            self.worker = threading.Thread(target=self._update_worker, daemon=True)
            self.worker.start()
            return self._tick_damp(now, q, dq) if self.fell else self._land_cmd(q, dq)
        if self.fell:
            return self._tick_damp(now, q, dq)
        cmd = self._land_cmd(q, dq)
        # the landing is recorded too (t runs on past the jump); the ILC's log only
        # resamples the jump itself, samples 0..N
        q_des, dq_des, kp, kd, tau_ff = cmd
        kp, kd = np.full(NU, kp), np.full(NU, kd)
        self._record(self.T_jump + now - self.phase_t0, q, dq, q_des, dq_des, kp, kd, tau_ff,
                     kp * (q_des - q) + kd * (dq_des - dq) + tau_ff)
        return cmd

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
        return self._tick_damp(now, q, dq) if self.fell else self._land_cmd(q, dq)

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
        # the trunk pose by mocap frame: each dated where it was first seen, less the mocap
        # latency, with the joints at that time (they are read every tick)
        new = rec["pose_new"].astype(bool) if len(rec["pose_new"]) == len(t) \
            else np.ones(len(t), bool)
        new[0] = True
        tf = t[new] - self.pose_latency
        qf = np.column_stack([np.interp(tf, t, q_planar[:, i]) for i in range(4)])
        com = np.array([fb.com(np.concatenate([[x, z, th], qp])).full().ravel()
                        for x, z, th, qp in zip(fwd[new], rel[new, 2], theta[new], qf)])
        shift = ilc.x_ref[0, :2] - com[0]                   # start where the reference starts
        com = com + shift
        theta_f = theta[new]
        feet = np.array([fb.feet(st).full().reshape(2, 2) for st in states]) + shift

        grid = np.arange(ilc.N + 1) * ilc.dt
        grid[-1] = min(grid[-1], tf[-1])                    # last frame may fall just short
        at = lambda y: np.interp(grid, tf, y)
        slope = lambda y: self._local_slope(tf, y, grid, self.log_vel_window)
        X = np.column_stack([at(com[:, 0]), at(com[:, 1]), at(theta_f),
                             slope(com[:, 0]), slope(com[:, 1]), slope(theta_f)])
        X = self._free_flight_tail(X, (tf, com, theta_f))
        at_tick = lambda y: np.interp(grid, t, y)          # joint series: every tick
        Nc = ilc.Nc
        per_motor_pd = np.array([collapse(v) for v in rec["tau_pd"]])
        log = dict(
            X=X,
            q=np.column_stack([at_tick(q_planar[:, i]) for i in range(4)])[:Nc],
            theta=X[:Nc, 2],
            qdot=np.column_stack([at_tick(dq_planar[:, i]) for i in range(4)])[:Nc],
            tau_pd=np.column_stack([at_tick(per_motor_pd[:, i]) for i in range(4)])[:Nc],
            fell=bool(self.fell),             # a trial that fell is never the ILC's best
        )
        if self.measure_until is not None:    # free-flight tail from this sample on
            log["measured_until"] = int(self.measure_until)
        if self.lever_arms == "measured":
            # lever arms (contact point - whole-body CoM) in contact, each from one measured
            # state, so the trunk position (and its mocap noise) cancels. The front pair
            # keeps the plan's once it swings (rear-leg contact): it carries no force there
            tc = t <= (Nc + 1) * ilc.dt + 1e-9
            lever = np.array([fb.feet(st).full().reshape(2, 2) - fb.com(st).full().ravel()
                              for st in np.asarray(states)[tc]]) - np.array([0.0, fb.foot_radius])
            arm = lambda i: np.column_stack([np.interp(grid[:Nc], t[tc], lever[:, i, j])
                                             for j in range(2)])
            R1, R2 = np.array(ilc.R1, float), np.array(ilc.R2, float)
            R1[:ilc.Ndc] = arm(0)[:ilc.Ndc]
            R2[:Nc] = arm(1)
            log.update(R1=R1, R2=R2)
        return log, rec, self._measured_clearance(t, feet)

    @staticmethod
    def _local_slope(ts, y, grid, w):
        """dy/dt at each grid time: a line through the samples within +-w of it (at least
        the three nearest)."""
        out = np.empty(len(grid))
        for i, g in enumerate(grid):
            m = np.flatnonzero(np.abs(ts - g) <= w + 1e-9)
            if m.size < 3:
                m = np.argsort(np.abs(ts - g))[:3]
            out[i] = np.polyfit(ts[m] - g, y[m], 1)[0]
        return out

    def _free_flight_tail(self, X, frames=None):
        """
        Keep the touchdown controllers out of what the ILC learns from. From the sample
        where the first of them acts -- the level-feet legs (level_feet_samples before
        the end), or touchdown, whichever is earlier -- the measured state is replaced by
        the reference plus the error at that sample carried on as in free flight: position
        and pitch errors grow by the velocity and pitch-rate errors, which stay constant.
        That is how the ILC's SRB model propagates an error once the feet are off the
        ground, and exact for the CoM, which is ballistic whatever the legs do; what the
        landing controllers do after it (legs reaching for the surface, the balance
        forces at contact) is not something the learned forces act through.

        The error there comes from fits over the flight up to that sample (`frames`: pose
        frame times, CoM, pitch): the CoM ballistic (a line in x, a parabola of known g in
        z), the pitch a parabola. The tail multiplies the velocity error by up to 0.1 s,
        and mocap noise differentiated frame to frame would dominate it; the fits average
        ~20 frames, and match the nominal sim's measured CoM to 0.1 mm.
        """
        ilc = self.ilc
        starts = []
        if self.flight_att[0] + self.flight_att[1] > 0:
            starts.append(ilc.Nc)                 # flight attitude feedback from takeoff
        if self.level_feet_samples > 0:
            starts.append(ilc.N - self.level_feet_samples)
        if self.t_touchdown is not None:
            starts.append(int(self.t_touchdown / ilc.dt + 1e-6))
        if not starts:
            return X
        k0 = max(min(starts), ilc.Nc + 1)
        if k0 >= ilc.N:
            return X
        X = X.copy()
        if frames is not None:
            tf, com, theta = frames
            tb = k0 * ilc.dt
            m = (tf >= (ilc.Nc + 2) * ilc.dt - 1e-9) & (tf <= tb + 1e-9)
            if m.sum() >= 5:
                g, h_f = self.fb.g, tf[m] - tb
                bx = np.polyfit(h_f, com[m, 0], 1)
                bz = np.polyfit(h_f, com[m, 1] + 0.5 * g * h_f ** 2, 1)
                bth = np.polyfit(h_f, theta[m], 2)
                ks = np.arange(ilc.Nc + 2, k0 + 1)
                h = ks * ilc.dt - tb
                X[ks, 0], X[ks, 3] = np.polyval(bx, h), bx[0]
                X[ks, 1], X[ks, 4] = np.polyval(bz, h) - 0.5 * g * h ** 2, bz[0] - g * h
                X[ks, 2], X[ks, 5] = np.polyval(bth, h), np.polyval(np.polyder(bth), h)
        e = X[k0] - ilc.x_ref[k0]
        for k in range(k0 + 1, ilc.N + 1):
            h = (k - k0) * ilc.dt
            X[k, 0:3] = ilc.x_ref[k, 0:3] + e[0:3] + h * e[3:6]
            X[k, 3:6] = ilc.x_ref[k, 3:6] + e[3:6]
        self.measure_until = k0
        return X

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
            result = self.ilc.update(log)
            result["clearance"] = clearance           # measured, not the TO's legs
            result["fell"] = bool(self.fell)          # landing failed: tilted past max_tilt
            if self.measure_until is not None:        # free-flight tail from this sample on
                result["measured_until"] = int(self.measure_until)
            self.last_result = result
            self.get_logger().info(describe_result(result))
            if self.recorder is not None:
                self.recorder.record(self.ilc, self.U_flown, log, rec)
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
