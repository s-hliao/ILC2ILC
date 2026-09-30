#!/usr/bin/env python3
"""ILC jumping against MuJoCo in one process, in lockstep: for tests and weight sweeps.

    ilc_jump_lockstep.py --jump 0.6 0.0 --ground 2e3 5e2 --log-dir /ilc_ws/log/soft
    ilc_jump_lockstep.py --jump 0.5 0.2 --box 0.25 0.2 --qu 1e-5 --qe 1 3 3 .01 .01 .01
    ilc_jump_lockstep.py --jump 0.6 0.0 --payload 2.0 --viewer --realtime
    ilc_jump_lockstep.py --robot go1 --jump 0.6 0.0 --act-delay 0.004 --pose-rate 240 \
        --pose-delay 0.008 --pose-noise 0.001 0.005 --mass-scale 1.1 --motor-curve

It runs the very same controller as the launch file -- `IlcJumpNode` from
ilc_jump_sim.py, its phases, gains, feedforward and ILC update -- but feeds it
MuJoCo state by calling its callbacks directly instead of over topics, and steps
the physics only once it has answered. Every tick is therefore exactly what
sim_node + ROS give when nothing is late (one control period of delay, the PD
re-evaluated every physics step), with no dependence on machine load. Trials
run as fast as the CPU allows, so several runs can go in parallel.

The ground stiffness and payload go to the simulated robot only; the controller
builds its own nominal QuadModel and never learns of them (as in the paper,
Sec. III-A). So does the rest of the reality gap (`Reality`: mass, friction,
actuator delay and torque-speed limits, mocap latency/rate/noise, encoder noise,
raw foot sensors); each trial then also records the true trunk pose and joints
(rec_true_*), since what the controller measured is no longer the truth. `--summary` writes the per-trial results as JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import mujoco
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ilc_jump_sim import IlcJumpNode  # noqa: E402

from ilc_quad.sim_quad_model import (  # noqa: E402
    CANONICAL_JOINT_NAMES,
    QuadModel,
    default_menagerie_root,
    pd_torque,
    unpack_joint_cmd,
)


def ros_value(v) -> str:
    """A parameter override that ROS parses back as the declared type."""
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(ros_value(x) for x in v) + "]"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return np.format_float_positional(v, trim="0")   # '0.00003', not '3e-05'
    return str(v)


class _CommandSink:
    """Stands in for the node's joint_cmd publisher: keeps the latest command."""

    def __init__(self):
        self.cmd = np.zeros((5, 12))

    def publish(self, msg):
        self.cmd = unpack_joint_cmd(msg.data)


def _stamp(msg, t):
    msg.header.stamp.sec = int(t)
    msg.header.stamp.nanosec = min(int(round((t - int(t)) * 1e9)), 999_999_999)


# Go1 joint output speeds at which the motors run out of voltage (Unitree's spec sheet:
# hip/thigh 30.1 rad/s, knee 20.06 rad/s), canonical order
GO1_NO_LOAD_SPEED = np.tile([30.1, 30.1, 20.06], 4)


def _small_rotation(rng, sigma):
    """(w, x, y, z) of a random rotation, each axis ~ N(0, sigma) rad."""
    r = rng.normal(0.0, sigma, 3)
    a = float(np.linalg.norm(r))
    if a < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([[math.cos(a / 2)], math.sin(a / 2) * r / a])


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


class Reality:
    """
    What separates the simulated robot from the controller's model of it, as on the real
    Go1 -- all of it hidden from the controller:

      model      mass_scale (every robot body's mass and inertia), com_offset (trunk CoM
                 shifted, m, base frame x/z), friction (ground mu), joint_friction
                 (Coulomb, N m, added), payload/ground (QuadModel's own options)
      actuators  act_delay (s from the controller's command to the motors: SDK UDP and
                 the motor board), motor_scale (torque limits), motor_curve (the Go1
                 torque-speed envelope: full torque up to motor_knee x the no-load speed,
                 then down linearly to none at the no-load speed x motor_speed_scale,
                 while motoring; a sagging battery lowers the latter)
      sensing    pose (mocap): pose_rate Hz frames, sample-and-hold, pose_delay s
                 latency, pose_noise [m, rad] per frame; joints: joint_noise [rad, rad/s]
                 per tick; foot_sensor [bias, gain spread, noise]: Go1's raw footForce
                 reads bias_i + gain_i F + noise, bias_i ~ U(0, bias), gain_i ~ U(1-g, 1+g)
    """

    def __init__(self, args, sim, period):
        self.sim, self.period = sim, period
        self.rng = np.random.default_rng(args.seed)
        m = sim.model
        robot = [b for b in range(m.nbody) if m.body_rootid[b] == m.body_rootid[sim.base_body_id]
                 and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) != "payload"]
        if args.mass_scale != 1.0:
            m.body_mass[robot] *= args.mass_scale
            m.body_inertia[robot] *= args.mass_scale
        if any(args.com_offset):
            m.body_ipos[sim.base_body_id, [0, 2]] += args.com_offset
        if args.friction > 0:
            m.geom_friction[:, 0] = args.friction
        if args.joint_friction > 0:
            m.dof_frictionloss[sim.qvel_adr] += args.joint_friction
        mujoco.mj_setConst(m, sim.data)
        sim.torque_limit = sim.torque_limit * args.motor_scale
        self.motor_curve = args.motor_curve
        self.w_max = GO1_NO_LOAD_SPEED * args.motor_speed_scale
        self.w_knee = args.motor_knee * self.w_max
        self.act_delay = args.act_delay
        self.pose_rate, self.pose_delay = args.pose_rate, args.pose_delay
        self.pose_noise = args.pose_noise
        self.joint_noise = args.joint_noise
        bias, spread, self.ff_noise = args.foot_sensor
        self.ff_bias = self.rng.uniform(0.0, bias, 4) if bias > 0 else np.zeros(4)
        self.ff_gain = self.rng.uniform(1 - spread, 1 + spread, 4) if spread > 0 else np.ones(4)
        self.reset()

    def reset(self):
        self.cmds = []                            # (time it reaches the motors, cmd)
        self.poses = []                           # true (t, pos, quat), for the mocap's delay
        self.frame = None                         # (frame index, pos, quat) held by the mocap

    # -- actuators --
    def command(self, t, cmd):
        self.cmds.append((t + self.act_delay - 1e-9, cmd.copy()))

    def motor_torque(self, t):
        """The command the motors hold at time t, through the PD and the motor envelope."""
        while len(self.cmds) > 1 and self.cmds[1][0] <= t:
            self.cmds.pop(0)
        cmd = self.cmds[0][1] if self.cmds and self.cmds[0][0] <= t else np.zeros((5, 12))
        q, dq = self.sim.joint_positions(), self.sim.joint_velocities()
        tau = pd_torque(cmd, q, dq)
        lim = self.sim.torque_limit.copy()
        if self.motor_curve:
            drop = np.clip((self.w_max - np.abs(dq)) / (self.w_max - self.w_knee), 0.0, 1.0)
            motoring = tau * dq > 0
            lim = np.where(motoring, lim * drop, lim)
        return np.clip(tau, -lim, lim)

    # -- sensing --
    def pose(self, t):
        """The mocap's latest frame at time t: (pos, quat, whether it is new this tick)."""
        pos, quat = self.sim.base_position(), self.sim.base_quat()
        if self.pose_rate <= 0 and self.pose_delay <= 0 and not any(self.pose_noise):
            return pos, quat, True
        self.poses.append((t, pos, quat))
        seen = t - self.pose_delay
        if self.pose_rate > 0:
            seen = math.floor(seen * self.pose_rate + 1e-9) / self.pose_rate
        while len(self.poses) > 1 and self.poses[1][0] <= seen + 1e-9:
            self.poses.pop(0)
        idx = round(seen * (self.pose_rate or 1e6))
        new = self.frame is None or self.frame[0] != idx
        if new:
            _, p, q = self.poses[0]
            p = p + self.rng.normal(0.0, self.pose_noise[0], 3)
            q = _quat_mul(q, _small_rotation(self.rng, self.pose_noise[1]))
            self.frame = (idx, p, q / np.linalg.norm(q))
        return self.frame[1].copy(), self.frame[2].copy(), new

    def joints(self):
        q, dq = self.sim.joint_positions(), self.sim.joint_velocities()
        sq, sdq = self.joint_noise
        return q + self.rng.normal(0.0, sq, 12), dq + self.rng.normal(0.0, sdq, 12)

    def foot_forces(self):
        f = self.sim.foot_normal_forces()
        return self.ff_bias + self.ff_gain * f + self.rng.normal(0.0, self.ff_noise, 4) \
            if self.ff_noise > 0 or self.ff_bias.any() or (self.ff_gain != 1).any() else f

    def describe(self, args):
        return dict(mass_scale=args.mass_scale, com_offset=list(args.com_offset),
                    friction=args.friction, joint_friction=args.joint_friction,
                    act_delay=args.act_delay, motor_scale=args.motor_scale,
                    motor_curve=args.motor_curve, motor_knee=args.motor_knee,
                    motor_speed_scale=args.motor_speed_scale, pose_rate=args.pose_rate,
                    pose_delay=args.pose_delay, pose_noise=list(args.pose_noise),
                    joint_noise=list(args.joint_noise), foot_sensor=list(args.foot_sensor),
                    foot_bias=self.ff_bias.tolist(), foot_gain=self.ff_gain.tolist(),
                    seed=args.seed)


def run(args) -> list[dict]:
    box = dict(x_front=args.box[0], height=args.box[1]) if args.box else None
    ground = dict(kp=args.ground[0], kd=args.ground[1]) if args.ground else None
    payload = dict(mass=args.payload) if args.payload else None
    root = args.menagerie_root or default_menagerie_root()
    sim = QuadModel(args.robot, root, box=box, ground=ground, payload=payload)
    rate = 500.0
    substeps = sim.substeps_for(rate)
    period = substeps * sim.dt

    params = dict(
        backend="sim", robot=args.robot, menagerie_root=root, control_rate_hz=rate,
        jump_dx=args.jump[0], jump_dz=args.jump[1],
        box_x_front=args.box[0] if args.box else 0.25, box_height=args.box[1] if args.box else 0.0,
        max_trials=args.max_trials, qu=args.qu, qu_stage3=args.qu3, qe=[float(v) for v in args.qe],
        reference_file=args.reference_file, log_dir=args.log_dir, run_name=args.run_name,
        resume_from=args.resume_from, transfer_from=args.transfer_from,
        transfer_mode=args.transfer_mode, auto_start=True,
        exit_when_done=True,
    )
    for item in args.param:
        name, _, value = item.partition(":=")
        params[name] = value                      # passed through verbatim
    ros_args = ["--ros-args"]
    for name, value in params.items():
        if value == "":                           # rcl rejects empty overrides; "" is the default
            continue
        ros_args += ["-p", f"{name}:={ros_value(value)}"]

    rclpy.init(args=["ilc_jump_lockstep"] + ros_args)
    node = IlcJumpNode()
    real = Reality(args, sim, period)
    if node.recorder is not None:         # what the controller is not told, for the record
        node.recorder.annotate(sim=dict(box=box, ground=ground, payload=payload,
                                        runner="ilc_jump_lockstep", reality=real.describe(args)))
    sink = _CommandSink()
    node.cmd_pub = sink

    # the true state next to what the controller measured, in each trial's recording
    # (rec_true_pos, rec_true_quat, rec_true_q): trials are scored on what happened
    record = node._record

    def record_with_truth(*a, **kw):
        record(*a, **kw)
        for k, v in (("true_pos", sim.base_position()), ("true_quat", sim.base_quat()),
                     ("true_q", sim.joint_positions()),
                     ("true_contacts", sim.foot_normal_forces())):
            node.rec.setdefault(k, []).append(v)
    node._record = record_with_truth

    def begin_trial():                            # sim_node's reset_trial, inline
        sim.reset_home()
        real.reset()
        sink.cmd = np.zeros((5, 12))
        node._set_phase("stand")
        node.stand_from = None
        node.get_logger().info(
            f"trial {node.ilc.trial + 1} (stage {node.ilc.stage()}): standing up")
    node._begin_trial = begin_trial

    viewer = None
    if args.viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(sim.model, sim.data)

    js, odom = JointState(), Odometry()
    js.name = list(CANONICAL_JOINT_NAMES)
    n_tick, t_sim, wall0 = 0, 0.0, time.monotonic()
    per_trial = node.stand_time + node.settle_time + node.T_jump + node.land_time + 1.0
    t_end = args.max_trials * per_trial + 5.0
    try:
        while t_sim < t_end:
            # publish the state (sim_node._publish), then let the controller answer it
            _stamp(js, t_sim)
            q_meas, dq_meas = real.joints()
            js.position = q_meas.tolist()
            js.velocity = dq_meas.tolist()
            js.effort = sim.applied_torques().tolist()
            _stamp(odom, t_sim)
            pos, quat, new_frame = real.pose(t_sim)
            p = odom.pose.pose
            p.position.x, p.position.y, p.position.z = (float(v) for v in pos)
            p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z = (
                float(v) for v in quat)
            if new_frame:                          # a mocap frame arrives only when taken
                node._on_pose(odom)
            node.contacts = real.foot_forces()
            node._on_joint_states(js)
            if node.worker is not None:           # the ILC update: wait, keep it deterministic
                node.worker.join()

            real.command(t_sim, sink.cmd)
            for i in range(substeps):
                sim.set_torques(real.motor_torque(t_sim + i * sim.dt))
                mujoco.mj_step(sim.model, sim.data)
            n_tick += 1
            t_sim = n_tick * period               # no accumulated rounding

            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
            if args.realtime:
                lag = t_sim - (time.monotonic() - wall0)
                if lag > 0:
                    time.sleep(lag)
        else:
            node.get_logger().warn(f"stopped at sim time {t_sim:.0f} s without finishing")
    except SystemExit:
        pass
    finally:
        history = list(node.ilc.history)
        args.run_dir = node.recorder.run_dir if node.recorder is not None else ""
        if viewer is not None:
            viewer.close()
        node.destroy_node()
        rclpy.shutdown()
    return history


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--robot", default="go2")
    ap.add_argument("--menagerie-root", default="")
    ap.add_argument("--jump", type=float, nargs=2, default=(0.5, 0.1), metavar=("DX", "DZ"),
                    help="CoM displacement, m")
    ap.add_argument("--box", type=float, nargs=2, metavar=("X_FRONT", "HEIGHT"),
                    help="box front face (m ahead of the standing CoM) and height")
    ap.add_argument("--ground", type=float, nargs=2, metavar=("KP", "KD"),
                    help="ground stiffness N/m and damping N s/m per foot (sim only)")
    ap.add_argument("--payload", type=float, default=0.0, help="kg on the trunk (sim only)")
    r = ap.add_argument_group("reality gap (sim only; see Reality)")
    r.add_argument("--mass-scale", type=float, default=1.0, help="robot mass and inertia x")
    r.add_argument("--com-offset", type=float, nargs=2, default=(0.0, 0.0), metavar=("DX", "DZ"),
                   help="trunk CoM shift, m, base frame")
    r.add_argument("--friction", type=float, default=0.0, help="ground mu (0: the MJCF's)")
    r.add_argument("--joint-friction", type=float, default=0.0, help="N m Coulomb, added")
    r.add_argument("--act-delay", type=float, default=0.0, help="s, command -> motors")
    r.add_argument("--motor-scale", type=float, default=1.0, help="torque limits x")
    r.add_argument("--motor-curve", action="store_true", help="Go1 torque-speed envelope")
    r.add_argument("--motor-knee", type=float, default=0.5,
                   help="share of the no-load speed with full torque")
    r.add_argument("--motor-speed-scale", type=float, default=1.0,
                   help="no-load speed x (battery sag)")
    r.add_argument("--pose-rate", type=float, default=0.0, help="mocap Hz (0: every tick)")
    r.add_argument("--pose-delay", type=float, default=0.0, help="mocap latency, s")
    r.add_argument("--pose-noise", type=float, nargs=2, default=(0.0, 0.0), metavar=("M", "RAD"))
    r.add_argument("--joint-noise", type=float, nargs=2, default=(0.0, 0.0),
                   metavar=("RAD", "RAD_S"))
    r.add_argument("--foot-sensor", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                   metavar=("BIAS", "GAIN_SPREAD", "NOISE"), help="raw footForce model")
    r.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qu", type=float, default=1e-4)
    ap.add_argument("--qu3", type=float, default=1e-5, help="Qu in Stage III (0: --qu)")
    ap.add_argument("--qe", type=float, nargs=6, default=(3.0, 3.0, 3.0, 0.01, 0.01, 0.01))
    ap.add_argument("--max-trials", type=int, default=25)
    ap.add_argument("--reference-file", default="")
    ap.add_argument("--log-dir", default="", help="root folder; the run goes in <log-dir>/<run-name>")
    ap.add_argument("--run-name", default="", help="run folder name (default: a timestamp)")
    ap.add_argument("--resume-from", default="", help="trial npz or run folder to continue from")
    ap.add_argument("--transfer-from", default="",
                    help="another task's trial npz or run folder to start from")
    ap.add_argument("--transfer-mode", default="retarget", choices=("retarget", "paper"))
    ap.add_argument("--summary", default="", help="JSON file for the per-trial results")
    ap.add_argument("--param", action="append", default=[], metavar="NAME:=VALUE",
                    help="any other ilc_jump parameter, e.g. --param margin:=0.8")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--realtime", action="store_true", help="pace the sim to wall time")
    args = ap.parse_args()

    history = run(args)
    if args.summary:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
        with open(args.summary, "w") as f:
            json.dump(dict(args=vars(args), run_dir=args.run_dir, history=history), f,
                      default=float, indent=1)


if __name__ == "__main__":
    main()
