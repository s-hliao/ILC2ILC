#!/usr/bin/env python3
"""ILC jumping against MuJoCo in one process, in lockstep: for tests and weight sweeps.

    ilc_jump_lockstep.py --jump 0.6 0.0 --ground 2e3 5e2 --log-dir /ilc_ws/log/soft
    ilc_jump_lockstep.py --jump 0.5 0.2 --box 0.25 0.2 --qu 1e-5 --qe 1 3 3 .01 .01 .01
    ilc_jump_lockstep.py --jump 0.6 0.0 --payload 2.0 --viewer --realtime

It runs the very same controller as the launch file -- `IlcJumpNode` from
ilc_jump_sim.py, its phases, gains, feedforward and ILC update -- but feeds it
MuJoCo state by calling its callbacks directly instead of over topics, and steps
the physics only once it has answered. Every tick is therefore exactly what
sim_node + ROS give when nothing is late (one control period of delay, the PD
re-evaluated every physics step), with no dependence on machine load. Trials
run as fast as the CPU allows, so several runs can go in parallel.

The ground stiffness and payload go to the simulated robot only; the controller
builds its own nominal QuadModel and never learns of them (as in the paper,
Sec. III-A). `--summary` writes the per-trial results as JSON.
"""

from __future__ import annotations

import argparse
import json
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
    if node.recorder is not None:         # what the controller is not told, for the record
        node.recorder.annotate(sim=dict(box=box, ground=ground, payload=payload,
                                        runner="ilc_jump_lockstep"))
    sink = _CommandSink()
    node.cmd_pub = sink

    def begin_trial():                            # sim_node's reset_trial, inline
        sim.reset_home()
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
            js.position = sim.joint_positions().tolist()
            js.velocity = sim.joint_velocities().tolist()
            js.effort = sim.applied_torques().tolist()
            _stamp(odom, t_sim)
            pos, quat = sim.base_position(), sim.base_quat()
            p = odom.pose.pose
            p.position.x, p.position.y, p.position.z = (float(v) for v in pos)
            p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z = (
                float(v) for v in quat)
            node._on_pose(odom)
            node.contacts = sim.foot_normal_forces()
            node._on_joint_states(js)
            if node.worker is not None:           # the ILC update: wait, keep it deterministic
                node.worker.join()

            for _ in range(substeps):
                sim.set_torques(pd_torque(sink.cmd, sim.joint_positions(),
                                          sim.joint_velocities()))
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
