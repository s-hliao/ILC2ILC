"""Verify the simulation setup without ROS and without any control law.

Checks what `QuadModel` claims about a robot against what MuJoCo actually does:
that the actuators ended up as torque motors, that a commanded torque arrives at
the joint it was addressed to, and that foot contact forces add up to the robot's
weight when it is standing. Run it after changing anything in the model layer, or
when a controller is behaving oddly and you want the simulator ruled out.

    python -m ilc_quad.check_model --robot go2

The standing check needs the robot held up, which is the one place this file has
to apply a joint-space hold. It is a test fixture, not a controller -- nothing
here is meant to be built on.
"""

from __future__ import annotations

import argparse
import sys

import mujoco
import numpy as np

from .sim_quad_model import CANONICAL_JOINT_NAMES, QuadModel, default_menagerie_root


def check(robot: str, menagerie_root: str) -> list[str]:
    """Run the checks, returning a list of failures (empty means all passed)."""
    failures = []
    qm = QuadModel(robot, menagerie_root)
    print(f"\n{qm}")
    print(f"  scene        {qm.scene_path}")
    print(f"  canonical    {', '.join(CANONICAL_JOINT_NAMES[:3])}, ...")
    print(f"  act_idx      {qm.act_idx.tolist()}")
    print(f"  torque limit {qm.torque_limit.tolist()}")
    print(f"  joint range  hip {qm.joint_range[0].round(3).tolist()}  "
          f"thigh {qm.joint_range[1].round(3).tolist()}  "
          f"calf {qm.joint_range[2].round(3).tolist()}")

    # 1. Every actuator is a direct torque motor. QuadModel asserts this at load,
    #    so reaching here at all is most of the check; confirm the numbers too.
    m = qm.model
    bad = [i for i in range(m.nu)
           if m.actuator_biastype[i] != mujoco.mjtBias.mjBIAS_NONE
           or m.actuator_gainprm[i, 0] != 1.0]
    if bad:
        failures.append(f"actuators {bad} are not torque motors")
    else:
        print(f"  [ok] all {m.nu} actuators are direct torque motors")

    # 2. A torque command reaches the joint it names, and saturates at the limit.
    qm.reset_home()
    probe = np.linspace(-5.0, 5.0, 12)
    qm.set_torques(probe)
    mujoco.mj_step(m, qm.data)
    if not np.allclose(qm.applied_torques(), probe):
        failures.append(
            f"applied torque {qm.applied_torques().round(3)} != commanded {probe}"
        )
    else:
        print("  [ok] commanded torque arrives unchanged at each joint")

    if not np.allclose(qm.set_torques(np.full(12, 1e4)), qm.torque_limit):
        failures.append("torque command is not clipped to the model's limits")
    else:
        print("  [ok] torque commands clip to the model's limits")

    # 3. Ground reaction forces account for the robot's weight while standing.
    #    A joint-space hold is needed only to keep it upright for the measurement.
    qm.reset_home()
    q0 = qm.home_qpos
    for _ in range(1500):
        hold = 60.0 * (q0 - qm.joint_positions()) - 2.0 * qm.joint_velocities()
        qm.set_torques(hold)
        mujoco.mj_step(m, qm.data)

    grf = qm.foot_normal_forces()
    weight = qm.total_mass * 9.81
    ratio = grf.sum() / weight
    print(f"  standing at z={qm.base_position()[2]:.3f} m, GRF {grf.round(1)} "
          f"sum {grf.sum():.1f} N vs weight {weight:.1f} N (ratio {ratio:.3f})")
    if not 0.98 < ratio < 1.02:
        failures.append(
            f"standing GRF is {ratio:.3f} of body weight; contact sensing or the "
            "foot geom lookup is wrong"
        )
    else:
        print("  [ok] foot contact forces account for body weight")

    # 4. The control rate the sim node defaults to divides the timestep.
    try:
        substeps = qm.substeps_for(250.0)
        print(f"  [ok] 250 Hz control = {substeps} physics steps of "
              f"{qm.dt * 1e3:.1f} ms")
    except ValueError as exc:
        failures.append(str(exc))

    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--robot", default="all", choices=("all", "go2", "go1"))
    ap.add_argument("--menagerie-root", default=default_menagerie_root())
    args = ap.parse_args(argv)

    robots = ("go2", "go1") if args.robot == "all" else (args.robot,)
    failures = {}
    for robot in robots:
        found = check(robot, args.menagerie_root)
        if found:
            failures[robot] = found

    print()
    if failures:
        for robot, found in failures.items():
            for f in found:
                print(f"FAIL [{robot}] {f}")
        return 1
    print(f"all checks passed for {', '.join(robots)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
