#!/usr/bin/env python3
"""
policy_jump_node.py: the learned goal-conditioned force policy flying jumps on the robot -- the executor of deep ILC's
hardware stage (ilc_mjx/scripts/deploy.py --backend manual). It is IlcJumpNode (the same controller, estimator and
safety: fall detection and damping, the bridges' watchdog, joint limits and clipping) with the policy as its
force_policy (dilc_execute.Driver, exactly what the sim robots fly) and no learning: every /start_trial flies one jump
of (jump_dx, jump_dz), then the jump is saved as an episode in dilc_execute's format (what deploy.py reads) to
episode_dir, and the node waits for the next /start_trial.

Parameters on top of ilc_jump_sim's:
    policy_dir      the policy folder deploy.py exported (bank.json, manifest.json, policy.npz)
    policy_member   its member (default "policy")
    episode_dir     where each jump's episode goes (policy_g<dx>_<dz>_<episode_tag>_t<trial>.npz)
    episode_tag     e.g. the robot's name
reference_file must be the policy's plan for the goal (the bank's interpolation, which the policy was trained on):
`prepare` writes it, and the node refuses any other.

    python3 policy_jump_node.py prepare --policy DIR --goal 0.5 0 --out /tmp/ref_0.500.npz   # (or --goal 0.52 0.12: a box;
                                                     # its placement goes to /tmp/ref_..._box.json -> box_x_front, box_height)
    ros2 launch ilc_quad policy_jump_go1.launch.py pose_topic:=... policy_dir:=DIR reference_file:=/tmp/ref_0.500.npz \
        jump_dx:=0.5 jump_dz:=0 episode_dir:=DIR/real episode_tag:=go1
    ros2 service call /start_trial std_srvs/srv/Trigger          # one jump
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def prepare(argv):
    from dilc_execute import GoalBank
    ap = argparse.ArgumentParser(prog="policy_jump_node.py prepare")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--goal", type=float, nargs=2, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    bank = GoalBank(json.load(open(os.path.join(a.policy, "bank.json"))))
    ref, box, extrap = bank.reference(np.asarray(a.goal, float))
    np.savez(a.out, **ref)
    # the box this plan jumps onto (none: flat): the node's box_x_front / box_height, and where to put the real box --
    # its front face box_x_front ahead of the robot's standing CoM, box_height tall
    side = os.path.splitext(a.out)[0] + "_box.json"
    json.dump(dict(goal=list(map(float, a.goal)), box_x_front=float(box["x_front"]) if box else 0.25,
                   box_height=float(box["height"]) if box else 0.0), open(side, "w"), indent=1)
    print(f"{a.out}: the plan for goal {a.goal}{' (EXTRAPOLATED past the bank: fly with care)' if extrap else ''}"
          + (f"; BOX: front face {box['x_front']:.3f} m ahead of the standing CoM, {box['height']:.3f} m tall ({side})"
             if box else " (flat)"))


def node_main(argv=None):
    import rclpy

    from dilc_execute import Driver, EpisodeMaker, load_policy
    from ilc_jump_sim import IlcJumpNode

    class PolicyJumpNode(IlcJumpNode):
        def __init__(self):
            super().__init__()
            for name, default in (("policy_dir", ""), ("policy_member", "policy"), ("episode_dir", ""),
                                  ("episode_tag", "robot")):
                self.declare_parameter(name, default)
            gp = lambda n: self.get_parameter(n).value
            if not gp("policy_dir") or not gp("episode_dir"):
                raise SystemExit("policy_jump_node: policy_dir and episode_dir are required")
            self.bank, self.actor, _, _ = load_policy(gp("policy_dir"), gp("policy_member"))
            self.goal = np.array([float(gp("jump_dx")), float(gp("jump_dz"))])
            ref_file = gp("reference_file")
            if not ref_file or not os.path.exists(ref_file):
                raise SystemExit("policy_jump_node: reference_file must exist (policy_jump_node.py prepare)")
            self.ref = dict(np.load(ref_file))
            own, _, _ = self.bank.reference(self.goal)
            if not (np.allclose(self.ilc.x_ref, self.ref["x_ref"]) and np.allclose(own["x_ref"], self.ref["x_ref"])):
                raise SystemExit("policy_jump_node: reference_file is not the policy's plan for this goal")
            self.maker = EpisodeMaker(self.bank, gp("menagerie_root"))
            self.episode_dir, self.episode_tag = gp("episode_dir"), gp("episode_tag")
            os.makedirs(self.episode_dir, exist_ok=True)
            self.n_flown = 0
            # a session (hw_session.py) launches one node per goal for the jumps it owes: exit when they are flown,
            # on hardware too (the bridge, launched apart, damps once the commands stop)
            self.exit_when_done = bool(gp("exit_when_done"))
            self._new_driver()
            self.get_logger().info(f"policy {gp('policy_dir')} ({gp('policy_member')}) flies goal {self.goal}; "
                                   f"episodes -> {self.episode_dir}")

        def _new_driver(self):
            self.force_policy = Driver(self.bank, self.goal, self.ref, actor=self.actor,
                                       window=self.bank.cfg["est_window"])

        def _update_worker(self):                    # no learning: save the jump, ready for the next
            try:
                log, rec, _ = self._trial_log()
                drv = self.force_policy
                fell = bool(self.fell)
                ep = self.maker.make(self.goal, self.ref, drv.X, drv.U, log["X"], fell, rec)
                self.n_flown += 1
                name = (f"policy_g{self.goal[0]:.3f}_{self.goal[1]:.3f}_{self.episode_tag}"
                        f"_t{self.n_flown:03d}.npz")
                np.savez(os.path.join(self.episode_dir, name), **ep,
                         **{f"rec_{k}": np.asarray(v) for k, v in rec.items()})
                e = ep["info"]
                self.get_logger().info(
                    f"jump {self.n_flown}: landing error x {e[0] * 100:+.1f} cm, z {e[1] * 100:+.1f} cm, "
                    f"pitch {np.degrees(e[2]):+.1f} deg{' -- FELL' if fell else ''}; saved {name}")
                self.last_result = dict(converged=False)
            except Exception as err:                  # never leave the robot in 'update'
                self.get_logger().error(f"saving the jump failed: {err!r}")
                self.last_result = dict(converged=False, failed=True)
            finally:
                self.ilc.trial += 1
                self._new_driver()

    rclpy.init(args=argv)
    node = None
    try:
        node = PolicyJumpNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prepare":
        prepare(sys.argv[2:])
    else:
        node_main()
