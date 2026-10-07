#!/usr/bin/env python3
"""
hw_session.py RUN [--backend go1|sim]: the operator's side of deep ILC's hardware stage, for a whole session. RUN is
deploy.py --backend manual's OUT/<robot> folder (shared with the GPU machine). The runner follows deploy.py: for the
newest iteration's REQUEST.json, goal by goal, it writes the policy's plan for that goal, launches policy_jump_node for
exactly the jumps still owed, and -- on hardware -- triggers each jump when the operator presses Enter (after the robot
is reset at the start mark). Then it waits for deploy.py's next request (the ILC update, ~30-60 s, runs while the robot
is reset) and goes on, until deploy.py is done (OUT/<robot>/summary.json or its last iteration's policy) or RUN/ABORT.

    # once: the bridge, kept up all session (it damps whenever no node commands)
    ros2 launch ilc_quad go1_bridge.launch.py sdk_path:=...
    # the session
    python3 hw_session.py RUN --pose-topic /optitrack/go1/pose --pose-latency 0.006 --mocap-offset 0 0 0

At the prompt: Enter = jump, d = damp (/damp), a = abort the session (writes RUN/ABORT for deploy.py too).
--backend sim flies the same thing against sim_node (policy_jump_sim.launch.py; jumps start on their own): the test.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("run", help="deploy.py's OUT/<robot> folder")
ap.add_argument("--backend", default="go1", choices=("go1", "sim"))
ap.add_argument("--pose-topic", default="")
ap.add_argument("--pose-type", default="pose")
ap.add_argument("--pose-latency", type=float, default=0.006)
ap.add_argument("--mocap-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0])
ap.add_argument("--menagerie-root", default="/mujoco_menagerie")
ap.add_argument("--path-map", default="", metavar="HOST=LOCAL",
                help="the request's paths as this machine sees the shared folder, e.g. /home/henry/ilc_ws=/ilc_ws")
ap.add_argument("--iters", type=int, default=0, help="stop after this many iterations (0: until deploy.py is done)")
a = ap.parse_args()
if a.backend == "go1" and not a.pose_topic:
    raise SystemExit("--pose-topic is required on hardware")


def local(p):
    if a.path_map:
        src, dst = a.path_map.split("=", 1)
        if p.startswith(src):
            return dst + p[len(src):]
    return p


def load_req(f):
    r = json.load(open(f))
    r["policy"], r["episode_dir"] = local(r["policy"]), local(r["episode_dir"])
    return r


def owed(req):
    """{goal: jumps still owed} for one REQUEST.json."""
    have = {}
    for f in glob.glob(os.path.join(req["episode_dir"], "*.npz")):
        if os.path.basename(f).startswith("ref_"):
            continue
        try:
            g = gkey(np.load(f)["goal"])
            have[g] = have.get(g, 0) + 1
        except Exception:
            pass
    return {gkey(g): max(0, req["reps"] - have.get(gkey(g), 0)) for g in req["goals"]}


def gkey(g):
    """A goal -> (x, h): deploy.py writes (x, h) pairs; a bare x (older requests) is flat."""
    g = np.atleast_1d(np.asarray(g, float))
    return (round(float(g[0]), 4), round(float(g[1]) if len(g) > 1 else 0.0, 4))


def aborted():
    return os.path.exists(os.path.join(a.run, "ABORT"))


def fly(req, g, n):
    gx, gh = g
    ref = os.path.join(req["episode_dir"], f"ref_{gx:.4f}" + (f"_{gh:.4f}" if gh else "") + ".npz")
    subprocess.run([sys.executable, os.path.join(HERE, "policy_jump_node.py"), "prepare", "--policy", req["policy"],
                    "--goal", str(gx), str(gh), "--out", ref], check=True)
    bx = json.load(open(os.path.splitext(ref)[0] + "_box.json"))
    if bx["box_height"] > 0:
        print(f"   BOX for this goal: front face {bx['box_x_front']:.3f} m ahead of the standing CoM, "
              f"{bx['box_height']:.3f} m tall -- place it before the first jump", flush=True)
    common = [f"policy_dir:={req['policy']}", f"reference_file:={ref}", f"jump_dx:={gx}", f"jump_dz:={gh}",
              f"box_x_front:={bx['box_x_front']}", f"box_height:={bx['box_height']}",
              f"episode_dir:={req['episode_dir']}", f"episode_tag:={req['robot']}_it{req['it']}", f"max_trials:={n}",
              f"menagerie_root:={a.menagerie_root}"]
    if a.backend == "sim":
        r = subprocess.run(["ros2", "launch", "ilc_quad", "policy_jump_sim.launch.py", *common],
                           capture_output=True, text=True)
        for ln in r.stdout.splitlines():
            if "landing error" in ln:
                print("   " + ln.split("]: ")[-1].replace("\x1b[0m", ""), flush=True)
        return
    p = subprocess.Popen(["ros2", "launch", "ilc_quad", "policy_jump_go1.launch.py", *common, "with_bridge:=false",
                          "exit_when_done:=true", f"pose_topic:={a.pose_topic}", f"pose_type:={a.pose_type}",
                          f"pose_latency:={a.pose_latency}", "mocap_offset:=[" + ", ".join(map(str, a.mocap_offset)) + "]"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    time.sleep(8)                                         # the node loads the plan and the policy
    for i in range(n):
        while True:
            c = input(f"   goal {gx:.4f},{gh:.4f}, jump {i + 1}/{n}: reset the robot at the start mark, then Enter "
                      f"(d = damp, a = abort) > ").strip().lower()
            if c == "d":
                subprocess.run(["ros2", "service", "call", "/damp", "std_srvs/srv/Trigger"], capture_output=True)
                print("   damped")
                continue
            if c == "a":
                open(os.path.join(a.run, "ABORT"), "w").write("operator\n")
                subprocess.run(["ros2", "service", "call", "/damp", "std_srvs/srv/Trigger"], capture_output=True)
                p.terminate()
                raise SystemExit("aborted (RUN/ABORT written)")
            break
        subprocess.run(["ros2", "service", "call", "/start_trial", "std_srvs/srv/Trigger"], capture_output=True)
        while True:                                        # until this jump's episode is saved (or the node ends)
            ln = p.stdout.readline()
            if not ln:
                break
            if "landing error" in ln or "saving the jump failed" in ln:
                print("   " + ln.split("]: ")[-1].strip().replace("\x1b[0m", ""), flush=True)
                break
    p.wait(timeout=60)


done_its = 0
# stale Fast-DDS shared-memory segments (from killed ROS processes) can fill /dev/shm, after which new nodes no longer
# see each other: clean the dead ones (segments in use are kept)
try:
    subprocess.run("fastdds shm clean", shell=True, capture_output=True, timeout=60)
except Exception as err:
    print(f"(fastdds shm clean failed: {err!r}; continuing)")
print(f"session on {a.run} ({a.backend})", flush=True)
while not aborted():
    reqs = sorted(glob.glob(os.path.join(a.run, "it*", "REQUEST.json")), key=lambda f: int(f.split("/it")[-1].split("/")[0]))
    req = load_req(reqs[-1]) if reqs else None
    todo = owed(req) if req else {}
    if req and any(todo.values()):
        print(f"iteration {req['it']}: {sum(todo.values())} jumps ({', '.join(f'{g}: {n}' for g, n in todo.items() if n)})",
              flush=True)
        for g, n in todo.items():
            if n and not aborted():
                fly(req, g, n)
        done_its += 1
        if a.iters and done_its >= a.iters:
            break
        print("   flown; deploy.py is updating the policy (reset the robot meanwhile)", flush=True)
        continue
    if os.path.exists(os.path.join(os.path.dirname(a.run), "summary.json")) or \
            os.path.exists(os.path.join(a.run, "summary.json")):
        print("deploy.py is done", flush=True)
        break
    time.sleep(3)
