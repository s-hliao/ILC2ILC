#!/usr/bin/env python3
"""
sim_operator.py OUT: stands in for the hardware operator to test deploy.py --backend manual end to end -- every
OUT/<robot>/it<k>/REQUEST.json gets its jumps flown through ROS (policy_jump_node.py with sim_node in place of the
Go1 bridge: policy_jump_sim.launch.py, in the ilc_quad container), its episodes landing where deploy.py waits for
them. Exits when every robot's run has finished (summary.json) or after --timeout s.
"""
import argparse
import glob
import json
import os
import subprocess
import time

WS, CWS = "/home/henry/ilc_ws", "/ilc_ws"
cpath = lambda p: CWS + os.path.abspath(p)[len(WS):]
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--robots", type=int, default=1, help="runs to wait for (their summary.json)")
ap.add_argument("--timeout", type=float, default=6 * 3600)
ap.add_argument("--domain", type=int, default=151)
a = ap.parse_args()

done, t0 = set(), time.time()
while time.time() - t0 < a.timeout:
    if len(glob.glob(os.path.join(a.out, "*", "summary.json"))) >= a.robots:
        break
    for req in sorted(glob.glob(os.path.join(a.out, "*", "it*", "REQUEST.json"))):
        if req in done:
            continue
        r = json.load(open(req))
        for g in r["goals"]:
            ref = os.path.join(r["episode_dir"], f"ref_{g:.4f}.npz")
            cmd = (f"cd /ilc_ws && source install/setup.bash && export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID={a.domain} "
                   f"OMP_NUM_THREADS=2 && python3 install/ilc_quad/lib/ilc_quad/policy_jump_node.py prepare "
                   f"--policy {cpath(r['policy'])} --goal {g} 0 --out {cpath(ref)} && "
                   f"timeout 600 ros2 launch ilc_quad policy_jump_sim.launch.py policy_dir:={cpath(r['policy'])} "
                   f"reference_file:={cpath(ref)} jump_dx:={g} jump_dz:=0.0 episode_dir:={cpath(r['episode_dir'])} "
                   f"episode_tag:={r['robot']}_it{r['it']} max_trials:={r['reps']}")
            res = subprocess.run(["sg", "docker", "-c", f"docker exec ilc_quad bash -lc '{cmd}'"],
                                 capture_output=True, text=True)
            lines = [ln for ln in res.stdout.splitlines() if "jump " in ln and "landing error" in ln]
            print(f"{r['robot']} it {r['it']} goal {g}: " + (lines[-1].split("]: ")[-1] if lines else
                                                              f"NO JUMP (exit {res.returncode})"), flush=True)
        done.add(req)
    time.sleep(5)
