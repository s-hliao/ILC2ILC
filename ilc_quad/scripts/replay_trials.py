#!/usr/bin/env python3
"""
Replay recorded ILC trials in the MuJoCo viewer (or render them to a video).

Every trial file a TrialRecorder run writes carries the flown MuJoCo state at each
control tick (rec_t, rec_pos, rec_quat, rec_q). This plays those states back
kinematically in the same scene -- robot, box, payload -- with no physics, so what
you see is exactly what was flown.

    replay_trials.py RUN_DIR                     # every trial of the run, in order
    replay_trials.py RUN_DIR --trials 1 10 20    # just these
    replay_trials.py RUN_DIR --trials last --speed 0.25 --loop
    replay_trials.py RUN_DIR/trial_007.npz       # one file
    replay_trials.py RUN_DIR --trials 1 20 --video jump.mp4   # offscreen, no display

Viewer keys: space pause/resume, right/left arrow next/previous trial, backspace
restart the trial.
"""
import argparse
import glob
import json
import os
import sys
import time

import mujoco
import numpy as np

try:
    from ilc_quad.sim_quad_model import QuadModel
except ImportError:                     # run from the source tree, outside ROS
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from ilc_quad.sim_quad_model import QuadModel

KEY_SPACE, KEY_RIGHT, KEY_LEFT, KEY_BACKSPACE = 32, 262, 263, 259


def trial_files(path, trials):
    if path.endswith(".npz"):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "trial_*.npz")))
    if not files:
        raise SystemExit(f"no trial_*.npz under {path}")
    if not trials:
        return files
    by_num = {int(os.path.basename(f)[6:9]): f for f in files}
    out = []
    for t in trials:
        n = max(by_num) if t == "last" else int(t)
        if n not in by_num:
            raise SystemExit(f"no trial {n} in {path} (have {min(by_num)}..{max(by_num)})")
        out.append(by_num[n])
    return out


def scene_for(files):
    """The QuadModel the run flew in: robot and box from the plan, ground and payload
    from the run's meta.json (the sim conditions)."""
    d = np.load(files[0], allow_pickle=True)
    cfg = json.loads(str(d["config"]))
    sim = {}
    meta = os.path.join(os.path.dirname(os.path.abspath(files[0])), "meta.json")
    if os.path.exists(meta):
        sim = json.load(open(meta)).get("sim") or {}
    box = sim.get("box", cfg.get("box"))
    return QuadModel(cfg["robot"], box=box, ground=sim.get("ground"),
                     payload=sim.get("payload"))


def load(f):
    d = np.load(f, allow_pickle=True)
    if "rec_t" not in d.files:
        raise SystemExit(f"{f} has no recorded MuJoCo states (rec_*)")
    res = json.loads(str(d["result"])) if "result" in d.files else {}
    label = f"trial {int(d['trial'])}" if "trial" in d.files else os.path.basename(f)
    if res:
        label += (f"  stage {res.get('stage')}  miss {100 * res['pos_err']:.1f} cm"
                  f"  pitch {np.degrees(res['theta_err']):.1f} deg"
                  + ("  FELL" if res.get("fell") else ""))
    return dict(t=d["rec_t"] - d["rec_t"][0], pos=d["rec_pos"], quat=d["rec_quat"],
                q=d["rec_q"], label=label)


def set_state(qm, tr, i):
    qm.data.qpos[0:3] = tr["pos"][i]
    qm.data.qpos[3:7] = tr["quat"][i]
    qm.data.qpos[qm.qpos_adr] = tr["q"][i]
    mujoco.mj_forward(qm.model, qm.data)


def side_camera(cam, box):
    """Side view of the sagittal plane, framing the take-off spot and the box."""
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.25 if box is None else 0.5 * box["x_front"] + 0.15, 0.0, 0.25]
    cam.distance, cam.azimuth, cam.elevation = 1.8, 90.0, -10.0


def play_viewer(qm, trials, speed, loop):
    from mujoco import viewer as mj_viewer
    state = dict(paused=False, jump=0, restart=False)

    def on_key(key):
        if key == KEY_SPACE:
            state["paused"] = not state["paused"]
        elif key == KEY_RIGHT:
            state["jump"] = 1
        elif key == KEY_LEFT:
            state["jump"] = -1
        elif key == KEY_BACKSPACE:
            state["restart"] = True

    with mj_viewer.launch_passive(qm.model, qm.data, key_callback=on_key) as v:
        side_camera(v.cam, qm.box)
        k = 0
        while v.is_running() and 0 <= k < len(trials):
            tr = trials[k]
            print(tr["label"], flush=True)
            if hasattr(v, "set_texts"):
                v.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, tr["label"], ""))
            t_wall, t_sim, i = time.perf_counter(), 0.0, 0
            while v.is_running() and i < len(tr["t"]):
                if state["jump"] or state["restart"]:
                    break
                if state["paused"]:
                    time.sleep(0.02)
                    t_wall = time.perf_counter() - t_sim / speed
                    continue
                t_sim = (time.perf_counter() - t_wall) * speed
                i = int(np.searchsorted(tr["t"], t_sim))
                set_state(qm, tr, min(i, len(tr["t"]) - 1))
                v.sync()
                time.sleep(1 / 120)
            if state["restart"]:
                state["restart"] = False
                continue
            skipped = state["jump"] != 0
            k = max(0, k + (state["jump"] or 1))
            state["jump"] = 0
            if loop and k >= len(trials):
                k = 0
            if not skipped:
                time.sleep(0.4)              # a beat between trials


def render_video(qm, trials, speed, out, fps=60, size=(960, 540)):
    import imageio.v2 as imageio
    qm.model.vis.global_.offwidth = max(qm.model.vis.global_.offwidth, size[0])
    qm.model.vis.global_.offheight = max(qm.model.vis.global_.offheight, size[1])
    r = mujoco.Renderer(qm.model, height=size[1], width=size[0])
    cam = mujoco.MjvCamera()
    side_camera(cam, qm.box)
    writer = imageio.get_writer(out, fps=fps)
    for tr in trials:
        print(tr["label"], flush=True)
        for t in np.arange(0.0, tr["t"][-1] / speed, 1.0 / fps):
            set_state(qm, tr, min(int(np.searchsorted(tr["t"], t * speed)), len(tr["t"]) - 1))
            r.update_scene(qm.data, cam)
            writer.append_data(r.render())
    writer.close()
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a run folder (TrialRecorder output) or one trial_NNN.npz")
    ap.add_argument("--trials", nargs="*", default=[], help="trial numbers, or 'last' (default: all)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback rate, 0.25 = 4x slow motion")
    ap.add_argument("--loop", action="store_true", help="start over after the last trial")
    ap.add_argument("--video", default="", help="render to this file (mp4/gif) instead of the viewer")
    args = ap.parse_args()

    files = trial_files(args.path, args.trials)
    qm = scene_for(files)
    trials = [load(f) for f in files]
    if args.video:
        render_video(qm, trials, args.speed, args.video)
    else:
        play_viewer(qm, trials, args.speed, args.loop)


if __name__ == "__main__":
    main()
