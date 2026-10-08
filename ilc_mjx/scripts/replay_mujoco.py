#!/usr/bin/env python3
"""
replay_mujoco.py TRAJ.npz [--robot R] [--iteration K] [--goal X H] [--index I] [--all] [--speed S] [--check]:
replay recorded Go1 jumps (export_trajectories.py) in the MuJoCo viewer -- the ground truth the CPU robot recorded,
posed kinematically (qpos per 2 ms tick), with the jump's box in the scene.

  --index I            one jump; otherwise every jump matching --robot / --iteration / --goal, one after another
  --all                with no filter: every jump in the file
  --speed S            playback speed (1 = real time)
  --check              no viewer: pose every selected jump and report the feet's heights at the start and at the
                       landing (a sanity check of the export and the box placement; runs headless)
  --record OUT         no viewer: render the selected jumps offscreen (side camera) into OUT -- .gif (Pillow) or .mp4
                       (needs imageio-ffmpeg); --size W H, --fps (default 30). Needs OpenGL (a desktop; on a headless
                       machine MUJOCO_GL=egl or osmesa if installed)
Examples:
  replay_mujoco.py trajectories/gate_loose.npz --robot real_r1 --goal 0.5 0.15        # its 4 hardware-stage tries
  replay_mujoco.py trajectories/eval_ppo_dr.npz --robot real_r4 --goal 0.54 0.14      # a baseline's transfer
Needs a display for the viewer, and a mujoco_menagerie checkout ($MUJOCO_MENAGERIE_PATH or ~/mujoco_menagerie).
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traj_lib as tl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("traj")
ap.add_argument("--robot")
ap.add_argument("--iteration", type=int)
ap.add_argument("--goal", type=float, nargs=2)
ap.add_argument("--index", type=int)
ap.add_argument("--all", action="store_true")
ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--check", action="store_true")
ap.add_argument("--record")
ap.add_argument("--size", type=int, nargs=2, default=[640, 400])
ap.add_argument("--fps", type=int, default=30)
a = ap.parse_args()
T = tl.load(a.traj)
if a.index is not None:
    idx = [a.index]
else:
    idx = tl.select(T, robot=a.robot, iteration=a.iteration, goal=a.goal)
    if not (a.robot or a.iteration is not None or a.goal or a.all):
        idx = idx[:1]
if not idx:
    raise SystemExit("no jump matches")
import mujoco  # noqa: E402

if a.check:
    for i in idx:
        mt = T["meta"][i]
        m, d = tl.go1_model(tl.box_of(mt))
        n = mt["n_ticks"]
        feet = lambda k: [round(float(tl.side_points(m, d, T, i, k)["legs"][l][2, 1]), 3) for l in tl.LEGS]
        last = feet(n - 1)
        fx = [round(float(tl.side_points(m, d, T, i, n - 1)["legs"][l][2, 0]), 3) for l in tl.LEGS]
        print(f"[{i}] {mt['robot']} it {mt['iteration']} goal {mt['goal']} box {tl.box_of(mt)}: feet z start {feet(0)} "
              f"end {last} (feet x end {fx}); ex {100 * mt['ex']:+.1f} cm, fell {mt['fell']}")
    sys.exit(0)

if a.record:
    frames = []
    for i in idx:
        mt = T["meta"][i]
        m, d = tl.go1_model(tl.box_of(mt))
        m.vis.global_.offwidth, m.vis.global_.offheight = max(a.size[0], 640), max(a.size[1], 480)
        r = mujoco.Renderer(m, a.size[1], a.size[0])
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = 1.9, -10, 90
        dt = float(T["t"][1] - T["t"][0])
        step = max(1, int(round(a.speed / (a.fps * dt))))
        for k in list(range(0, mt["n_ticks"], step)) + [mt["n_ticks"] - 1] * int(0.5 * a.fps):
            tl.set_pose(m, d, T, i, k)
            mujoco.mj_forward(m, d)
            cam.lookat[:] = [float(T["pos"][i, 0, 0]) + 0.35, 0.0, 0.25]
            r.update_scene(d, cam)
            frames.append(r.render().copy())
        print(f"[{i}] {mt['robot']} goal {mt['goal']}: {len(frames)} frames so far")
    if a.record.endswith(".mp4"):
        import imageio.v2 as iio
        iio.mimsave(a.record, frames, fps=a.fps)
    else:
        from PIL import Image
        ims = [Image.fromarray(f) for f in frames]
        ims[0].save(a.record, save_all=True, append_images=ims[1:], duration=int(1000 / a.fps), loop=0)
    print(f"-> {a.record}")
    sys.exit(0)

import mujoco.viewer  # noqa: E402
for i in idx:
    mt = T["meta"][i]
    m, d = tl.go1_model(tl.box_of(mt))
    print(f"[{i}] {mt['robot']} ({mt.get('cond')}) it {mt['iteration']} goal {mt['goal']}: ex {100 * mt['ex']:+.1f} cm "
          f"ez {100 * mt['ez']:+.1f} cm{' FELL' if mt['fell'] else ''}")
    dt = float(T["t"][1] - T["t"][0])
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 1.8, -12, 90
        v.cam.lookat[:] = [0.3, 0, 0.25]
        for k in range(mt["n_ticks"]):
            if not v.is_running():
                break
            t0 = time.time()
            tl.set_pose(m, d, T, i, k)
            mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(max(0.0, dt / a.speed - (time.time() - t0)))
        time.sleep(0.8)
