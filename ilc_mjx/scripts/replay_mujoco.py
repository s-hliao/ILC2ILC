#!/usr/bin/env python3
"""
replay_mujoco.py TRAJ [--robot R] [--iteration K] [--goal X H] [--index I] [--all] [--speed S] [--check]:
replay recorded Go1 jumps (export_trajectories.py) in the MuJoCo viewer -- the ground truth the CPU robot recorded,
posed kinematically (qpos per 2 ms tick), with the jump's box in the scene. TRAJ: a path, or a name in ../trajectories
(e.g. eval_ours_24jumps).

  --list               what can be replayed (no MuJoCo needed): without TRAJ every recording in ../trajectories (what it
                       is, robots, goals, jumps); with TRAJ its robots, goals and trials, each with its landing error
                       (ok = no fall, |ex| <= 5 cm, |ez| <= 3 cm), and the command to replay it

  --index I            one jump; otherwise every jump matching --robot / --iteration / --goal, one after another
  --all                with no filter: every jump in the file
  --speed S            playback speed (1 = real time, 0.25 = quarter-speed slow motion)
Viewer (one window for all selected jumps): space pause / resume, ',' / '.' step 10 ms back / forward, R restart,
N / P next / previous jump (each jump loops until then), Q or closing the window quits.
  --check              no viewer: pose every selected jump and report the feet's heights at the start and at the
                       landing (a sanity check of the export and the box placement; runs headless)
  --record OUT         no viewer: render the selected jumps offscreen (side camera) into OUT -- .gif (Pillow) or .mp4
                       (needs imageio-ffmpeg); --size W H (even numbers), --fps (default 30). Needs OpenGL (a desktop; on a headless
                       machine MUJOCO_GL=egl or osmesa if installed)
Examples:
  replay_mujoco.py --list                                                             # every recording
  replay_mujoco.py --list eval_ours_24jumps                                           # its robots / goals / trials
  replay_mujoco.py trajectories/gate_loose.npz --robot real_r1 --goal 0.5 0.15        # its 4 hardware-stage tries
  replay_mujoco.py trajectories/eval_ppo_dr.npz --robot real_r4 --goal 0.54 0.14      # a baseline's transfer
Needs a display for the viewer, and a mujoco_menagerie checkout ($MUJOCO_MENAGERIE_PATH or ~/mujoco_menagerie).
"""
import argparse, json, os, sys, textwrap, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traj_lib as tl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("traj", nargs="?")
ap.add_argument("--list", action="store_true")
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
TRDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trajectories"))
TP = dict(blk15="1.5 cm block under the feet", blk2="2 cm block under the feet", crouch="crouched start",
          tall="tall start", noseup="nose-up start", nosedn="nose-down start", mocapbad="bad mocap (120 Hz, 15 ms)",
          delay10="10 ms actuation delay")
DESC = {
    "gate_loose": "ours: hardware stage (deployment recipe), 6 training goals x 4 batches",
    "hwstage_drA_hw24": "our learner + DR (A: DR fine-tune): hardware stage, 24 jumps",
    "hwstage_drB_hw24": "our learner + DR (B: DR from scratch): hardware stage, 24 jumps",
    "hwstage_rma_crosstrial_calibration": "RMA, cross-trial: its 6 calibration jumps",
    "hwstage_fada_lora": "FADA: on-robot LoRA adaptation",
    "hwstage_jumpilc_pergoal": "per-goal ILC (JumpILC): trials flown on the test goals themselves",
    "eval_ours_zeroshot": "TEST GOALS -- ours, zero-shot",
    "eval_ours_24jumps": "TEST GOALS -- ours, after 24 real jumps",
    "eval_learner_dr": "TEST GOALS -- our learner + DR (A), zero-shot",
    "eval_dr_finetune_24jumps": "TEST GOALS -- our learner + DR (A), after 24 real jumps",
    "eval_dr_scratch_24jumps": "TEST GOALS -- our learner + DR (B), after 24 real jumps",
    "eval_ppo_dr": "TEST GOALS -- PPO + DR, zero-shot",
    "eval_rma": "TEST GOALS -- RMA, zero-shot",
    "eval_rma_crosstrial": "TEST GOALS -- RMA, cross-trial, after calibration",
    "eval_fada": "TEST GOALS -- FADA, after adaptation",
    **{f"hwstage_tp_{c}": f"ours: hardware stage under a constant perturbation: {t}" for c, t in TP.items()},
    **{f"eval_tp_{c}": f"TEST GOALS -- ours after the hardware stage under '{c}' (nominal robot)" for c in TP},
}


def resolve(p):
    for c in (p, os.path.join(TRDIR, p), os.path.join(TRDIR, p + ".npz")):
        if os.path.isfile(c):
            return c
    raise SystemExit(f"no recording {p!r}: see replay_mujoco.py --list")


def meta_of(path):                                # only the meta array (npz members load lazily)
    return json.loads(str(np.load(path)["meta"]))


ok = lambda m: not m["fell"] and abs(m["ex"]) <= 0.05 and abs(m["ez"]) <= 0.03
gstr = lambda g: f"{g[0]:.4g} {g[1]:.3g}"
here = os.path.relpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_mujoco.py"))
if a.list and not a.traj:
    files = sorted(f for f in os.listdir(TRDIR) if f.endswith(".npz"))
    print(f"{len(files)} recordings in {TRDIR} (replay with: python {here} <name> --robot R [--goal X H])\n")
    for f in sorted(files, key=lambda f: (not f.startswith("eval_"), f)):
        M = meta_of(os.path.join(TRDIR, f))
        name = f[:-4]
        rob = sorted({m["robot"] for m in M})
        goals = {tuple(np.round(m["goal"], 4)) for m in M}
        its = sorted({m["iteration"] for m in M if m["iteration"] >= 0})
        print(f"{name:38s} {DESC.get(name, '')}")
        print(f"{'':38s}   {len(M)} jumps; robots {' '.join(r.replace('real_', '') for r in rob)}; {len(goals)} goals"
              + (f"; {len(its)} iterations / trials" if its else "") + f"; ok {sum(map(ok, M))}/{len(M)}")
    print(f"\ndetails of one: python {here} --list <name>")
    sys.exit(0)
if not a.traj:
    raise SystemExit("give a recording (see --list)")
a.traj = resolve(a.traj)
if a.list:
    M = meta_of(a.traj)
    name = os.path.splitext(os.path.basename(a.traj))[0]
    print(f"{name}: {DESC.get(name, '')} -- {len(M)} jumps (ok = no fall, |ex| <= 5 cm, |ez| <= 3 cm)")
    for rb in sorted({m["robot"] for m in M}):
        R = [(i, m) for i, m in enumerate(M) if m["robot"] == rb]
        print(f"\n{rb}  ({sum(ok(m) for _, m in R)}/{len(R)} ok)    all of them: python {here} {name} --robot {rb} --speed 0.5")
        for g in sorted({tuple(np.round(m["goal"], 4)) for _, m in R}, key=lambda g: (g[1], g[0])):
            G = [(i, m) for i, m in R if np.allclose(m["goal"], g, atol=5e-4)]
            cells = [(f"it{m['iteration']}\u00a0" if m["iteration"] >= 0 else "") +
                     ("FELL" if m["fell"] else f"{100 * m['ex']:+.1f}/{100 * m['ez']:+.1f}") + ("\u00a0ok" if ok(m) else "")
                     for _, m in sorted(G, key=lambda x: x[1]["iteration"])]
            box = f"box {g[1] * 100:.1f} cm" if g[1] > 0.004 else "flat"
            head = f"  --goal {gstr(g):12s} {box:12s} ex/ez (cm): "      # (no-break spaces keep each trial on one line)
            print(textwrap.fill(", ".join(cells), width=118, initial_indent=head, subsequent_indent=" " * len(head),
                                break_long_words=False, break_on_hyphens=False).replace("\u00a0", " "))
    sys.exit(0)
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
        r.close()
        print(f"[{i}] {mt['robot']} goal {mt['goal']}: {len(frames)} frames so far")
    if a.record.endswith(".mp4"):                 # H.264 / yuv420p: pausable and seekable in any player
        import imageio_ffmpeg
        h, w = frames[0].shape[:2]
        wr = imageio_ffmpeg.write_frames(a.record, (w, h), fps=a.fps, codec="libx264", quality=None,
                                         output_params=["-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart"])
        wr.send(None)
        for f in frames:
            wr.send(np.ascontiguousarray(f))
        wr.close()
    else:
        from PIL import Image
        ims = [Image.fromarray(f) for f in frames]
        ims[0].save(a.record, save_all=True, append_images=ims[1:], duration=int(1000 / a.fps), loop=0)
    print(f"-> {a.record}")
    sys.exit(0)

import mujoco.viewer  # noqa: E402
# ONE viewer window for every selected jump (re-creating the GLFW window per jump crashes on some desktops, e.g.
# Wayland); each jump's box is moved / resized in place (the replay is kinematic, so only its drawing matters).
# Keys: space pause / resume, ',' and '.' step 10 ms back / forward, R restart, N next jump, P previous jump
# (each jump loops until then), Q or closing the window quits.
KEY = dict(space=32, comma=44, period=46, n=78, p=80, r=82, q=81)
m, d = tl.go1_model((0.3, 0.1))
gbox = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box")
dt = float(T["t"][1] - T["t"][0])
st = dict(j=0, k=0, paused=False, quit=False, shown=-1)


def on_key(c):
    n = T["meta"][idx[st["j"]]]["n_ticks"]
    if c == KEY["space"]:
        st["paused"] = not st["paused"]
    elif c == KEY["comma"]:
        st["k"] = max(0, st["k"] - 5)
    elif c == KEY["period"]:
        st["k"] = min(n - 1, st["k"] + 5)
    elif c == KEY["r"]:
        st["k"] = 0
    elif c == KEY["n"]:
        st["j"], st["k"] = (st["j"] + 1) % len(idx), 0
    elif c == KEY["p"]:
        st["j"], st["k"] = (st["j"] - 1) % len(idx), 0
    elif c == KEY["q"]:
        st["quit"] = True


print(f"{len(idx)} jump(s)   [space pause, , . step, R restart, N next, P previous, Q quit]", flush=True)
with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as v:
    v.cam.distance, v.cam.elevation, v.cam.azimuth = 1.8, -12, 90
    hold = 0
    while v.is_running() and not st["quit"]:
        t0 = time.time()
        i = idx[st["j"]]
        mt = T["meta"][i]
        n = mt["n_ticks"]
        if st["shown"] != st["j"]:                                  # a new jump: place its box, report it
            st["shown"], hold = st["j"], 0
            xf, h = tl.box_of(mt)
            with v.lock():
                if xf is not None and h >= 0.005:
                    m.geom_size[gbox] = [0.5, 0.5, h / 2]
                    m.geom_pos[gbox] = [xf + 0.5, 0.0, h / 2]
                    m.geom_rgba[gbox, 3] = 1.0
                else:
                    m.geom_rgba[gbox, 3] = 0.0                       # flat goal: hide the box
            v.cam.lookat[:] = [float(T["pos"][i, 0, 0]) + 0.3, 0, 0.25]
            print(f"[{st['j'] + 1}/{len(idx)}] #{i} {mt['robot']} ({mt.get('cond')}) it {mt['iteration']} goal {mt['goal']}: "
                  f"ex {100 * mt['ex']:+.1f} cm ez {100 * mt['ez']:+.1f} cm{' FELL' if mt['fell'] else ''}", flush=True)
        with v.lock():
            tl.set_pose(m, d, T, i, st["k"])
            mujoco.mj_forward(m, d)
        v.sync()
        if not st["paused"]:
            if st["k"] < n - 1:
                st["k"] += 1
            elif (hold := hold + 1) * dt > 1.0:                    # hold the landing 1 s, then loop
                st["k"], hold = 0, 0
        time.sleep(max(0.0, dt / a.speed - (time.time() - t0)))
sys.stdout.flush()
os._exit(0)                                       # skip GLFW's interpreter-exit teardown (segfaults on some Wayland setups)
