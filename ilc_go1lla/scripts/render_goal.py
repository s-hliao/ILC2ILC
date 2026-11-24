"""GOAL-TO-GOAL video (2026-11-19, render_walk.py adapted to the stage_goal.py task): zero-shot vs after the DistILC stage,
same perturbed robot, same held-out goals (seed 300, the first --clips are shown, no picking), the standing controller takes
over at the goal, falls judged relative to the feet (foam), foam drawn as a slab. Closing card: every held-out walk's time
to goal / fall, and the nominal robot's time to goal.

Original description (render_walk.py):
Side-by-side video: the zero-shot walker vs the walker after the DistILC stage, on the same perturbed robot and goals.

Each clip replays one held-out walk (the stage's evaluation goals, seed 300) in both panels from the same camera. The
perturbation is drawn (payload as a yellow box on the trunk); the reference path, its moving reference point and the
goal are drawn; counters show time, speed, distance to the goal and FELL. A closing card lists every held-out walk.

    MUJOCO_GL=egl ~/venvs/pg0/bin/python scripts/render_walk.py --case payload4R --policy runs/stage5_payload4R_s0/policy_it13.pkl \
        --stage-walks 56 --speed 0.8 --clips 3 --out videos/go1_payload4R_stage.mp4
"""
import argparse, os, pickle, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ilc_go1lla import walkenv as WE, srb as S        # noqa: E402
import jax                                             # noqa: E402
import mujoco                                          # noqa: E402
from PIL import Image, ImageDraw, ImageFont            # noqa: E402
from zeroshot_cases import CASES, parse_case           # noqa: E402

FFMPEG = '/home/henry/miniconda3/envs/f1t/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
ap = argparse.ArgumentParser()
ap.add_argument('--case', required=True)
ap.add_argument('--policy', required=True, help='stage policy pickle (policy_it<k>.pkl)')
ap.add_argument('--params', default='runs/walker_nom_s0/params.pkl')
ap.add_argument('--stage-walks', type=int, required=True, help='real practice walks the stage used (for the label)')
ap.add_argument('--speed', type=float, default=0.8)
ap.add_argument('--dist', type=float, default=2.5)
ap.add_argument('--clips', type=int, default=3)
ap.add_argument('--eval-walks', type=int, default=20)
ap.add_argument('--seed-label', default='')
ap.add_argument('--title', default='')
ap.add_argument('--out', required=True)
a = ap.parse_args()
W_PANEL, H = 640, 480
TOP = 70


def goals():
    th = np.radians(np.random.default_rng(300).uniform(-15, 15, a.eval_walks))
    return [np.array([a.dist * np.cos(t), a.dist * np.sin(t), a.speed], np.float32) for t in th]


t_sw = float(WE.ref_end_time(np.array([a.dist, 0.0, a.speed], np.float32))) + WE.STAND_DELAY
n = int((t_sw + 0.3) / 0.02)
PERT = parse_case(a.case)
zs = WE.Walker(a.params, stand_at_goal=True, **PERT)
st = WE.Walker(a.params, stand_at_goal=True, **PERT)
nomw = WE.Walker(a.params, stand_at_goal=True)
KIN = S.LegKin(zs.env.mj_model)
pk = pickle.load(open(a.policy, 'rb'))
st.params = (pk['norm'], pk['pol']) + tuple(st.params[2:])
st._roll = jax.jit(st._rollout, static_argnames=('n',))
payload = PERT.get('payload')
weak = PERT.get('weak_leg')
foam = PERT.get('foam')
desc = []
if payload: desc.append(f'{payload[0]:.0f} kg payload, {"right" if payload[1] < 0 else "left"} side')
if weak: desc.append(f'{weak[0]} leg at {weak[1]:.0%} torque')
if foam: desc.append(f'soft foam floor (stiffness time constant {foam[0]:.2f} s)')
desc = ', '.join(desc)

m = zs.env.mj_model
mj_d = [mujoco.MjData(m), mujoco.MjData(m)]
ren = mujoco.Renderer(m, H - TOP, W_PANEL)
fnt, fntb, fnts = ImageFont.truetype(FONT, 18), ImageFont.truetype(FONTB, 22), ImageFont.truetype(FONT, 15)
fbig = ImageFont.truetype(FONTB, 46)


def add_geom(scn, typ, size, pos, rgba, mat=None):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, typ, np.asarray(size, float), np.asarray(pos, float),
                        (np.eye(3) if mat is None else mat).ravel(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def shown_index(out, k, kf):
    """The feet-only collision model lets a fallen trunk sink through the floor: hold the pose 0.4 s after the fall, or
    earlier, as soon as the trunk sinks below 12 cm after the fall (on foam it would otherwise vanish under the floor)."""
    if kf is None:
        return k
    low = np.where(out['p'][kf:, 2] < 0.12)[0]
    return min(k, kf + 20, kf + (int(low[0]) if len(low) else 20))


def frame(out, k, goal, cam, kf):
    d = mj_d[0]
    k = shown_index(out, k, kf)
    d.qpos[:3], d.qpos[3:7], d.qpos[7:] = out['p'][k], out['quat'][k], out['qj'][k]
    mujoco.mj_forward(m, d)
    ren.update_scene(d, camera=cam)
    scn = ren.scene
    u = goal[:2] / np.linalg.norm(goal[:2])
    if foam:                                                   # the foam strip (feet sink below its surface)
        h = np.arctan2(u[1], u[0])
        Rf = np.array([[np.cos(h), -np.sin(h), 0], [np.sin(h), np.cos(h), 0], [0, 0, 1]])
        L = np.linalg.norm(goal[:2])
        add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [L / 2 + 0.4, 0.6, 0.004], [L / 2 * u[0], L / 2 * u[1], 0.002],
                 [0.95, 0.55, 0.65, 0.45], Rf)
    for s in np.linspace(0, np.linalg.norm(goal[:2]), 26):
        add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0], [s * u[0], s * u[1], 0.005], [0.2, 0.4, 1.0, 0.8])
    add_geom(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.15, 0.003, 0], [goal[0], goal[1], 0.003], [0.1, 0.8, 0.2, 0.6])
    pr, _, _ = WE.reference(k * 0.02, goal)
    add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.03, 0, 0], [float(pr[0]), float(pr[1]), 0.03], [0.2, 0.4, 1.0, 1.0])
    if payload:
        R = np.asarray(S.quat_R(out['quat'][k]))
        pos = out['p'][k] + R @ np.array([0.0, payload[1], payload[2] + 0.02])
        add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.05, 0.05, 0.04], pos, [1.0, 0.8, 0.0, 1.0], R)
    return ren.render()


def panel(img, label, sub, out, k, goal, fell_k, t_goal):
    im = Image.new('RGB', (W_PANEL, H), (24, 24, 28))
    im.paste(Image.fromarray(img), (0, TOP))
    dr = ImageDraw.Draw(im)
    dr.text((12, 8), label, font=fntb, fill=(255, 255, 255))
    dr.text((12, 38), sub, font=fnts, fill=(190, 190, 200))
    t = k * 0.02
    k0 = max(0, k - 10)
    v = np.linalg.norm(out['p'][k, :2] - out['p'][k0, :2]) / max((k - k0) * 0.02, 1e-6)
    dist = np.linalg.norm(out['p'][k, :2] - goal[:2])
    dr.text((12, TOP + 8), f't {t:4.1f} s   speed {v:4.2f} m/s   to goal {dist:4.2f} m', font=fnt, fill=(255, 255, 255),
            stroke_width=2, stroke_fill=(0, 0, 0))
    if fell_k is not None and k >= fell_k:
        dr.text((W_PANEL // 2 - 70, TOP + 150), 'FELL', font=fbig, fill=(235, 60, 60), stroke_width=3, stroke_fill=(0, 0, 0))
    elif t_goal is not None and t >= t_goal:
        dr.text((W_PANEL // 2 - 190, TOP + 150), f'AT GOAL {t_goal:.2f} s', font=fbig, fill=(80, 220, 110), stroke_width=3, stroke_fill=(0, 0, 0))
    return im


proc = subprocess.Popen([FFMPEG, '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{2 * W_PANEL}x{H + 40}',
                         '-r', '50', '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', a.out], stdin=subprocess.PIPE)


def emit(im):
    proc.stdin.write(np.asarray(im.convert('RGB'), np.uint8).tobytes())


def banner(text, sub=''):
    im = Image.new('RGB', (2 * W_PANEL, H + 40), (24, 24, 28))
    dr = ImageDraw.Draw(im)
    y = 120
    for line in text.split('\n'):
        dr.text((60, y), line, font=fntb, fill=(255, 255, 255)); y += 36
    y += 10
    for line in sub.split('\n'):
        dr.text((60, y), line, font=fnt, fill=(200, 200, 210)); y += 28
    return im


def judge(out, g):
    kf = S.fall_index_feet(out, KIN)
    end = n if kf is None else kf
    p = out['p'][:end, :2]
    d = np.linalg.norm(p - g[:2], axis=1)
    arr = np.where(d < 0.15)[0]
    u = g[:2] / np.linalg.norm(g[:2])
    cross = np.abs(u[0] * p[:, 1] - u[1] * p[:, 0])
    k_end = arr[0] + 1 if len(arr) else end
    t_goal = None if (kf is not None or not len(arr)) else float(arr[0] * 0.02)
    return kf, t_goal, float(100 * np.sqrt(np.mean(cross[:k_end] ** 2)))


gl = goals()
t_nom = float(np.mean([judge(nomw.rollout(g, n=n), g)[1] for g in gl[:5]]))
title = a.title or f'Go1 walking goal to goal with a {desc} (sim)'
emit_n = lambda im, s: [emit(im) for _ in range(int(50 * s))]
emit_n(banner(title, f'Walk 2.5 m to a goal at {a.speed:.1f} m/s, then stand. Same robot, same goals (held out: never practised).\n'
                     f'Nominal robot (no perturbation): reaches the goal in {t_nom:.2f} s, never falls.\n'
                     f'Left: the walker trained on the nominal Go1 only.\n'
                     f'Right: the same network after the DistILC stage ({a.stage_walks} practice walks on this robot{a.seed_label}).\n'
                     f'Blue dots: straight path and the moving reference point. Green: goal.'), 5)
summary = []
for gi, g in enumerate(gl):
    oz, os_ = zs.rollout(g, n=n), st.rollout(g, n=n)
    (fz, tz, ez), (fs, ts, es) = judge(oz, g), judge(os_, g)
    summary.append((fz, fs, tz, ts, ez, es))
    if gi >= a.clips:
        continue
    h = np.degrees(np.arctan2(g[1], g[0]))
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [g[0] / 2, g[1] / 2, 0.15]
    cam.distance, cam.elevation, cam.azimuth = 3.0, -18, h - 90
    for k in range(n):
        im = Image.new('RGB', (2 * W_PANEL, H + 40), (24, 24, 28))
        im.paste(panel(frame(oz, k, g, cam, fz), 'Zero-shot', 'trained on the nominal Go1', oz, k, g, fz, tz), (0, 0))
        im.paste(panel(frame(os_, k, g, cam, fs), 'After the DistILC stage', f'{a.stage_walks} practice walks on this robot', os_, k, g, fs, ts), (W_PANEL, 0))
        ImageDraw.Draw(im).text((12, H + 8), f'{desc}  |  {a.speed:.1f} m/s  |  held-out walk {gi + 1} of {a.eval_walks}  |  nominal robot: {t_nom:.2f} s',
                                font=fnt, fill=(200, 200, 210))
        emit(im)
    emit_n(im, 1.0)


def agg(fs, ts, es):
    nf = sum(f is not None for f in fs)
    tt = [t for t in ts if t is not None]
    return nf, len(tt), (float(np.mean(tt)) if tt else None), float(np.mean(es))


Z = agg([x[0] for x in summary], [x[2] for x in summary], [x[4] for x in summary])
T = agg([x[1] for x in summary], [x[3] for x in summary], [x[5] for x in summary])
fmt = lambda r: (f'falls {r[0]:2d}/{len(summary)}   reached the goal {r[1]:2d}/{len(summary)}   time to goal '
                 + (f'{r[2]:.2f} s' if r[2] is not None else '  -   ') + f'   off the straight path {r[3]:4.1f} cm (RMS)')
lines = [f'zero-shot:        {fmt(Z)}', f'after the stage:  {fmt(T)}', f'nominal robot:    falls  0   time to goal {t_nom:.2f} s', '',
         'reached = within 15 cm of the goal; a walk that falls never reaches it.']
emit_n(banner(f'All {len(summary)} held-out walks at {a.speed:.1f} m/s', '\n'.join(lines)), 7)
proc.stdin.close(); proc.wait()
print('wrote', a.out)
for l in lines:
    print(' ', l)
