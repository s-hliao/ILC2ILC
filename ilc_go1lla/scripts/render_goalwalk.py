"""(2026-11-24 GOAL walker version of render_walk.py: goalstage.GoalSimWalker, goals in any direction, the reference = the base
network's walk on the NOMINAL robot (blue path + moving point), the stage's 8 evaluation bearings.)
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
from ilc_go1lla import walkenv as WE, srb as S, goalstage as GS, goalwalk as GW   # noqa: E402
import jax                                             # noqa: E402
import mujoco                                          # noqa: E402
from PIL import Image, ImageDraw, ImageFont            # noqa: E402
from zeroshot_cases import CASES                       # noqa: E402

FFMPEG = '/home/henry/miniconda3/envs/f1t/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
ap = argparse.ArgumentParser()
ap.add_argument('--case', required=True)
ap.add_argument('--policy', required=True, help='stage policy pickle (policy_it<k>.pkl)')
ap.add_argument('--params', default='runs/goalwalk/L_vel2_s0/params.pkl')
ap.add_argument('--variant', default='vel2')
ap.add_argument('--rows', default='', help='label: the stage\'s row type')
ap.add_argument('--stage-walks', type=int, required=True, help='real practice walks the stage used (for the label)')
ap.add_argument('--speed', type=float, default=0.8)
ap.add_argument('--dist', type=float, default=1.5)
ap.add_argument('--clips', type=int, default=4)
ap.add_argument('--clip-bearings', type=float, nargs='+', default=[0, 90, 180, -135])
ap.add_argument('--eval-walks', type=int, default=8)
ap.add_argument('--title', default='')
ap.add_argument('--out', required=True)
a = ap.parse_args()
W_PANEL, H = 640, 480
TOP = 70


BEAR = [0, 45, -45, 90, -90, 135, -135, 180]


def goals():
    return [(b, np.array([a.dist * np.cos(np.radians(b)), a.dist * np.sin(np.radians(b))], np.float32)) for b in BEAR[:a.eval_walks]]


zs = GS.GoalSimWalker(a.params, a.variant, **CASES[a.case])
st = GS.GoalSimWalker(a.params, a.variant, **CASES[a.case])
nomw = GS.GoalSimWalker(a.params, a.variant)
pk = pickle.load(open(a.policy, 'rb'))
st.params = (pk['norm'], pk['pol']) + tuple(st.params[2:])
st._roll = jax.jit(st._rollout, static_argnames=('n',))
payload = CASES[a.case].get('payload')
weak = CASES[a.case].get('weak_leg')
foam = CASES[a.case].get('foam')
desc = []
if payload: desc.append(f'{payload[0]:.0f} kg payload, {"right" if payload[1] < 0 else "left"} side')
if weak: desc.append(f'{weak[0]} leg at {weak[1]:.0%} torque')
if foam: desc.append('soft foam floor')
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


def shown_index(out, k):
    """The feet-only collision model lets a fallen trunk sink through the floor: hold the pose once the trunk is down."""
    low = np.where(out['p'][:, 2] < 0.12)[0]
    return min(k, low[0]) if len(low) else k


def frame(out, k, goal, cam):
    d = mj_d[0]
    k = shown_index(out, k)
    d.qpos[:3], d.qpos[3:7], d.qpos[7:] = out['p'][k], out['quat'][k], out['qj'][k]
    mujoco.mj_forward(m, d)
    ren.update_scene(d, camera=cam)
    scn = ren.scene
    P = REF['p']
    for j in range(0, len(P), 6):
        add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0], [P[j, 0], P[j, 1], 0.005], [0.2, 0.4, 1.0, 0.8])
    add_geom(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [GW.ARRIVE, 0.003, 0], [goal[0], goal[1], 0.003], [0.1, 0.8, 0.2, 0.6])
    pr = P[min(k, len(P) - 1)]
    add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.03, 0, 0], [float(pr[0]), float(pr[1]), 0.03], [0.2, 0.4, 1.0, 1.0])
    for j in range(0, k, 5):                                     # this robot's own trail
        add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.008, 0, 0], [out['p'][j, 0], out['p'][j, 1], 0.004], [1.0, 0.5, 0.1, 0.9])
    if payload:
        R = np.asarray(S.quat_R(out['quat'][k]))
        pos = out['p'][k] + R @ np.array([0.0, payload[1], payload[2] + 0.02])
        add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.05, 0.05, 0.04], pos, [1.0, 0.8, 0.0, 1.0], R)
    return ren.render()


def panel(img, label, sub, out, k, goal, fell_k):
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
    elif dist < GW.ARRIVE + 0.03:
        dr.text((W_PANEL // 2 - 120, TOP + 150), 'AT GOAL', font=fbig, fill=(80, 220, 110), stroke_width=3, stroke_fill=(0, 0, 0))
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


title = a.title or f'Go1 goal walker with a {desc} (sim)'
emit_n = lambda im, s: [emit(im) for _ in range(int(50 * s))]
emit_n(banner(title, f'The network walks to a goal {a.dist} m away in any direction and stands there (no command law).\n'
                     f'Left: the walker trained on the nominal Go1 only.  Right: the same network after the DistILC stage\n'
                     f'({a.stage_walks} practice walks on this robot, random goals{", rows: " + a.rows if a.rows else ""}). Same robot, same evaluation goals.\n'
                     f'Blue: the nominal robot\'s walk to the goal (the reference) and its moving point. Orange: this robot\'s trail. Green: goal.'), 5)
summary = []
shown = 0
for gi, (bdeg, g) in enumerate(goals()):
    n = GS.walk_steps(g)
    REF = GS.reference(nomw.rollout(g, n))
    oz, os_ = zs.rollout(g, n=n), st.rollout(g, n=n)
    fz, fs = S.fall_index(oz), S.fall_index(os_)
    def trk(o, f):
        e = o['p'][:f or n, :2] - REF['p'][:f or n]
        return float(100 * np.sqrt(np.mean(np.sum(e ** 2, 1))))
    summary.append((fz, fs, trk(oz, fz), trk(os_, fs), np.linalg.norm(oz['p'][-1, :2] - g[:2]), np.linalg.norm(os_['p'][-1, :2] - g[:2]), bdeg))
    if bdeg not in a.clip_bearings or shown >= a.clips:
        continue
    shown += 1
    h = np.degrees(np.arctan2(g[1], g[0]))
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [g[0] / 2, g[1] / 2, 0.15]
    cam.distance, cam.elevation, cam.azimuth = 3.0, -18, h - 90
    for k in range(n):
        im = Image.new('RGB', (2 * W_PANEL, H + 40), (24, 24, 28))
        im.paste(panel(frame(oz, k, g, cam), 'Zero-shot', 'trained on the nominal Go1', oz, k, g, fz), (0, 0))
        im.paste(panel(frame(os_, k, g, cam), 'After the DistILC stage', f'{a.stage_walks} practice walks on this robot', os_, k, g, fs), (W_PANEL, 0))
        ImageDraw.Draw(im).text((12, H + 8), f'{desc}  |  goal {a.dist} m at {bdeg:+.0f} deg from the start heading  |  evaluation walk {gi + 1} of {a.eval_walks}', font=fnt, fill=(200, 200, 210))
        emit(im)
    emit_n(im, 1.0)
lines = []
for i, (fz, fs, tz, ts, dz, ds, bdeg) in enumerate(summary):
    z = f'FELL at {fz * 0.02:.1f} s' if fz is not None else f'off the nominal walk {tz:4.1f} cm, ends {100 * dz:3.0f} cm'
    s = f'FELL at {fs * 0.02:.1f} s' if fs is not None else f'off the nominal walk {ts:4.1f} cm, ends {100 * ds:3.0f} cm'
    lines.append(f'{bdeg:+4.0f} deg:  zero-shot {z:44s}  stage {s}')
nz, ns = sum(f is not None for f, *_ in summary), sum(f is not None for _, f, *_ in summary)
emit_n(banner(f'All {len(summary)} evaluation walks: zero-shot falls {nz}/{len(summary)}, after the stage {ns}/{len(summary)}',
              '\n'.join(lines)), 6)
proc.stdin.close(); proc.wait()
print('wrote', a.out, 'falls zero-shot', nz, 'stage', ns)
for l in lines:
    print(' ', l)
