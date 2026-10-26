#!/usr/bin/env python3
"""
animate_pogo.py: matplotlib 2-D animations of the pogo sticks (side-by-side panels, one per controller, all on the
same robot and goal): the ground, the body, the leg drawn as a spring (pinned foot in stance, the commanded angle in
flight), the body's trail and the apex target line h*. GIF (pillow) into src/paper/pogo/videos/.
    python animate_pogo.py --robot real_offset --goal 0.8 0.62 \
        --panel "Raibert=expert" --panel "ours zero-shot=runs/nom_s0/policy_final.npz" \
        --panel "ours +24 trials=runs/hw/nom_s0_hw24/real_offset_policy.npz" --out offset_ours.gif
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib                                  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                    # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter   # noqa: E402
import numpy as np                                 # noqa: E402

import common as C                                 # noqa: E402
import policies                                    # noqa: E402
import real_pogo as rp                             # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paper', 'pogo', 'videos')


def spring_xy(p0, p1, n=10, w=0.025):
    """A zigzag spring from p0 (body) to p1 (foot)."""
    d = np.asarray(p1) - np.asarray(p0)
    L = np.linalg.norm(d) + 1e-9
    u, v = d / L, np.array([-d[1], d[0]]) / L
    s = np.linspace(0, 1, 2 * n + 1)
    off = np.zeros_like(s)
    off[1:-1:2], off[2:-1:2] = w, -w
    core = (s > 0.15) & (s < 0.85)
    pts = np.asarray(p0)[None] + s[:, None] * d[None] + (off * core)[:, None] * v[None]
    return pts[:, 0], pts[:, 1]


def fly(policy, robot, goal, seconds, pert=0.0, seed=0):
    g = np.asarray(goal, float)[None]
    x0 = rp.start_states(g, pert, np.random.default_rng(seed))
    pp = rp.robot_params(robot) if robot != 'nominal' else dict(C.NOMINAL)
    res = rp.simulate(policies.as_policy(policy), g, x0, pp, int(seconds / C.DT))
    phi = res['x'][0, :, 6] + pp['phi_off']      # the leg's true angle (the state holds the encoder's)
    m = C.hop_metrics(res['x'][0], g[0], res['failed_end'][0])
    return res['x'][0], phi, m, pp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--robot', default='real_nom')
    ap.add_argument('--goal', type=float, nargs=2, default=[0.8, 0.62])
    ap.add_argument('--panel', action='append', required=True, help='"label=policy" (policy: expert or an npz)')
    ap.add_argument('--seconds', type=float, default=6.0)
    ap.add_argument('--pert', type=float, default=0.0)
    ap.add_argument('--fps', type=int, default=25)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    panels = [p.split('=', 1) for p in a.panel]
    runs = [fly(pol, a.robot, a.goal, a.seconds, a.pert) for _, pol in panels]
    k = len(panels)
    fig, axs = plt.subplots(k, 1, figsize=(9, 2.3 * k), squeeze=False)
    step = max(1, int(round(1.0 / (a.fps * C.DT))))
    frames = range(0, len(runs[0][0]), step)
    arts = []
    for ax, (label, _), (xs, phi, m, pp) in zip(axs[:, 0], panels, runs):
        ax.set_aspect('equal')
        ax.set_ylim(-0.05, 0.9)
        ax.axhline(0, color='0.3', lw=1.5)
        ax.axhline(a.goal[1], color='tab:green', ls='--', lw=0.8)
        ok = 'ok' if m['success'] else ('FELL' if m['failed'] else 'miss')
        ax.set_title(f'{label}: apex rms {100 * (m["rms_apex"] or np.nan):.1f} cm, speed err '
                     f'{(m["v_err"] if m["v_err"] is not None else np.nan):+.2f} m/s ({ok})', fontsize=9, loc='left')
        ax.set_yticks([0, 0.5])
        trail, = ax.plot([], [], color='tab:blue', lw=0.6, alpha=0.5)
        leg, = ax.plot([], [], color='0.2', lw=1.2)
        body, = ax.plot([], [], 'o', color='tab:red', ms=9)
        foot, = ax.plot([], [], 's', color='0.2', ms=4)
        txt = ax.text(0.99, 0.9, '', transform=ax.transAxes, ha='right', fontsize=8)
        arts.append((ax, xs, phi, pp, trail, leg, body, foot, txt))

    def draw(i):
        out = []
        for ax, xs, phi, pp, trail, leg, body, foot, txt in arts:
            j = min(i, len(xs) - 1)
            X, Z, VX, VZ, XF, ST = xs[j]
            fp = (XF, 0.0) if ST > 0.5 else (X + pp['l0'] * np.sin(phi[j]), Z - pp['l0'] * np.cos(phi[j]))
            leg.set_data(*spring_xy((X, Z), fp))
            body.set_data([X], [Z])
            foot.set_data([fp[0]], [fp[1]])
            trail.set_data(xs[max(0, j - 150):j + 1, 0], xs[max(0, j - 150):j + 1, 1])
            ax.set_xlim(X - 1.2, X + 1.2)
            txt.set_text(f't {j * C.DT:4.1f} s   vx {VX:4.2f} (goal {a.goal[0]:.2f})')
            out += [leg, body, foot, trail, txt]
        return out
    fig.tight_layout()
    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = a.out if os.path.isabs(a.out) else os.path.join(OUT_DIR, a.out)
    anim.save(path, writer=PillowWriter(fps=a.fps))
    print(path)


if __name__ == '__main__':
    main()
