"""Side-by-side top-down videos of the Go1 ball task: each column = one (label, policy, robot) on the SAME held-out goal
flight (test seed). Top: the mocap room (trunk as an oriented box, goals, path, current goal). Bottom: the box on the
dog's back (ball, its trail, the 60 % safe margin dashed; walls flash red on contact). Real time (20 fps = 50 ms).

    python scripts/render.py --out videos/compare.mp4 --seed 100 --goals 8 \
        "zero-shot (perturbed)=runs/stage_safe_s0/policy_g0000.npz@perturbed" \
        "after stage (perturbed)=runs/stage_safe_s0/policy_g0096.npz@perturbed"
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax.numpy as jnp                              # noqa: E402
import matplotlib                                    # noqa: E402
matplotlib.use('Agg')
import imageio_ffmpeg                                # noqa: E402
matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.animation import FFMpegWriter        # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle   # noqa: E402

from ilc_dogball import model as m, sim, stage       # noqa: E402
from ilc_dogball.ppo import actor_fn                 # noqa: E402
from stage_run import load_actor, SEG_T              # noqa: E402

ROBOTS = {'nominal': m.NOMINAL, 'perturbed': m.PERTURBED, 'filled': m.FILLED, 'hollow': m.HOLLOW}
STOP = False
TRUNK = (0.38, 0.19)                                 # Go1 trunk footprint (m)


def fly(spec, x0, goals):
    label, rest = spec.split('=', 1)
    path, robot = rest.rsplit('@', 1)
    T = goals.shape[0] * SEG_T // 2
    out = sim.rollout(load_actor(path), jnp.asarray(x0)[None], jnp.asarray(goals)[None], sim.params(ROBOTS[robot], 1),
                      actor_fn, jnp.zeros((T, 1, m.NA)), reset_ball=True, stop_mode=STOP)
    f = {k: np.asarray(v)[:, 0] for k, v in out.items()}
    done = np.where(f['reached'] | f['timeout'])[0]
    end = int(done[min(len(done), goals.shape[0]) - 1]) + 1 if len(done) else T
    f = {k: v[:end + 20] for k, v in f.items()}      # 1 s after the last goal
    return label, f


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('specs', nargs='+', help='"label=policy.npz@nominal|perturbed"')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=100)
    ap.add_argument('--goals', type=int, default=8)
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--stop', action='store_true', help='precise-stop task: goals are 10 cm floor markers, stops drawn with their error')
    a = ap.parse_args()
    global STOP
    STOP = a.stop
    x0, goals = sim.sample_goals(np.random.default_rng(a.seed), 1, a.goals)
    runs = [fly(s, x0[0], goals[0]) for s in a.specs]
    n = len(runs)
    T = max(len(f['x']) for _, f in runs)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 8.2), gridspec_kw=dict(height_ratios=[1.35, 1]), squeeze=False)
    fig.patch.set_facecolor('white')
    arts = []
    hx, hy = m.BOX_IN[0] / 2, m.BOX_IN[1] / 2
    for c, (label, f) in enumerate(runs):
        ax, bx = axes[0, c], axes[1, c]
        ax.set_xlim(*sim.ROOM[0]); ax.set_ylim(*sim.ROOM[1]); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=11)
        stops = [i for i in range(len(f['x'])) if f['reached'][i]]
        for k, g in enumerate(goals[0]):
            ax.add_patch(Circle(g, 0.05 if a.stop else sim.REACH, fill=a.stop, lw=1, color='0.6'))
            ax.text(g[0], g[1], str(k + 1), ha='center', va='center', fontsize=8, color='0.4')
        cur = Circle(goals[0, 0], 0.08 if a.stop else sim.REACH, fill=True, alpha=0.35, color='tab:green')
        ax.add_patch(cur)
        path, = ax.plot([], [], color='tab:blue', lw=1.2)
        trunk = Polygon(np.zeros((4, 2)), closed=True, color='tab:blue', alpha=0.85)
        ax.add_patch(trunk)
        txt = ax.text(0.02, 0.97, '', transform=ax.transAxes, va='top', fontsize=9, family='monospace')
        bx.set_xlim(-hx - 0.03, hx + 0.03); bx.set_ylim(-hy - 0.03, hy + 0.03); bx.set_aspect('equal')
        bx.set_xticks([]); bx.set_yticks([]); bx.set_title('ball in the box (dog frame, front ->)', fontsize=9)
        wall = Rectangle((-hx, -hy), 2 * hx, 2 * hy, fill=False, lw=3, color='0.3')
        bx.add_patch(wall)
        mg = stage.MARGIN + m.BALL_R
        bx.add_patch(Rectangle((-mg[0], -mg[1]), 2 * mg[0], 2 * mg[1], fill=False, lw=1, ls='--', color='tab:orange'))
        trail, = bx.plot([], [], color='tab:red', lw=0.8, alpha=0.5)
        ball = Circle((0, 0), m.BALL_R, color='tab:red', alpha=0.8)
        bx.add_patch(ball)
        stop_dots, = ax.plot([], [], 'o', color='tab:red', ms=5, zorder=5)
        arts.append((f, cur, path, trunk, txt, wall, trail, ball, stop_dots, stops))
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    writer = FFMpegWriter(fps=a.fps, bitrate=2400)
    with writer.saving(fig, a.out, dpi=100):
        for t in range(T):
            for f, cur, path, trunk, txt, wall, trail, ball, stop_dots, stops in arts:
                k = min(t, len(f['x']) - 1)
                x = f['x'][k]
                gi = int(min(f['goal_idx'][k], goals.shape[1] - 1))
                cur.center = goals[0, gi]
                path.set_data(f['x'][:k + 1, 0], f['x'][:k + 1, 1])
                c, s = np.cos(x[2]), np.sin(x[2])
                corners = np.array([[TRUNK[0], TRUNK[1]], [TRUNK[0], -TRUNK[1]], [-TRUNK[0], -TRUNK[1]], [-TRUNK[0], TRUNK[1]]]) / 2
                trunk.set_xy(x[:2] + corners @ np.array([[c, s], [-s, c]]))
                b = f['x'][max(0, k - 30):k + 1, 6:8]
                trail.set_data(b[:, 0], b[:, 1])
                ball.center = x[6:8]
                touching = np.any(np.abs(x[6:8]) > m.TRAVEL - 0.005)
                wall.set_color('tab:red' if touching else '0.3')
                bb = f['x'][:k + 1, 6:8]
                contact = np.any(np.abs(bb) > m.TRAVEL, -1)
                hits = int(np.sum(np.diff(contact.astype(int)) == 1))
                outside = float(np.mean(np.any(np.abs(bb) > stage.MARGIN, -1)))
                done = int(np.sum(f['reached'][:k + 1] | f['timeout'][:k + 1]))
                V = float(np.hypot(x[3], x[4]))
                done_stops = [i for i in stops if i <= k]
                if done_stops:
                    stop_dots.set_data(f['x'][done_stops, 0], f['x'][done_stops, 1])
                se = f'  mean stop error {100 * np.mean(f["dist"][done_stops]):4.1f} cm' if (a.stop and done_stops) else ''
                txt.set_text(f't {k * m.DT:5.1f} s  goals {min(done, goals.shape[1])}/{goals.shape[1]}{se}\n'
                             f'speed {V:4.2f} m/s  wall hits {hits}\nball outside margin {100 * outside:4.1f} %')
            writer.grab_frame()
    print('wrote', a.out, T, 'frames')


if __name__ == '__main__':
    main()
