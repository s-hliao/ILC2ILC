#!/usr/bin/env python3
"""make_extra_tracks.py (2026-10-27): extra TRAINING tracks for the car's free-goal study -- on the two training tracks
alone our networks overfit them (held-out mocap_squareH: ours + stage 72-75 % vs PPO+DR / the plan LQR ~97 %,
log/dilc/car_free/eval_free). Inside the tight room's footprint (square2fast's), none of them a rounded square like the
held-out track:
  mocap_square2fastM  square2fast mirrored in x (the other turning direction)
  mocap_figfastM      figfast mirrored in x
  mocap_stadT         a stadium, radius 0.55 m, straights 1.2 m, counter-clockwise, design speed 2.9 m/s
-> ../lifted_linear_tire_20261009/tracks_tight/<name>.npz (use with F1T_TRACKS); mocap_squareH stays held out.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(HERE, '..', 'lifted_linear_tire_20261009', 'tracks_tight')


def theta(x, y):
    return np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))


def mirror(name):
    h = dict(np.load(os.path.join(DST, f'{name}.npz'), allow_pickle=True))
    x, y = np.asarray(h['x'], float), np.asarray(h['y'], float)
    cx = 0.5 * (x.min() + x.max())
    h['x'] = 2 * cx - x
    h['theta'] = theta(h['x'], y)
    np.savez(os.path.join(DST, f'{name}M.npz'), **h)


def stadium(R=0.55, S=1.2, c=(-0.75, -0.25), v=2.9, n=500):
    """Counter-clockwise: up the right straight, the top half circle, down the left, the bottom half circle."""
    per = 2 * S + 2 * np.pi * R
    s = np.linspace(0, per, n, endpoint=False)
    x, y = np.zeros(n), np.zeros(n)
    for i, si in enumerate(s):
        if si < S:                                           # right straight, going up
            x[i], y[i] = R, -S / 2 + si
        elif si < S + np.pi * R:                             # top arc
            a = (si - S) / R
            x[i], y[i] = R * np.cos(a), S / 2 + R * np.sin(a)
        elif si < 2 * S + np.pi * R:                         # left straight, going down
            x[i], y[i] = -R, S / 2 - (si - S - np.pi * R)
        else:                                                # bottom arc
            a = np.pi + (si - 2 * S - np.pi * R) / R
            x[i], y[i] = R * np.cos(a), -S / 2 + R * np.sin(a)
    x, y = x + c[0], y + c[1]
    np.savez(os.path.join(DST, 'mocap_stadT.npz'), x=x, y=y, speed=np.full(n, v), theta=theta(x, y))


mirror('mocap_square2fast')
mirror('mocap_figfast')
stadium()
os.environ['F1T_TRACKS'] = DST
import tracks                                                # noqa: E402
for nm in ('mocap_square2fastM', 'mocap_figfastM', 'mocap_stadT', 'mocap_squareH'):
    t = tracks.Track(nm)
    print(f'{nm}: {np.ptp(t.xy[:, 0]):.2f} x {np.ptp(t.xy[:, 1]):.2f} m, length {t.length:.2f} m, tightest radius '
          f'{1 / np.abs(t.kappa_smooth()).max():.2f} m, turns {t.turns:+.0f}, design speed {t.design_speed:.2f} m/s')
