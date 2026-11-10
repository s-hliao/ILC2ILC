#!/usr/bin/env python3
"""make_heldout_track.py: the HELD-OUT track of the car's perturbation suite (the quadruped's held-out goals): llampc's
mocap_square (a rounded square, 3.1 x 3.1 m, tightest radius 0.82 m) scaled by --scale (0.9: 2.8 x 2.8 m, radius
0.74 m) about its centre, so it fits the room (inside square2fast's tight footprint) -- a trajectory shape no network
was ever trained or adapted on. -> ../lifted_linear_tire_20261009/tracks_tight/mocap_squareH.npz (use with F1T_TRACKS).
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'lifted_linear_tire_20261009', 'tracks', 'mocap_square.npz')
DST = os.path.join(HERE, '..', 'lifted_linear_tire_20261009', 'tracks_tight', 'mocap_squareH.npz')
ap = argparse.ArgumentParser()
ap.add_argument('--scale', type=float, default=0.9)
a = ap.parse_args()
h = dict(np.load(SRC, allow_pickle=True))
x, y = np.asarray(h['x'], float), np.asarray(h['y'], float)
cx, cy = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
h['x'], h['y'] = cx + (x - cx) * a.scale, cy + (y - cy) * a.scale
h['theta'] = np.unwrap(np.arctan2(np.gradient(h['y']), np.gradient(h['x'])))
np.savez(DST, **h)
os.environ['F1T_TRACKS'] = os.path.dirname(DST)
import tracks                                     # noqa: E402
t = tracks.Track('mocap_squareH')
print(f'mocap_squareH: {np.ptp(t.xy[:, 0]):.2f} x {np.ptp(t.xy[:, 1]):.2f} m, length {t.length:.2f} m, tightest radius '
      f'{1 / np.abs(t.kappa_smooth()).max():.2f} m, design speed {t.design_speed:.2f} m/s -> {DST}')
