#!/usr/bin/env python3
"""make_orca_track.py TRACK_JSON|CENTER_TXT [--scale 4.3] [--name NAME]
(2026-11-05, user: "ETHZMobil Track": also BayesRace's ETHZMobil centreline, github.com/jainachin/bayesrace
bayes_race/tracks/src/ethzMobil_center.txt, 3.0 x 1.8 m, 12.8 m, radius 0.30 m; x4.3 -> 55 m, tightest radius 1.28 m, within the
F1TENTH's 0.93 m minimum turn: --name ethzmobil43.)
Originally (2026-11-05; user: "run training on harder track like ethzurich orca"): the
ETH Zurich ORCA track (Liniger et al., the 1:43 RC-car racing track; centreline X, Y of
github.com/alexliniger/MPCC C++/Params/track.json) scaled by --scale about its centre -> tracks_tight/orca43.npz.
Default 4.3 = 43 / 10, the ORCA cars' scale over the F1TENTH's: 17.8 m -> 77 m per lap, tightest radius 0.185 -> 0.80 m
(the bank's tightest so far: mocap_squareH 0.74 m), 1610 deg of turning per lap (the held-out peanut: 414). SIMULATION ONLY:
at this size it does not fit the mocap room. Design speed 2.5 m/s (as the peanut).
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(HERE, '..', 'lifted_linear_tire_20261009', 'tracks_tight')
ap = argparse.ArgumentParser()
ap.add_argument('track_json')
ap.add_argument('--scale', type=float, default=4.3)
ap.add_argument('--name', default='orca43')
a = ap.parse_args()
if a.track_json.endswith('.txt'):                # BayesRace's centreline files (e.g. ethzMobil_center.txt): two rows, x and y
    x, y = np.loadtxt(a.track_json, delimiter=',')
else:
    t = json.load(open(a.track_json))
    x, y = np.asarray(t['X'], float), np.asarray(t['Y'], float)
assert x.shape == y.shape and x.ndim == 1 and np.all(np.isfinite(x)) and np.all(np.isfinite(y)), 'bad centreline'
cx, cy = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
x, y = (x - cx) * a.scale, (y - cy) * a.scale
theta = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
out = os.path.join(DST, f'{a.name}.npz')
np.savez(out, x=x, y=y, theta=theta, speed=np.float64(2.5))
os.environ['F1T_TRACKS'] = DST
import tracks                                     # noqa: E402
tr = tracks.Track(a.name)
print(f'{a.name}: {np.ptp(tr.xy[:, 0]):.2f} x {np.ptp(tr.xy[:, 1]):.2f} m, length {tr.length:.2f} m, tightest radius '
      f'{1 / np.abs(tr.kappa_smooth()).max():.2f} m, turns {tr.turns:+.2f}, design speed {tr.design_speed:.2f} m/s -> {out}')
