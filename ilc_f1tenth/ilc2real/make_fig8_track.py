#!/usr/bin/env python3
"""make_fig8_track.py (2026-10-31; user: figfast "with more forgiving geometry ... still needs to be challenging, just not
infeasible", inside the current figfast area): a figure-eight of two circular lobes of radius R joined by straights that
cross at a half-angle alpha from the lobes' axis, rotated and centred inside the room box (figfast's and square2fast's
footprint, x -2.19..0.69, y -2.25..1.65). mocap_figfast's tightest radius is 0.58 m, below the car's kinematic steering
limit (wheelbase / tan(s_max) = 0.93 m), so 37 % of it can only be taken by rotating the car.

    python make_fig8_track.py NAME --R 0.95 --alpha 60 --rot 26 [--speed 2.5]
-> ../lifted_linear_tire_20261009/tracks_tight/NAME.npz (x, y, speed, theta: the track files' format)
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(HERE, '..', 'lifted_linear_tire_20261009', 'tracks_tight')
BOX = ((-2.19, 0.69), (-2.25, 1.65))


def fig8(R, alpha, n=600):
    """Centre line from the crossing: straight, left lobe (turn pi + 2 alpha), straight through the crossing, right lobe."""
    c = R / np.sin(alpha)
    l = c * np.cos(alpha)
    turn = np.pi + 2 * alpha
    segs = [(l, 0.0), (R * turn, 1 / R), (2 * l, 0.0), (R * turn, -1 / R), (l, 0.0)]
    L = sum(s for s, _ in segs)
    ds = L / n
    h, x, y, P = np.pi / 2 - alpha, 0.0, 0.0, []
    for s, k in segs:
        for _ in range(int(round(s / ds))):
            P.append((x, y))
            x += ds * np.cos(h + ds * k / 2)
            y += ds * np.sin(h + ds * k / 2)
            h += ds * k
    return np.array(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--R', type=float, required=True)
    ap.add_argument('--alpha', type=float, required=True, help='crossing half-angle from the lobe axis (deg)')
    ap.add_argument('--rot', type=float, default=0.0, help='rotation in the room (deg)')
    ap.add_argument('--speed', type=float, default=2.5, help='design speed (m/s), as mocap_figfast')
    ap.add_argument('--any-size', action='store_true', help='(user: a larger floor area for the figure-eight) do not require the '
                    'current room box; the track is still centred on it')
    a = ap.parse_args()
    P = fig8(a.R, np.radians(a.alpha))
    c, s = np.cos(np.radians(a.rot)), np.sin(np.radians(a.rot))
    P = P @ np.array([[c, s], [-s, c]])
    P += np.array([np.mean(BOX[0]), np.mean(BOX[1])]) - 0.5 * (P.min(0) + P.max(0))
    w, h = np.ptp(P[:, 0]), np.ptp(P[:, 1])
    assert a.any_size or (w <= np.ptp(BOX[0]) + 1e-3 and h <= np.ptp(BOX[1]) + 1e-3), (w, h)
    x, y = P[:, 0], P[:, 1]
    th = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    np.savez(os.path.join(DST, f'{a.name}.npz'), x=x, y=y, speed=np.array(a.speed), theta=th)
    print(f'{a.name}: {w:.2f} x {h:.2f} m, x {x.min():.2f}..{x.max():.2f} y {y.min():.2f}..{y.max():.2f}, '
          f'length {np.sum(np.hypot(np.diff(x), np.diff(y))):.2f} m, radius {a.R} m')


if __name__ == '__main__':
    main()
