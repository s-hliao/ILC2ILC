#!/usr/bin/env python3
"""make_big_tracks.py (2026-11-07; user: plastic must run >= 2 m/s, "can make it wider if necessary"; room "y is about 5 m, x is
about 3.5"): two tracks sized to the plastic car's MEASURED steering limit (11-05 / 11-07 flights, scratchpad yaw_auth.py:
curvature at full lock 0.72 1/m at 1.75-2.0 m/s, 0.60 at 2.0-2.25 -> turns of radius >= ~1.65 m for ~2 m/s corners).
Box: 3.5 x 5 m centred on the old room box's centre (x -0.75, y -0.30) -> x -2.50..1.00, y -2.80..2.20, footprint MARGIN inside it.
  oval  a stadium: two semicircles of radius R joined by straights along y            (practice)
  bean  two lobes of radius R at y = +-d joined by reverse (concave) arcs of radius R2 (held out: the curvature changes sign)
Each written counter-clockwise (NAME) and clockwise (NAME + 'CW'), as the track files' format (x, y, speed, theta) in
../lifted_linear_tire_20261009/tracks_tight/.   python make_big_tracks.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(HERE, '..', 'lifted_linear_tire_20261009', 'tracks_tight')
CX, CY, BW, BH, MARGIN = -0.75, -0.30, 3.5, 5.0, 0.1


def arc(c, r, a0, a1, ds=0.01):
    n = max(2, int(abs(a1 - a0) * r / ds))
    t = np.linspace(a0, a1, n, endpoint=False)
    return np.c_[c[0] + r * np.cos(t), c[1] + r * np.sin(t)]


def line(p, q, ds=0.01):
    n = max(2, int(np.hypot(*(np.subtract(q, p))) / ds))
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    return np.asarray(p) + t * (np.asarray(q) - np.asarray(p))


def oval(R, Ls):
    """Counter-clockwise stadium, straights along y of length Ls, centred at 0."""
    h = Ls / 2
    return np.concatenate([line((R, -h), (R, h)), arc((0, h), R, 0, np.pi), line((-R, h), (-R, -h)), arc((0, -h), R, np.pi, 2 * np.pi)])


def bean(R, d, R2):
    """Counter-clockwise: lobes centred (0, +-d), radius R; concave arcs centred (+-xc, 0), radius R2, tangent to both lobes."""
    xc = np.sqrt((R + R2) ** 2 - d ** 2)
    g = np.arctan2(d, xc)                                   # angle of the lobe-centre -> arc-centre lines from the x axis
    top = arc((0, d), R, -g, np.pi + g)                     # top lobe, over the top (CCW)
    west = arc((-xc, 0), R2, g, -g)                         # west concave arc (CW around its centre)
    bot = arc((0, -d), R, np.pi - g, 2 * np.pi + g)         # bottom lobe, under the bottom (CCW)
    east = arc((xc, 0), R2, np.pi + g, np.pi - g)           # east concave arc (CW)
    return np.concatenate([top, west, bot, east]), xc - R2


def save(name, P, speed=2.5):
    P = P + np.array([CX, CY]) - 0.5 * (P.min(0) + P.max(0))
    w, h = np.ptp(P[:, 0]), np.ptp(P[:, 1])
    assert w <= BW - 2 * MARGIN + 1e-6 and h <= BH - 2 * MARGIN + 1e-6, (name, w, h)
    for nm, Q in ((name, P), (name + 'CW', P[::-1])):
        x, y = Q[:, 0], Q[:, 1]
        th = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
        np.savez(os.path.join(DST, f'{nm}.npz'), x=x, y=y, speed=np.array(speed), theta=th)
        print(f'{nm}: {w:.2f} x {h:.2f} m, x {x.min():.2f}..{x.max():.2f} y {y.min():.2f}..{y.max():.2f}, '
              f'length {np.sum(np.hypot(np.diff(x), np.diff(y))):.2f} m')


def main():
    R = (BW - 2 * MARGIN) / 2                               # 1.65 m: the widest turn the box allows
    save('mocap_oval165', oval(R, BH - 2 * MARGIN - 2 * R))
    d = (BH - 2 * MARGIN - 2 * R) / 2
    P, waist = bean(R, d, R)
    print(f'bean: lobes R {R:.2f} at y +-{d:.2f}, concave arcs R {R:.2f}, waist half-width {waist:.2f} m (lobes {R:.2f})')
    save('mocap_bean165', P)


if __name__ == '__main__':
    main()


def peanuts():
    """(2026-11-07 ~23:55, user: the held-out "needs to be a much deeper waist, but yes, that is fine" -- slower corners accepted):
    peanuts filling the 4.8 m length, lobes R at y = +-(2.4 - R), concave arcs R2 -> waist half-width sqrt((R+R2)^2 - d^2) - R2."""
    for R, R2 in ((1.25, 1.0), (1.2, 0.8), (1.15, 0.7)):
        d = (BH - 2 * MARGIN) / 2 - R
        P, waist = bean(R, d, R2)
        nm = f'mocap_pnut{int(round(R * 100))}w{int(round(R2 * 100))}'
        print(f'{nm}: lobes R {R} at y +-{d:.2f}, concave R2 {R2}, waist half-width {waist:.2f} m = {waist / R:.0%} of the lobes')
        save(nm, P)


if __name__ == '__main__' and os.environ.get('PEANUTS') == '1':
    peanuts()
