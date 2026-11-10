#!/usr/bin/env python3
"""make_tight_tracks.py [--each 0.5] [--each-fig 0.25]: the two tracks shortened in their long (y) direction at each end
(square2fast by --each, figfast by --each-fig),
for a room with small tolerances (user, 2026-10-09): square2fast by cutting its two long straights (corners
unchanged), figfast (no straights) by a uniform y-scale, the gentlest option on curvature. Writes ../lifted_linear_tire_20261009/tracks_tight/
<name>.npz with the original fields (x, y and theta recomputed; speed, vs, mus kept). Use: F1T_TRACKS=<that dir>.
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'lifted_linear_tire_20261009', 'tracks')
DST = os.path.join(HERE, '..', 'lifted_linear_tire_20261009', 'tracks_tight')
ap = argparse.ArgumentParser()
ap.add_argument('--each', type=float, default=0.5, help='m taken off each end of square2fast\'s long direction')
ap.add_argument('--each-fig', type=float, default=0.25, help='m taken off each end of figfast\'s: 0.5 makes its lobes '
                'tighter (0.47 m) than the car steers through on mu 0.2 -- every controller, even the plan LQR, hit the '
                '0.3 m envelope; 0.25 (lobes 0.58 m) is the user\'s choice (2026-10-09)')
ap.add_argument('--plot', default=None)
ap.add_argument('--dst', default=None, help='output directory (default ../lifted_linear_tire_20261009/tracks_tight)')
a = ap.parse_args()
DST = a.dst or DST


def cut_straights(x, y, c, r):
    """square2fast: its two long sides are straight (heading within 4 deg of +-y) for y in -0.76..0.27; the band
    [c - r/2, c + r/2] inside that is cut out and the two halves joined (points above move down r/2, below up r/2),
    so the corners are unchanged and the straights just shorter."""
    keep = np.abs(y - c) >= r / 2
    y2 = np.where(y > c, y - r / 2, y + r / 2)
    return x[keep], y2[keep], keep


def squash(y, f):
    """figfast (no straights: 0.2 m near-vertical per lobe): a uniform y-scale about the centre -- of the options
    compared (lobe-local bands of 0.6-1.1 m half-width: max |kappa| 2.9-3.6 1/m), the gentlest on curvature
    (max |kappa| 1.59 -> 2.13 1/m, the 95th percentile 1.39 -> 1.76)."""
    c = 0.5 * (y.min() + y.max())
    return c + (y - c) * f


os.makedirs(DST, exist_ok=True)
rows = []
for name in ('mocap_square2fast', 'mocap_figfast'):
    h = dict(np.load(os.path.join(SRC, f'{name}.npz'), allow_pickle=True))
    x, y = np.asarray(h['x'], float), np.asarray(h['y'], float)
    ext = y.max() - y.min()
    if name == 'mocap_square2fast':
        x2, y2, keep = cut_straights(x, y, 0.5 * (y.min() + y.max()), 2 * a.each)
        how = f'straights cut by {2 * a.each:.2f} m'
    else:
        x2, y2, keep = x, squash(y, (ext - 2 * a.each_fig) / ext), np.ones(len(x), bool)
        how = f'y scaled by {(ext - 2 * a.each_fig) / ext:.3f}'
    for k in [k for k, v in h.items() if np.ndim(v) >= 1 and np.shape(v)[-1] == len(x) and k not in ('x', 'y', 'theta')]:
        h[k] = np.asarray(h[k])[..., keep]          # per-waypoint fields (speed, vs) follow the kept waypoints
    h['x'], h['y'] = x2, y2
    h['theta'] = np.unwrap(np.arctan2(np.gradient(y2), np.gradient(x2)))
    np.savez(os.path.join(DST, f'{name}.npz'), **h)
    print(f'{name}: {how}: y {y.min():.2f}..{y.max():.2f} -> {y2.min():.2f}..{y2.max():.2f} m '
          f'(long extent {ext:.2f} -> {y2.max() - y2.min():.2f}), x {x.min():.2f}..{x.max():.2f} kept')
    rows.append((name, x, y, x2, y2))

# curvature / length check on the arc-length splines the pipeline uses
os.environ['F1T_TRACKS'] = DST
import tracks                                     # noqa: E402
for name, *_ in rows:
    t_new = tracks.Track(name)
    tracks.TRACK_DIR = SRC
    t_old = tracks.Track(name)
    tracks.TRACK_DIR = DST
    ko, kn = np.abs(t_old.kappa_smooth()), np.abs(t_new.kappa_smooth())
    print(f'{name}: length {t_old.length:.2f} -> {t_new.length:.2f} m, max |kappa| {ko.max():.2f} -> {kn.max():.2f} 1/m, '
          f'95th pct {np.percentile(ko, 95):.2f} -> {np.percentile(kn, 95):.2f}')
if a.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 6))
    for k, (name, x, y, x2, y2) in enumerate(rows):
        ax[k].plot(x, y, color='0.7', lw=1, label='original')
        ax[k].plot(x2, y2, color='#0072B2', lw=2, label='tight')
        ax[k].set_aspect('equal'); ax[k].grid(alpha=0.3); ax[k].set_title(name)
    ax[0].legend(fontsize=8)
    fig.savefig(a.plot, dpi=90)
