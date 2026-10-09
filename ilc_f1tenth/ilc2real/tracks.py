"""
Track geometry from llampc's mocap raceline waypoints (lifted_linear_tire_20261009/tracks/*.npz): a periodic cubic
spline through the waypoints, resampled by arc length. The path is the CoG's.
"""
import os

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d

TRACK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lifted_linear_tire_20261009', 'tracks')


class Track:
    def __init__(self, name, ds=0.005):
        h = np.load(os.path.join(TRACK_DIR, f'{name}.npz'), allow_pickle=True)
        pts = np.c_[h['x'], h['y']]
        keep = np.r_[True, np.hypot(*np.diff(pts, axis=0).T) > 1e-6]
        pts = pts[keep]
        if np.hypot(*(pts[-1] - pts[0])) < 1e-6:
            pts = pts[:-1]
        sp = np.asarray(h['speed'], float)
        self.design_speed = float(np.asarray(h['vs'][0]).mean()) if sp.ndim == 0 and 'vs' in h.files \
            else float(np.mean(sp))
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=True)
        uu = np.linspace(0, 1, 200001)
        xy = np.array(splev(uu, tck)).T
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        self.length = float(arc[-1])
        self.name = name
        n = int(round(self.length / ds))
        self.ds = self.length / n
        self.s = np.arange(n) * self.ds                    # periodic grid, s in [0, L)
        u_s = np.interp(self.s, arc, uu)
        self.xy = np.array(splev(u_s, tck)).T
        d1 = np.array(splev(u_s, tck, der=1)).T
        d2 = np.array(splev(u_s, tck, der=2)).T
        self.theta = np.unwrap(np.arctan2(d1[:, 1], d1[:, 0]))
        self.kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.linalg.norm(d1, axis=1) ** 3
        self.turns = (self.theta[-1] + (self.theta[1] - self.theta[0]) - self.theta[0]) / (2 * np.pi)
        self.turn_offset = 2 * np.pi * np.round(self.turns)  # theta(s + L) = theta(s) + turn_offset

    def kappa_smooth(self, sigma_m=0.15):
        return gaussian_filter1d(self.kappa, sigma_m / self.ds, mode='wrap')

    def at(self, s, arr):
        """Periodic linear interpolation of a per-grid-point array at arc length s."""
        n = len(self.s)
        f = np.mod(s, self.length) / self.ds
        i = np.floor(f).astype(int) % n
        w = f - np.floor(f)
        return (1 - w)[..., None] * arr[i] + w[..., None] * arr[(i + 1) % n] if arr.ndim > 1 else \
            (1 - w) * arr[i] + w * arr[(i + 1) % n]

    def theta_at(self, s):
        s = np.asarray(s, float)
        laps = np.floor(s / self.length)
        s_ext = np.r_[self.s, self.length]
        th_ext = np.r_[self.theta, self.theta[0] + self.turn_offset]
        return np.interp(s - laps * self.length, s_ext, th_ext) + laps * self.turn_offset

    def to_global(self, s, e_y, e_psi):
        xy = self.at(s, self.xy)
        th = self.theta_at(s)
        return xy[..., 0] - e_y * np.sin(th), xy[..., 1] + e_y * np.cos(th), th + e_psi
