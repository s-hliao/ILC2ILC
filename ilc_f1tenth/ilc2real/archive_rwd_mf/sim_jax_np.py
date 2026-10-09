"""numpy twin of sim_jax's per-lane plan functions (projection, path coordinates, observation, ILC error, expert),
for the CPU real cars. Kept line-for-line with sim_jax; check_real_vs_sim.py compares them."""
import numpy as np

import car_model as cm
from consts import PREVIEW, DEV_SCALE, FAIL_EY, FAIL_EPSI, WIN   # noqa: F401


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


class PlanNP:
    def __init__(self, plan):
        self.xy = np.asarray(plan['xy'])
        self.theta = np.asarray(plan['theta'])
        self.kap = np.asarray(plan['kappa_s'])
        self.z = np.asarray(plan['z'])
        self.u = np.asarray(plan['u'])
        self.K = np.asarray(plan['K']) if 'K' in plan else np.zeros((len(self.z), 2, 9))
        self.M = len(self.xy)
        self.ds = float(plan['ds'])
        self.L = float(plan['length'])

    def project(self, idx, pos):
        ii = np.mod(idx + WIN, self.M)
        d2 = np.sum((self.xy[ii] - pos) ** 2, -1)
        return int(ii[np.argmin(d2)])

    def ref_at(self, i, off, arr):
        f = off / self.ds
        fl = np.floor(f)
        w = f - fl
        i0 = int(np.mod(i + int(fl), self.M))
        i1 = (i0 + 1) % self.M
        return (1 - w) * arr[i0] + w * arr[i1]

    def path_coords(self, i, x):
        th = self.theta[i]
        d = x[:2] - self.xy[i]
        s_off = d[0] * np.cos(th) + d[1] * np.sin(th)
        e_y = -d[0] * np.sin(th) + d[1] * np.cos(th)
        e_psi = wrap(x[2] - th - s_off * self.kap[i])
        return s_off, e_y, e_psi

    def features(self, i, x):
        s_off, e_y, e_psi = self.path_coords(i, x)
        zr = self.ref_at(i, s_off, self.z)
        Rw = cm.NOMINAL['R_w']
        dev = np.array([e_y - zr[0], wrap(e_psi - zr[1]), x[3] - zr[2], x[4] - zr[3], x[5] - zr[4],
                        (x[6] - zr[5]) * Rw, (x[7] - zr[6]) * Rw, x[8] - zr[7]])
        raw = np.array([x[3] / 3.0, x[4], x[5] / 3.0, x[8] / 0.4, x[7] * Rw / 4.0])
        pv = []
        for d in PREVIEW:
            z = self.ref_at(i, s_off + d, self.z)
            u = self.ref_at(i, s_off + d, self.u)
            k = self.ref_at(i, s_off + d, self.kap)
            pv.append([k, z[1], z[2] / 3.0, z[3], z[4] / 3.0, u[0] / 0.4, u[1] / 20.0])
        obs = np.concatenate([dev / DEV_SCALE, raw, np.ravel(pv)])
        return obs, dev[:5]

    def expert(self, i, x):
        s_off, e_y, e_psi = self.path_coords(i, x)
        zr = self.ref_at(i, s_off, self.z)
        ur = self.ref_at(i, s_off, self.u)
        K = self.ref_at(i, s_off, self.K)
        dz = np.array([e_y - zr[0], wrap(e_psi - zr[1]), x[3] - zr[2], x[4] - zr[3], x[5] - zr[4],
                       x[6] - zr[5], x[7] - zr[6], x[8] - zr[7], 0.0])
        return ur - K @ dz
