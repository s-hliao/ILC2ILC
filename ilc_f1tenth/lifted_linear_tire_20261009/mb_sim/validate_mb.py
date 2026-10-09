"""Offline checks of the MB plant (no ROS): substep convergence, rest stability, and the
same command sequences through the stock f110_gym single-track plant for comparison."""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mb_plant import MBPlant                                  # noqa: E402
from lifted_ilc import command_chain, F110                    # noqa: E402
from f110_gym.envs.dynamic_models import vehicle_dynamics_st, pid   # noqa: E402  (stock)

ST_PARAMS = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145,
             'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189,
             'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min': -5.0,
             'v_max': 20.0}                               # f110_env defaults (the stock plant)


class STPlant:
    """f110_gym RaceCar.update_pose with its RK4, for comparison."""
    def __init__(self, timestep=0.01):
        self.timestep = timestep
        self.reset((0, 0, 0))

    def reset(self, pose):
        self.x = np.zeros(7); self.x[0], self.x[1], self.x[4] = pose
        self.buf = np.empty((0,))

    def _f(self, x, u):
        P = ST_PARAMS
        return vehicle_dynamics_st(x, u, P['mu'], P['C_Sf'], P['C_Sr'], P['lf'], P['lr'], P['h'],
                                   P['m'], P['I'], P['s_min'], P['s_max'], P['sv_min'],
                                   P['sv_max'], P['v_switch'], P['a_max'], P['v_min'], P['v_max'])

    def step(self, raw_steer, speed):
        if self.buf.shape[0] < 2:
            steer = 0.; self.buf = np.append(raw_steer, self.buf)
        else:
            steer = self.buf[-1]; self.buf = np.append(raw_steer, self.buf[:-1])
        P = ST_PARAMS
        accl, sv = pid(speed, steer, self.x[3], self.x[2], P['sv_max'], P['a_max'], P['v_max'], P['v_min'])
        u, h, x = np.array([sv, accl]), self.timestep, self.x
        k1 = self._f(x, u); k2 = self._f(x + h / 2 * k1, u); k3 = self._f(x + h / 2 * k2, u)
        k4 = self._f(x + h * k3, u)
        self.x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def pose(self):
        return self.x[0], self.x[1], self.x[4]


def run(plant, steer_cmd, speed_cmd, dt_cmd, tick=0.01):
    """Commands held over dt_cmd, physics at tick (lockstep); poses at command instants."""
    plant.reset((0.0, 0.0, 0.0))
    poses, t, k = [], 0.0, 0
    n_ticks = int(round(len(steer_cmd) * dt_cmd / tick))
    for i in range(n_ticks):
        k = min(int((i * tick + 1e-9) // dt_cmd), len(steer_cmd) - 1)
        if i * tick + 1e-9 >= len(poses) * dt_cmd:
            poses.append(plant.pose() if hasattr(plant, 'pose') else plant.odom()[:3])
        plant.step(steer_cmd[k], speed_cmd[k])
    poses.append(plant.pose() if hasattr(plant, 'pose') else plant.odom()[:3])
    return np.asarray(poses)


def main():
    car = F110()
    out = {}
    for name in ['s_curve', 'figure_eight']:
        h = np.load(HERE.parent / f'results/40hz_{name}_lifted/initial_to.npz') \
            if (HERE.parent / f'results/40hz_{name}_lifted/initial_to.npz').exists() else None
        if h is None:
            continue
        ref, U, dt = h['reference'], h['controls'][0], float(h['dt'])
        chain = command_chain(U, ref[0], car, dt)                 # I, steer, speed commands
        steer, speed = chain[:, 1], chain[:, 2]
        res = {}
        if name == 's_curve':
            ref_pose = None
            for sub in [1e-3, 5e-4, 2.5e-4, 1.25e-4]:
                t0 = time.time()
                P = run(MBPlant(substep=sub), steer, speed, dt)
                res[f'mb_sub{sub:g}'] = dict(final=P[-1].tolist(), sec_per_tick_ms=1e3 * (time.time() - t0) / (len(steer) * 2.5))
            fine = np.array(res['mb_sub0.000125']['final'])
            for k in res:
                res[k]['final_err_vs_finest_m'] = float(np.hypot(*(np.array(res[k]['final'][:2]) - fine[:2])))
        mb = run(MBPlant(), steer, speed, dt)
        st = run(STPlant(), steer, speed, dt)
        d = np.hypot(mb[:, 0] - st[:, 0], mb[:, 1] - st[:, 1])
        res['mb_vs_st'] = dict(rms_m=float(np.sqrt(np.mean(d ** 2))), max_m=float(d.max()),
                               mb_rmse_to_ref=float(np.sqrt(np.mean(np.sum((mb[:, :2] - ref[:, :2]) ** 2, 1)))),
                               st_rmse_to_ref=float(np.sqrt(np.mean(np.sum((st[:, :2] - ref[:, :2]) ** 2, 1)))))
        out[name] = res
    # rest: zero commands for 2 s
    pl = MBPlant(); pl.reset((0, 0, 0))
    for _ in range(200):
        pl.step(0.0, 0.0)
    out['rest_2s'] = dict(pos=list(pl.odom()[:3]), vx=pl.x[3], z=pl.x[11], roll=pl.x[6], pitch=pl.x[8])
    # steady cornering sweep: speed, steer -> yaw rate and sideslip after 4 s
    sweep = []
    for v, s in [(1.0, 0.1), (2.0, 0.1), (2.0, 0.2), (3.0, 0.2), (3.0, 0.34), (5.0, 0.34)]:
        pm, ps = MBPlant(), STPlant()
        pm.reset((0, 0, 0)); ps.reset((0, 0, 0))
        for _ in range(400):
            pm.step(s, v); ps.step(s, v)
        sweep.append(dict(speed=v, steer=s, yaw_rate_mb=pm.x[5], yaw_rate_st=ps.x[5],
                          kinematic=v * np.tan(s) / (car['lf'] + car['lr']),
                          beta_mb=float(np.arctan2(pm.x[10], pm.x[3])), beta_st=ps.x[6],
                          roll_mb_deg=float(np.degrees(pm.x[6]))))
    out['steady_cornering'] = sweep
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
