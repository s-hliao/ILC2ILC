#!/usr/bin/env python3
"""make_ref_plans.py TRACK --speed V | --row K [--tag TAG] (2026-10-31; user: "there shouldn't even be a solve, the ref trajectory
is given"; "planned speed can be high"; plastic gets a lower target speed than rubber): the plan built DIRECTLY from the given
reference -- the track's path and a target speed -- with no trajectory optimization. Along the path:
  states   e_y = e_psi = 0, vx = v(s), vy = 0, r = v kappa_s, omega_w and I = the motor's steady state at v (straight line,
           car_model.steady_state_residual with R -> inf; a 2-unknown root find per speed), delta = atan(L kappa_s)
  inputs   [delta, I] = the feed-forward of the same
written in trajopt_drift's plan format (Z / Zc / U on the Radau collocation grid, dense_plan's fields), so every network, the
LQR expert (plan_lqr), the stage and the evaluation read it unchanged. The plan may exceed the tire's grip: the car does what it
can, and pace is scored against this reference.
  --speed V   a constant target speed (m/s)          --row K   the track file's own speed profile row K (vs[K] / speed)
  --scale X   multiplies the speed profile           -> $F1T_PLANS/<TRACK>_b0_ref<TAG>.npz
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import fsolve

import car_model as cm
from tracks import Track, TRACK_DIR
from trajopt_drift import NZ, dense_plan, radau3


def steady_straight(v, p):
    """omega_w, I holding speed v on a straight (vx' = 0, omega_w' = 0)."""
    def res(x):
        w, I = x
        f = cm.dynamics([0, 0, 0, v, 0.0, 0.0, w, I, 0.0], [0.0, I], p, cm._NP)
        return [f[3], f[6]]
    w, I = fsolve(res, [v / p['rw'], 2.0 + 3.0 * v])
    return float(w), float(I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('track')
    ap.add_argument('--speed', type=float, default=0.0)
    ap.add_argument('--row', type=int, default=-1)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--N', type=int, default=0)
    ap.add_argument('--out', default=os.environ.get('F1T_PLANS', '.'))
    a = ap.parse_args()
    tr = Track(a.track)
    p = dict(cm.NOMINAL)
    L_wb = p['lf'] + p['lr']
    N = a.N or int(round(tr.length / 0.06))
    h = tr.length / N
    if a.speed > 0:
        v_of = lambda s: np.full_like(np.asarray(s, float), a.speed)
    else:
        raw = np.load(os.path.join(TRACK_DIR, f'{a.track}.npz'), allow_pickle=True)
        prof = np.asarray(raw['vs'][a.row] if a.row >= 0 and 'vs' in raw.files else raw['speed'], float)
        if prof.ndim == 0:
            prof = np.full(len(raw['x']), float(prof))
        s_raw = np.linspace(0.0, tr.length, len(prof), endpoint=False)
        v_of = lambda s: np.interp(np.mod(s, tr.length), s_raw, prof, period=tr.length)
    v_of_s = lambda s: a.scale * v_of(s)
    vs_grid = np.linspace(0.3, 8.0, 78)
    ss = np.array([steady_straight(v, p) for v in vs_grid])          # (w, I) per speed
    ks = tr.kappa_smooth()

    def z_at(s):
        s = np.asarray(s, float)
        v = v_of_s(s)
        k = np.interp(np.mod(s, tr.length), tr.s, ks, period=tr.length)
        w = np.interp(v, vs_grid, ss[:, 0])
        I = np.interp(v, vs_grid, ss[:, 1])
        delta = np.clip(np.arctan(L_wb * k), -p['s_max'], p['s_max'])
        return np.stack([np.zeros_like(v), np.zeros_like(v), v, np.zeros_like(v), v * k, w, I, delta])   # (NZ, len(s))

    tau, _, _ = radau3()
    s_k = np.arange(N) * h
    Zs = z_at(s_k)
    Zcs = [z_at(np.array([(k + tau[1]) * h, (k + tau[2]) * h])) for k in range(N)]
    Us = np.stack([Zs[7], Zs[6]])                                     # [delta, I]
    plan = dense_plan(tr, (Zs, Zcs, Us), N, p)
    info = dict(success=True, status='reference (no solve)', lap_time=plan['lap_time'], track=a.track, N=N,
                speed=a.speed, row=a.row, scale=a.scale, source='make_ref_plans.py')
    out = os.path.join(a.out, f'{a.track}_b0_ref{a.tag}.npz')
    np.savez(out, info=json.dumps(info), N=N, Z=Zs, Zc=np.stack(Zcs), U=Us, **plan)
    v = plan['z'][:, 2]
    print(f'{out}: lap {plan["lap_time"]:.2f} s, v {v.min():.2f}..{v.max():.2f} m/s, |delta| max '
          f'{np.degrees(np.abs(plan["z"][:, 7]).max()):.1f} deg, lateral accel max {np.max(np.abs(v ** 2 * plan["kappa_s"])):.1f} m/s^2')


if __name__ == '__main__':
    main()
