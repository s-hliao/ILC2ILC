"""Drift parking plans: from a steady drift on a circle into a parking slot, at rest, in the NOMINAL car (car_model.py).

The car circles on a steady drift (radius R0, sideslip beta0: an equilibrium of the nominal model, as held on hardware
-- sq41 'free': 13-18 deg at 2.4-2.6 m/s, R ~1.7 m) and leaves it to stop in a slot: position (X, Y) and heading psi,
v = 0. Time-domain direct collocation (Hermite-Simpson; the arc-length plans of trajopt_drift.py cannot stop), free
final time, IPOPT. Cost: time + control rates + a sideslip target during the spiral (|beta| >= beta_min while the car
is fast, so the entry stays a drift rather than a grip turn) + small terminal speed. Terminal constraints: the slot pose
within (tol_xy, tol_psi), |v| <= v_end.

Slots are given relative to the drift circle's start state, so a bank of plans is a set of (dx, dy, dpsi) offsets.

    F1T_MU=0.6 python -m ilc_park.trajopt --R0 1.5 --beta0 -15 --slot 1.2 1.0 90 --out plans_park/
"""
import argparse
import json
import os
import time

import casadi as ca
import numpy as np

from ilc_park import car_model as cm

NX, NU = cm.NX, cm.NA


def drift_equilibrium(R, beta_deg, p, guess=(2.4, 60.0, -0.1, 20.0)):
    """[V, omega_w, delta, I] of the steady drift on a circle of radius R at sideslip beta (the car turning left, rear
    out: beta < 0 for a left turn in LLA's convention)."""
    o = cm.casadi_ops()
    z = ca.SX.sym('z', 4)
    res = cm.steady_state_residual([z[i] for i in range(4)], R, np.radians(beta_deg), p, o)
    solver = ca.rootfinder('eq', 'newton', ca.Function('f', [z], [res]), {'max_iter': 200, 'abstol': 1e-10})
    sol = np.array(solver(np.array(guess, float))).ravel()
    return sol


def start_state(R, beta_deg, p):
    """The drift state at the circle's start point: at the origin heading +x, turning left (centre at (0, R))."""
    V, w, delta, I = drift_equilibrium(R, beta_deg, p)
    beta = np.radians(beta_deg)
    # velocity direction along +x (tangent of the left circle at the origin); the body is rotated by -beta from it
    psi = -beta
    x0 = np.array([0.0, 0.0, psi, V * np.cos(beta), V * np.sin(beta), V / R, w, I, delta])
    return x0, (V, w, delta, I)


def solve(x0, slot, p, N=80, T_max=4.0, beta_min_deg=10.0, v_drift=1.2, tol_xy=0.03, tol_psi_deg=3.0, v_end=0.05,
          w=None, guess=None, max_cpu_s=120.0):
    """One plan; returns (X (N+1, 9), U (N+1, 2), T, info)."""
    w = {**dict(T=1.0, du=(0.5, 1e-4), beta=20.0, end_v=5.0), **(w or {})}
    o = cm.casadi_ops()
    xs = ca.SX.sym('x', NX)
    us = ca.SX.sym('u', NU)
    f = ca.Function('f', [xs, us], [cm.dynamics([xs[i] for i in range(NX)], [us[0], us[1]], p, o)])
    opti = ca.Opti()
    X = opti.variable(NX, N + 1)
    Xm = opti.variable(NX, N)                     # Hermite-Simpson midpoints
    U = opti.variable(NU, N + 1)
    T = opti.variable()
    h = T / N
    J = w['T'] * T
    beta_min = np.radians(beta_min_deg)
    for k in range(N):
        fk, fk1 = f(X[:, k], U[:, k]), f(X[:, k + 1], U[:, k + 1])
        um = 0.5 * (U[:, k] + U[:, k + 1])
        fm = f(Xm[:, k], um)
        opti.subject_to(Xm[:, k] == 0.5 * (X[:, k] + X[:, k + 1]) + h / 8 * (fk - fk1))
        opti.subject_to(X[:, k + 1] == X[:, k] + h / 6 * (fk + 4 * fm + fk1))
        du = U[:, k + 1] - U[:, k]
        J += (w['du'][0] * du[0] ** 2 + w['du'][1] * du[1] ** 2) / h
        # sideslip target: while fast, keep |beta| >= beta_min (a soft hinge, smooth: softplus)
        vx, vy = X[3, k], X[4, k]
        V2 = vx ** 2 + vy ** 2
        beta = ca.atan2(vy, ca.sqrt(vx ** 2 + 1e-3))
        fast = 0.5 * (1 + ca.tanh(4 * (ca.sqrt(V2 + 1e-4) - v_drift)))
        short = beta_min - ca.fabs(beta)
        J += h * w['beta'] * fast * ca.log1p(ca.exp(10 * short)) / 10
    for k in range(N + 1):
        opti.subject_to(opti.bounded(-p['s_max'], U[0, k], p['s_max']))
        opti.subject_to(opti.bounded(p['I_min'], U[1, k], p['I_max']))
        opti.subject_to(opti.bounded(-p['s_max'], X[8, k], p['s_max']))
        opti.subject_to(X[6, k] >= 0)
    opti.subject_to(X[:, 0] == x0)
    dx, dy, dpsi = slot
    goal = np.array([dx, dy])
    opti.subject_to(opti.bounded(-tol_xy, X[0, N] - goal[0], tol_xy))
    opti.subject_to(opti.bounded(-tol_xy, X[1, N] - goal[1], tol_xy))
    opti.subject_to(opti.bounded(-np.radians(tol_psi_deg), X[2, N] - dpsi, np.radians(tol_psi_deg)))
    opti.subject_to(X[3, N] ** 2 + X[4, N] ** 2 <= v_end ** 2)
    opti.subject_to(opti.bounded(-0.2, X[5, N], 0.2))
    opti.subject_to(opti.bounded(0.3, T, T_max))
    J += w['end_v'] * (X[3, N] ** 2 + X[4, N] ** 2)
    opti.minimize(J)
    if guess is None:                              # straight-line-in-time guess: x0 decaying to the slot at rest
        s = np.linspace(0, 1, N + 1)
        Xg = np.outer(x0, 1 - s)
        Xg[0] = (1 - s) * x0[0] + s * dx
        Xg[1] = (1 - s) * x0[1] + s * dy
        Xg[2] = (1 - s) * x0[2] + s * dpsi
        Xg[6] = np.maximum(Xg[6], 0)
        opti.set_initial(X, Xg)
        opti.set_initial(Xm, 0.5 * (Xg[:, :-1] + Xg[:, 1:]))
        opti.set_initial(U, np.stack([x0[8] * (1 - s), x0[7] * (1 - s)]))
        opti.set_initial(T, 2.0)
    else:
        Xg, Xmg, Ug, Tg = guess
        opti.set_initial(X, Xg)
        opti.set_initial(Xm, Xmg)
        opti.set_initial(U, Ug)
        opti.set_initial(T, Tg)
    opti.solver('ipopt', {'print_time': False, 'expand': True},
                {'print_level': 0, 'sb': 'yes', 'max_iter': 3000, 'tol': 1e-6, 'acceptable_tol': 1e-4,
                 'mu_strategy': 'adaptive', 'max_cpu_time': float(max_cpu_s)})
    t0 = time.time()
    try:
        sol = opti.solve()
        ok = True
    except RuntimeError:
        sol, ok = opti.debug, False
    Xs, Xms, Us, Ts = (np.array(sol.value(v)) for v in (X, Xm, U, T))
    beta = np.degrees(np.arctan2(Xs[4], np.sqrt(Xs[3] ** 2 + 1e-3)))
    V = np.hypot(Xs[3], Xs[4])
    info = dict(success=ok, status=opti.stats().get('return_status'), T=float(Ts), seconds=time.time() - t0,
                slot=[float(dx), float(dy), float(np.degrees(dpsi))], beta_max=float(np.abs(beta).max()),
                beta_fast_min=float(np.abs(beta[V > v_drift]).min()) if (V > v_drift).any() else None,
                V_max=float(V.max()), end_err_xy=float(np.hypot(Xs[0, -1] - dx, Xs[1, -1] - dy)),
                end_err_psi_deg=float(np.degrees(Xs[2, -1] - dpsi)), end_v=float(V[-1]),
                I_range=[float(Us[1].min()), float(Us[1].max())], N=N)
    return (Xs, Xms, Us, Ts), info


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--R0', type=float, default=1.5)
    ap.add_argument('--beta0', type=float, default=-15.0, help='the drift circle sideslip (deg; left turn, rear out: < 0)')
    ap.add_argument('--slot', type=float, nargs=3, action='append', metavar=('DX', 'DY', 'DPSI_DEG'),
                    help='slot pose relative to the start state (repeatable)')
    ap.add_argument('--beta-min', type=float, default=10.0)
    ap.add_argument('--N', type=int, default=80)
    ap.add_argument('--T-max', type=float, default=4.0)
    ap.add_argument('--out', default='plans_park')
    a = ap.parse_args()
    p = dict(cm.NOMINAL)
    x0, eq = start_state(a.R0, a.beta0, p)
    print(json.dumps(dict(R0=a.R0, beta0=a.beta0, mu=p['muf'], equilibrium=dict(V=eq[0], omega_w=eq[1],
                     delta_deg=float(np.degrees(eq[2])), I=eq[3]))), flush=True)
    os.makedirs(a.out, exist_ok=True)
    for dx, dy, dpsi in a.slot or [(1.2, 1.0, 90.0)]:
        sol, info = solve(x0, (dx, dy, np.radians(dpsi)), p, N=a.N, T_max=a.T_max, beta_min_deg=a.beta_min)
        print(json.dumps(info), flush=True)
        if info['success']:
            name = f'park_R{a.R0:g}_b{abs(a.beta0):g}_x{dx:+.2f}_y{dy:+.2f}_p{dpsi:+.0f}'
            np.savez(os.path.join(a.out, name + '.npz'), X=sol[0], Xm=sol[1], U=sol[2], T=sol[3],
                     t=np.linspace(0, sol[3], a.N + 1), x0=x0, info=json.dumps(dict(info, R0=a.R0, beta0=a.beta0, mu=p['muf'])))


if __name__ == '__main__':
    main()
