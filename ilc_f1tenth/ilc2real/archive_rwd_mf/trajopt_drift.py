"""
The trial-1 plan / reference for a track (the quadruped's TO bank analog): a periodic lap of the nominal model
(car_model.py) along the track's path, by IPOPT (CasADi), direct collocation in ARC LENGTH (Radau IIA, degree 3:
stiff-stable for the wheel-spin modes; no time grid, so there is no phase to drift and the lap end is tied to the
lap start -- the plan can be flown lap after lap, indefinitely).

Spatial state z = [e_y, e_psi, vx, vy, r, w_f, w_r, delta, dFz] (lateral offset and heading relative to the path
tangent), action u = [delta_cmd, I_cmd] piecewise constant per interval. dz/ds = f(z, u) / s_dot with
s_dot = (vx cos e_psi - vy sin e_psi) / (1 - kappa e_y).

Cost per meter: w_y e_y^2 + w_b (beta - beta*(s))^2 + w_v (V - V*)^2 + w_r (r - r*(s))^2 + rates of u. The drift
target beta*(s) = -beta_d tanh(kappa_s(s) / kappa_0) (rear out on both turn directions; kappa_s the path curvature
smoothed over 15 cm), V* the track's design speed. Constraints: periodicity z(L) = z(0), |e_y| <= e_max, actuator
limits. Solved by homotopy in beta_d from grip (0) to the target, each solve warm-started from the last.

    python trajopt_drift.py mocap_square2fast --beta 25 --out plans/
"""
import argparse
import json
import os
import time

import casadi as ca
import numpy as np

import car_model as cm
from tracks import Track

NZ = 9


def radau3():
    tau = np.r_[0.0, ca.collocation_points(3, 'radau')]
    C = np.zeros((4, 4))
    D = np.zeros(4)
    for j in range(4):
        coeff = np.poly1d([1.0])
        for r in range(4):
            if r != j:
                coeff *= np.poly1d([1.0, -tau[r]]) / (tau[j] - tau[r])
        D[j] = coeff(1.0)
        dcoeff = np.polyder(coeff)
        for r in range(4):
            C[j, r] = dcoeff(tau[r])
    return tau, C, D


def spatial_rhs(p):
    o = cm.casadi_ops()
    z = ca.SX.sym('z', NZ)
    u = ca.SX.sym('u', 2)
    k = ca.SX.sym('k')
    e_y, e_psi, vx, vy, r = z[0], z[1], z[2], z[3], z[4]
    x = ca.vertcat(0, 0, 0, vx, vy, r, z[5], z[6], z[7], z[8])
    f = cm.dynamics([x[i] for i in range(10)], [u[0], u[1]], p, o)
    sdot = (vx * ca.cos(e_psi) - vy * ca.sin(e_psi)) / (1 - k * e_y)
    dz = ca.vertcat(vx * ca.sin(e_psi) + vy * ca.cos(e_psi), r - k * sdot, f[3], f[4], f[5], f[6], f[7], f[8], f[9])
    return ca.Function('rhs', [z, u, k], [dz / sdot, 1 / sdot])


def solve(track, beta_d, N, p, guess=None, w=None, e_max=0.3, v_star=None, max_iter=3000, kappa0=0.25):
    w = dict(y=200.0, b=60.0, v=4.0, r=0.5, du=(2.0, 2e-4), u=(0.0, 1e-5), **(w or {}))
    L = track.length
    h = L / N
    tau, C, D = radau3()
    ks = track.kappa_smooth()
    s_nodes = (np.arange(N)[:, None] + tau[None, 1:]) * h          # collocation points
    kap = track.at(s_nodes, track.kappa)
    beta_t = -beta_d * np.tanh(track.at(s_nodes, ks) / kappa0)
    V_t = v_star or track.design_speed
    rhs = spatial_rhs(p)
    opti = ca.Opti()
    Z = opti.variable(NZ, N)                  # interval starts (periodic: z_N = z_0)
    Zc = [opti.variable(NZ, 3) for _ in range(N)]
    U = opti.variable(2, N)
    J = 0
    T = 0
    for k in range(N):
        zk = Z[:, k]
        znext = Z[:, (k + 1) % N]
        pts = [zk] + [Zc[k][:, j] for j in range(3)]
        for j in range(1, 4):
            dz = sum(C[r, j] * pts[r] for r in range(4))
            fz, dt_ds = rhs(pts[j], U[:, k], kap[k, j - 1])
            opti.subject_to(dz == h * fz)
            zz = pts[j]
            beta = ca.atan2(zz[3], zz[2])
            V = ca.sqrt(zz[2] ** 2 + zz[3] ** 2)
            r_t = V * kap[k, j - 1]
            wq = h / 3.0
            J += wq * (w['y'] * zz[0] ** 2 + w['b'] * (beta - beta_t[k, j - 1]) ** 2 + w['v'] * (V - V_t) ** 2
                       + w['r'] * (zz[4] - r_t) ** 2)
            T += wq * dt_ds
        opti.subject_to(znext == sum(D[r] * pts[r] for r in range(4)))
        du = U[:, (k + 1) % N] - U[:, k]
        J += (w['du'][0] * du[0] ** 2 + w['du'][1] * du[1] ** 2) / h
        J += h * (w['u'][0] * U[0, k] ** 2 + w['u'][1] * U[1, k] ** 2)
        for zz in [zk] + [Zc[k][:, j] for j in range(3)]:
            opti.subject_to(opti.bounded(-e_max, zz[0], e_max))
            opti.subject_to(zz[2] >= 0.5)
            opti.subject_to(zz[5] >= 0)
            opti.subject_to(zz[6] >= 0)
            opti.subject_to(opti.bounded(-p['s_max'], zz[7], p['s_max']))
    opti.subject_to(opti.bounded(-p['s_max'], U[0, :], p['s_max']))
    opti.subject_to(opti.bounded(-p['I_max'], U[1, :], p['I_max']))
    opti.minimize(J)
    if guess is None:
        s_k = np.arange(N) * h
        kk = track.at(s_k, ks)
        b0 = -beta_d * np.tanh(kk / kappa0)
        vx0 = V_t * np.cos(b0)
        Zg = np.stack([0 * s_k, -b0, vx0, V_t * np.sin(b0), V_t * kk, vx0 / p['R_w'], vx0 / p['R_w'],
                       np.clip((p['lf'] + p['lr']) * kk, -0.3, 0.3), 0 * s_k])
        Ug = np.stack([Zg[7], 3.0 + 0 * s_k])
        Zcg = [np.repeat(Zg[:, k:k + 1], 3, 1) for k in range(N)]
    else:
        Zg, Zcg, Ug = guess
    opti.set_initial(Z, Zg)
    for k in range(N):
        opti.set_initial(Zc[k], Zcg[k])
    opti.set_initial(U, Ug)
    opti.solver('ipopt', {'print_time': False, 'expand': True},
                {'print_level': 0, 'sb': 'yes', 'max_iter': max_iter, 'tol': 1e-6, 'acceptable_tol': 1e-4,
                 'mu_strategy': 'adaptive'})
    t0 = time.time()
    try:
        sol = opti.solve()
        ok = True
    except RuntimeError:
        sol, ok = opti.debug, False
    st = opti.stats()
    Zs = np.array(sol.value(Z))
    Zcs = [np.array(sol.value(Zc[k])) for k in range(N)]
    Us = np.array(sol.value(U))
    info = dict(success=ok, status=st.get('return_status'), iterations=int(st.get('iter_count', -1)),
                cost=float(sol.value(J)), lap_time=float(sol.value(T)), seconds=time.time() - t0, beta_d=beta_d,
                N=N, track=track.name, v_star=V_t, solver='IPOPT (CasADi Opti), Radau IIA 3 collocation in s')
    return (Zs, Zcs, Us), info


def dense_plan(track, sol, N, p):
    """The plan on the track's fine s-grid (linear between collocation points), plus time along s and the
    global-frame states, for the policy's reference / preview and for resets."""
    Zs, Zcs, Us = sol
    L = track.length
    h = L / N
    tau, _, _ = radau3()
    # tau[1:] are the Radau nodes; tau[3] = 1 is the next interval's start (same as Zs[:, k+1])
    s_pts = np.concatenate([np.r_[k * h, (k + tau[1]) * h, (k + tau[2]) * h] for k in range(N)] + [[L]])
    z_pts = np.concatenate([np.c_[Zs[:, k], Zcs[k][:, 0], Zcs[k][:, 1]].T for k in range(N)] + [Zs[:, :1].T])
    z = np.stack([np.interp(track.s, s_pts, z_pts[:, i]) for i in range(NZ)], 1)
    u_idx = np.minimum((track.s / h).astype(int), N - 1)
    u = Us[:, u_idx].T
    sdot = (z[:, 2] * np.cos(z[:, 1]) - z[:, 3] * np.sin(z[:, 1])) / (1 - track.kappa * z[:, 0])
    t = np.r_[0, np.cumsum(track.ds / sdot)[:-1]]
    lap_time = float(np.sum(track.ds / sdot))
    X, Y, psi = track.to_global(track.s, z[:, 0], z[:, 1])
    glob = np.stack([X, Y, psi, z[:, 2], z[:, 3], z[:, 4], z[:, 5], z[:, 6], z[:, 7], z[:, 8]], 1)
    return dict(s=track.s, z=z, u=u, t=t, lap_time=lap_time, x=glob, kappa=track.kappa,
                kappa_s=track.kappa_smooth(), xy=track.xy, theta=track.theta, length=L, ds=track.ds,
                turn_offset=track.turn_offset, design_speed=track.design_speed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('track')
    ap.add_argument('--beta', type=float, default=25.0, help='target drift sideslip magnitude, deg')
    ap.add_argument('--steps', type=float, nargs='+', default=[0, 10, 18, 25])
    ap.add_argument('--N', type=int, default=0, help='intervals per lap (0: one per 6 cm)')
    ap.add_argument('--vstar', type=float, default=0.0)
    ap.add_argument('--T-se', type=float, default=0.0)
    ap.add_argument('--mu-plan', type=float, default=1.0, help='friction scale the plan is solved for (conservative plans: < 1)')
    ap.add_argument('--tag', default='', help='suffix of the saved plan names, e.g. _mu80')
    ap.add_argument('--out', default='plans')
    a = ap.parse_args()
    tr = Track(a.track)
    N = a.N or int(round(tr.length / 0.06))
    p = dict(cm.NOMINAL, T_se=a.T_se)
    p['mu_x'] *= a.mu_plan
    p['mu_y'] *= a.mu_plan
    os.makedirs(a.out, exist_ok=True)
    guess = None
    for b in [x for x in a.steps if x < a.beta] + [a.beta]:
        sol, info = solve(tr, np.radians(b), N, p, guess, v_star=a.vstar or None)
        print(json.dumps(info), flush=True)
        if info['success'] or guess is None:
            guess = sol
        if info['success']:
            plan = dense_plan(tr, sol, N, p)
            beta = np.degrees(np.arctan2(plan['z'][:, 3], plan['z'][:, 2]))
            print(f'  beta_d {b}: lap {plan["lap_time"]:.2f} s, |e_y| max {np.abs(plan["z"][:, 0]).max():.3f}, '
                  f'beta {beta.min():.1f}..{beta.max():.1f} deg, V {np.hypot(plan["z"][:, 2], plan["z"][:, 3]).mean():.2f}, '
                  f'delta {np.degrees(plan["z"][:, 7]).min():.1f}..{np.degrees(plan["z"][:, 7]).max():.1f}, '
                  f'I {plan["u"][:, 1].min():.1f}..{plan["u"][:, 1].max():.1f}', flush=True)
            np.savez(os.path.join(a.out, f'{a.track}_b{b:g}{a.tag}.npz'), info=json.dumps(dict(info, mu_plan=a.mu_plan)), N=N,
                     Z=sol[0], Zc=np.stack(sol[1]), U=sol[2], **plan)


if __name__ == '__main__':
    main()
