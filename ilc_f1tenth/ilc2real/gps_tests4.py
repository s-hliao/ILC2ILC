#!/usr/bin/env python3
"""gps_tests.py (2026-11-07): independent checks of gps_core3.py (MDGPS core) before any GPS run.
 1 lqr_maxent mean controller == the brute-force QP optimum of the same surrogate objective (deterministic dynamics)
 2 propagate == Monte Carlo state marginals;  3 expected_cost == Monte Carlo;  4 kl_traj == Monte Carlo
 5 c_step hits |KL - eps| < 0.1 eps; the predicted cost falls as eps grows;  6 c_step from a non-PD eta0 recovers (del doubling)
 7 fit_dynamics recovers a known A, B, F with many samples; returns the prior with no data weight
 8 new_step_mult == the code's _set_new_mult formula"""
import numpy as np
import gps_core4 as g

rng = np.random.default_rng(0)
ok = True


def check(name, cond, info=''):
    global ok
    ok &= bool(cond)
    print(('PASS ' if cond else 'FAIL ') + name + (' ' + info if info else ''), flush=True)


def random_problem(T=4, dx=3, du=2):
    A = np.stack([np.eye(dx) + 0.1 * rng.standard_normal((dx, dx)) for _ in range(T)])
    B = 0.3 * rng.standard_normal((T, dx, du))
    M = rng.standard_normal((T, dx, dx)); Cxx = np.einsum('tij,tkj->tik', M, M) + 0.1 * np.eye(dx)
    cx = rng.standard_normal((T, dx)); c0 = rng.uniform(0, 1, T)
    Kp = 0.2 * rng.standard_normal((T, du, dx)); kp = 0.1 * rng.standard_normal((T, du))
    Sp = np.diag(rng.uniform(0.05, 0.2, du))
    F = np.stack([np.diag(rng.uniform(1e-3, 1e-2, dx)) for _ in range(T)])
    S0 = np.diag(rng.uniform(1e-3, 1e-2, dx))
    return A, B, Cxx, cx, c0, Kp, kp, Sp, F, S0


# 1 LQR vs brute-force QP over the controls, for a fixed x0 (deterministic part)
A, B, Cxx, cx, c0, Kp, kp, Sp, F, S0 = random_problem()
T, dx, du = B.shape
eta = 0.7
K, k, Quu = g.lqr_maxent(A, B, Cxx, cx, Kp, kp, np.repeat(np.linalg.inv(Sp)[None], T, 0), eta, lu_reg=0.0)
x0 = rng.standard_normal(dx)
Spi = np.linalg.inv(Sp)


def J(U):
    x = x0.copy(); tot = 0.0
    for t in range(T):
        u = U[t * du:(t + 1) * du]
        tot += (0.5 * x @ Cxx[t] @ x + cx[t] @ x) / eta
        d = u - Kp[t] @ x - kp[t]
        tot += 0.5 * d @ Spi @ d
        x = A[t] @ x + B[t] @ u
    return tot


# J is quadratic in U: recover its Hessian / gradient by finite differences and solve exactly
n = T * du; U0 = np.zeros(n); h = 1e-4
grad = np.array([(J(U0 + h * e) - J(U0 - h * e)) / (2 * h) for e in np.eye(n)])
H = np.array([[(J(U0 + h * (ei + ej)) - J(U0 + h * ei) - J(U0 + h * ej) + J(U0)) / h ** 2 for ej in np.eye(n)] for ei in np.eye(n)])
U_qp = np.linalg.solve(0.5 * (H + H.T), -grad)
x = x0.copy(); U_lqr = []
for t in range(T):
    u = K[t] @ x + k[t]; U_lqr.append(u); x = A[t] @ x + B[t] @ u
U_lqr = np.concatenate(U_lqr)
check('1 lqr_maxent == brute-force QP', np.allclose(U_qp, U_lqr, atol=1e-4), f'max diff {np.abs(U_qp - U_lqr).max():.2e}')

# 2-4 Monte Carlo
Su = np.linalg.inv(Quu)
mu, Sg = g.propagate(A, B, F, S0, K, k, Su)
N = 200000
X = rng.multivariate_normal(np.zeros(dx), S0, N); cost = np.zeros(N); kl = np.zeros(N); Xs = [X]
for t in range(T):
    cost += 0.5 * np.einsum('ni,ij,nj->n', X, Cxx[t], X) + X @ cx[t] + c0[t]
    m = X @ K[t].T + k[t]; mp = X @ Kp[t].T + kp[t]
    dm = m - mp
    kl += 0.5 * (np.trace(Spi @ Su[t]) - du + np.linalg.slogdet(Sp)[1] - np.linalg.slogdet(Su[t])[1] + np.einsum('ni,ij,nj->n', dm, Spi, dm))
    Uu = m + rng.multivariate_normal(np.zeros(du), Su[t], N)
    X = X @ A[t].T + Uu @ B[t].T + rng.multivariate_normal(np.zeros(dx), F[t], N); Xs.append(X)
emu = np.array([x.mean(0) for x in Xs]); eS = np.array([np.cov(x.T) for x in Xs])
check('2 propagate mean == MC', np.allclose(mu, emu, atol=0.01), f'max {np.abs(mu - emu).max():.3e}')
check('2 propagate cov == MC', np.allclose(Sg, eS, atol=0.01 * np.abs(Sg).max() + 1e-4), f'max {np.abs(Sg - eS).max():.3e}')
ec = g.expected_cost(mu, Sg, Cxx, cx, c0)
check('3 expected_cost == MC', abs(ec - cost.mean()) < 3 * cost.std() / np.sqrt(N) + 1e-3, f'{ec:.4f} vs {cost.mean():.4f}')
ek = g.kl_traj(mu, Sg, K, k, Su, Kp, kp, np.repeat(Sp[None], T, 0), np.repeat(Spi[None], T, 0))
check('4 kl_traj == MC', abs(ek - kl.mean()) < 3 * kl.std() / np.sqrt(N) + 1e-3, f'{ek:.4f} vs {kl.mean():.4f}')

# 5 c_step
res = []
for eps in (0.05, 0.2, 1.0, 3.0):
    s = g.c_step(A, B, F, S0, Cxx, cx, c0, Kp, kp, Sp, eps * T)
    res.append(s)
    check(f'5 c_step KL at eps {eps * T:.2f}', abs(s['kl'] - eps * T) < 0.1 * eps * T, f'kl {s["kl"]:.4f} eta {s["eta"]:.3g} iters {s["iters"]}')
check('5 predicted cost falls with eps', all(res[i + 1]['cost'] <= res[i]['cost'] + 1e-9 for i in range(len(res) - 1)), str([round(r['cost'], 4) for r in res]))

# 6 non-PD start: a Quu that is not PD at small eta (negative-curvature cost) -> del doubling recovers
Cneg = -5.0 * np.repeat(np.eye(dx)[None], T, 0)
s6 = g.c_step(A, B, F, S0, Cneg, cx, c0, Kp, kp, Sp, 0.5 * T, eta0=1e-6)
check('6 c_step recovers from non-PD (del doubling)', s6 is not None and np.all(np.linalg.eigvalsh(s6['Quu']) > 0), '' if s6 is None else f'eta {s6["eta"]:.3g} kl {s6["kl"]:.3f}')

# 7 fit_dynamics
T7, S7 = 3, 4000
At = np.stack([np.eye(dx) + 0.1 * rng.standard_normal((dx, dx)) for _ in range(T7)]); Bt = 0.3 * rng.standard_normal((T7, dx, du))
Ft = np.diag([1e-3, 2e-3, 5e-4])
Xd = np.zeros((S7, T7 + 1, dx)); Ud = rng.standard_normal((S7, T7, du)) * 0.3; Xd[:, 0] = rng.standard_normal((S7, dx)) * 0.2
for t in range(T7):
    Xd[:, t + 1] = Xd[:, t] @ At[t].T + Ud[:, t] @ Bt[t].T + rng.multivariate_normal(np.zeros(dx), Ft, S7)
V = np.ones((S7, T7), bool)
A0 = np.zeros_like(At); B0 = np.zeros_like(Bt)
Af, Bf, Ff, xm, um, ns = g.fit_dynamics(Xd, Ud, V, A0, B0, dx + du, np.full(dx, 0.04), np.full(du, 0.09), Fprior=np.eye(dx) * 1e-3)
check('7 fit_dynamics recovers A, B (many samples, wrong prior)', np.abs(Af - At).max() < 0.03 and np.abs(Bf - Bt).max() < 0.03,
      f'A err {np.abs(Af - At).max():.3f} B err {np.abs(Bf - Bt).max():.3f}')
check('7 fit_dynamics recovers F', np.abs(Ff - Ft).max() < 3e-4, f'F err {np.abs(Ff - Ft).max():.2e}')
Ap, Bp, Fp, *_ = g.fit_dynamics(Xd[:5], Ud[:5], V[:5], At, Bt, 1e9, np.full(dx, 0.04), np.full(du, 0.09), Fprior=Ft)
check('7 fit_dynamics returns the prior at infinite strength', np.abs(Ap - At).max() < 1e-4 and np.abs(Bp - Bt).max() < 1e-4 and np.abs(Fp - Ft).max() < 1e-6)

# 8 step rule
def ref(sm, pl, pp, cl):
    pi, ai = pl - pp, pl - cl
    nm = max(0.1, min(5.0, pi / (2.0 * max(1e-4, pi - ai))))
    return max(min(nm * sm, 10.0), 0.01)
cases = [(1.0, 10, 5, 6), (1.0, 10, 5, 12), (2.0, 10, 9, 9), (0.02, 10, 5, 20), (5.0, 10, 5, 4)]
check('8 new_step_mult == _set_new_mult', all(abs(g.new_step_mult(*c) - ref(*c)) < 1e-12 for c in cases))

# 9 gauss_fit_joint_prior: weak prior, many points of y = Kx + k + noise -> K, k, noise cov
dX9, dU9, N9 = 4, 2, 5000
K9 = rng.standard_normal((dU9, dX9)); k9 = rng.standard_normal(dU9); S9 = np.diag([0.01, 0.02])
X9 = rng.standard_normal((N9, dX9)); Y9 = X9 @ K9.T + k9 + rng.multivariate_normal(np.zeros(dU9), S9, N9)
fd, fc, ds = g.gauss_fit_joint_prior(np.c_[X9, Y9], np.zeros(dX9 + dU9), 1e-3 * np.eye(dX9 + dU9), 1e-3, 1e-3, np.full(N9, 1.0 / N9), dX9, dU9, 0)
check('9 gauss_fit_joint_prior recovers K, k, cov', np.abs(fd - K9).max() < 0.02 and np.abs(fc - k9).max() < 0.02 and np.abs(ds - S9).max() < 2e-3,
      f'K {np.abs(fd - K9).max():.3f} k {np.abs(fc - k9).max():.3f} S {np.abs(ds - S9).max():.4f}')
# 10 PolicyPriorGMM.fit on a linear policy, 5 samples per t (the GPS regime), state-varying across t: recovers the policy's K
np.random.seed(0)
T10, N10, dX10, dU10 = 40, 5, 4, 2
K10 = rng.standard_normal((dU10, dX10)) * 0.5; k10 = rng.standard_normal(dU10) * 0.1
X10 = np.cumsum(rng.standard_normal((N10, T10, dX10)) * 0.1, 1) + rng.standard_normal((1, 1, dX10))
pol = lambda O: O @ K10.T + k10
pp = g.PolicyPriorGMM(); pp.update(X10, X10, np.ones((N10, T10), bool), pol)
pK, pk, pS = pp.fit(X10, pol(X10.reshape(-1, dX10)).reshape(N10, T10, dU10), 0.01 * np.eye(dU10), np.ones((N10, T10), bool))
errK = np.abs(pK - K10[None]).max(); errk = np.abs(pk - k10[None]).max()
check('10 PolicyPriorGMM.fit recovers a linear policy from 5 samples / t', errK < 0.05 and errk < 0.05 and np.allclose(pS, 0.01 * np.eye(dU10), atol=1e-3),
      f'K err {errK:.4f} k err {errk:.4f}')
# 11 c_step with a time-varying Sp
Spt = np.stack([np.diag(rng.uniform(0.05, 0.2, du)) for _ in range(T)])
s11 = g.c_step(A, B, F, S0, Cxx, cx, c0, Kp, kp, Spt, 0.5 * T)
check('11 c_step with time-varying Sp hits KL', abs(s11['kl'] - 0.5 * T) < 0.1 * 0.5 * T, f'kl {s11["kl"]:.3f}')
print('ALL PASS' if ok else 'SOME FAILED')
