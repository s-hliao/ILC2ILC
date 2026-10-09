"""
The plan's stabilizer: periodic LQR in arc length around a drift plan (the warm-start structure the network is
first cloned from; the quadruped's TO + LQR). The spatial map over one TO interval, z_{k+1} = Phi_k(z_k, u_k)
(RK4 in s, the nominal model), is linearized along the plan; the Riccati recursion is run backward over several
laps (periodic plan) until the gains repeat: u = u*(s) - K(s) (z - z*(s)).
"""
import casadi as ca
import numpy as np

import car_model as cm
from trajopt_drift import spatial_rhs, NZ

Q_DIAG = np.array([1 / 0.05**2, 1 / 0.10**2, 1 / 0.3**2, 1 / 0.3**2, 1 / 0.5**2, 0.0, 0.0, 1 / 0.2**2, 0.0])
R_DIAG = np.array([1 / 0.1**2, 1 / 3.0**2])   # swept: softer current gains avoid wheel-spin chatter


def lqr_gains(plan, p, substeps=40, laps=6, q=Q_DIAG, r=R_DIAG):
    Z, U, N = plan['Z'], plan['U'], int(plan['N'])
    L = float(plan['length'])
    h = L / N
    rhs = spatial_rhs(p)
    z = ca.SX.sym('z', NZ)
    u = ca.SX.sym('u', 2)
    k = ca.SX.sym('k')
    hh = h / substeps
    zz = z
    for _ in range(substeps):
        k1 = rhs(zz, u, k)[0]
        k2 = rhs(zz + hh / 2 * k1, u, k)[0]
        k3 = rhs(zz + hh / 2 * k2, u, k)[0]
        k4 = rhs(zz + hh * k3, u, k)[0]
        zz = zz + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    jac = ca.Function('phi_jac', [z, u, k], [ca.jacobian(zz, z), ca.jacobian(zz, u)])
    s_k = (np.arange(N) + 0.5) * h
    kap = np.interp(s_k, plan['s'], plan['kappa'])
    A = np.zeros((N, NZ, NZ))
    B = np.zeros((N, NZ, 2))
    for i in range(N):
        a, b = jac(Z[:, i], U[:, i], kap[i])
        A[i], B[i] = np.array(a), np.array(b)
    Qd, Rd = np.diag(q) * h, np.diag(r) * h
    P = np.diag(q)
    K = np.zeros((N, 2, NZ))
    for _ in range(laps):
        for i in reversed(range(N)):
            S = Rd + B[i].T @ P @ B[i]
            K[i] = np.linalg.solve(S, B[i].T @ P @ A[i])
            Acl = A[i] - B[i] @ K[i]
            P = Qd + K[i].T @ Rd @ K[i] + Acl.T @ P @ Acl
            P = 0.5 * (P + P.T)
    eig = np.abs(np.linalg.eigvals(np.linalg.multi_dot([(A[i] - B[i] @ K[i]) for i in reversed(range(N))])))
    open_eig = np.abs(np.linalg.eigvals(np.linalg.multi_dot([A[i] for i in reversed(range(N))])))
    return K, A, B, dict(closed_loop_lap_rho=float(eig.max()), open_loop_lap_rho=float(open_eig.max()))


def dense_gains(plan, K):
    """K on the fine s-grid (interval-wise constant)."""
    N = int(plan['N'])
    h = float(plan['length']) / N
    idx = np.minimum((plan['s'] / h).astype(int), N - 1)
    return K[idx]
