"""
Lifted-form (norm-optimal) ILC for the nine-state F1TENTH model, after the quadruped's
JumpILC (ilc_quad/ilc_gen.py): one constrained QP per trial on the lifted sensitivity
matrix G of a linear-tire model linearized along the trial just measured, and a measured-
cost safeguard that rolls a worse trial back to the best one with a shorter step.

    ilc = LiftedILC(reference, dt, U0)
    while True:
        X = fly(ilc.U)                 # (N+1, 9) public odometry + command states
        result = ilc.update(X)         # moves ilc.U for the next trial

State  x = [x, y, heading, vx, vy, yaw_rate, wheel_speed, current, steering_command]
Input  u = [current slew rate, steer rate]   (as the snapshot's blend model / NMPC)

Prediction (eq. 19 of the jumping paper):  e_{k+1} = e_k - G du,  e = x_ref - x_k.
At du = 0 it is the measured error exactly, so model bias is absorbed into e_k and no
additive defects are needed; only the sensitivities G come from the model.
"""
from pathlib import Path
import sys

import casadi as ca
import numpy as np

SNAPSHOT = Path(__file__).resolve().parent.parent / "successful_defect_aware_20261008"
sys.path.insert(0, str(SNAPSHOT))
from params import F110                                   # noqa: E402
from blend_solver import blend_model, linear_model, _get_tire_params  # noqa: E402

NX, NU = 9, 2


def linear_tire_model(x, u, car, Cf, Cr, v0=0.5):
    """
    Single-track model with linear tires, F_y = C alpha, small-angle slips over a
    guarded vx (sqrt(vx^2 + v0^2): finite at standstill, within 1.5% of vx at 3 m/s).
    Drive force from the current with no longitudinal slip, as blend_solver.linear_model;
    the wheel speed follows vx (the rollout derives it as vx / rw).
    """
    m, Iz, lf, lr, rw = car['mass'], car['Iz'], car['lf'], car['lr'], car['rw']
    k_drive = car['gear_ratio'] * 1.5 * car['pole_pairs'] * car['lambda'] / rw
    vx, vy, w, current, delta = x[3], x[4], x[5], x[7], x[8]
    vxd = ca.sqrt(vx**2 + v0**2)
    Ffy = Cf * (delta - (vy + lf * w) / vxd)
    Fry = Cr * ((lr * w - vy) / vxd)
    Frx = k_drive * current
    ax = (Frx - Ffy * ca.sin(delta)) / m + vy * w
    return ca.vertcat(vx * ca.cos(x[2]) - vy * ca.sin(x[2]),
                      vx * ca.sin(x[2]) + vy * ca.cos(x[2]),
                      w,
                      ax,
                      (Fry + Ffy * ca.cos(delta)) / m - vx * w,
                      (lf * Ffy * ca.cos(delta) - lr * Fry) / Iz,
                      ax / rw,
                      u[0],
                      u[1])


def make_model(kind, car, dt, substeps=5):
    """Discrete transition (RK4, u held over dt) and its Jacobians as casadi Functions.
    kind: "linear" (linear_tire_model), "blend_linear" (blend_solver.linear_model,
    slips over a fixed 1 m/s) or "blend" (the NMPC's blend model)."""
    p = _get_tire_params(car)
    x, u = ca.SX.sym('x', NX), ca.SX.sym('u', NU)
    if kind == "linear":
        f = lambda x_, u_: linear_tire_model(x_, u_, car, p[0], p[1])
    elif kind == "blend_linear":
        f = lambda x_, u_: linear_model(x_, u_, ca.DM(p), car)
    elif kind == "blend":
        f = lambda x_, u_: blend_model(x_, u_, ca.DM(p), car)
    else:
        raise ValueError(kind)
    h, xn = dt / substeps, x
    for _ in range(substeps):
        k1 = f(xn, u)
        k2 = f(xn + h / 2 * k1, u)
        k3 = f(xn + h / 2 * k2, u)
        k4 = f(xn + h * k3, u)
        xn = xn + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    step = ca.Function('step', [x, u], [xn])
    jac = ca.Function('step_jacobians', [x, u], [ca.jacobian(xn, x), ca.jacobian(xn, u)])
    return step, jac


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def tracking_error(reference, X):
    """x_ref - x, heading wrapped (the paper's e_k)."""
    e = np.asarray(reference, float) - np.asarray(X, float)
    e[..., 2] = wrap(e[..., 2])
    return e


def build_lifted_G(A_list, B_list):
    """G[t, j] = d x_{t+1} / d u_j, (N*nx, N*nu) time-major (ilc_gen.build_lifted_G)."""
    N = len(B_list)
    G = np.zeros((N, N, NX, NU))
    for j in range(N):
        G[j, j] = B_list[j]
        for t in range(j + 1, N):
            G[t, j] = A_list[t] @ G[t - 1, j]
    return G.transpose(0, 2, 1, 3).reshape(N * NX, N * NU)


def command_chain(U, x0, car, dt):
    """The controller-owned command states the rollout integrates from U
    (fixed_deadline_rollout): current I, steering command and speed command."""
    c = car['gear_ratio'] * 1.5 * car['pole_pairs'] * car['lambda'] / (car['mass'] * car['rw'])
    I, steer, v = x0[7], x0[8], x0[3]
    out = []
    for uI, ud in U:
        I += uI * dt
        v = max(0.0, v + c * I * dt)
        steer = float(np.clip(steer + ud * dt, -car['max_steer'], car['max_steer']))
        out.append((I, steer, v))
    return np.asarray(out)                                 # (N, 3), after each step


def trajectory_cost(X, U, reference, Q, Q_f, R):
    """The snapshot's task cost (ilc_f1tenth_ilqr_defect_aware.trajectory_cost)."""
    e = tracking_error(reference, X)
    return float(0.5 * np.einsum('ti,ij,tj->', e[:-1], Q, e[:-1])
                 + 0.5 * e[-1] @ Q_f @ e[-1]
                 + 0.5 * np.einsum('ti,ij,tj->', U, R, U))


def position_rmse(reference, X):
    d = np.asarray(reference)[:, :2] - np.asarray(X)[:, :2]
    return float(np.sqrt(np.mean(np.sum(d**2, axis=1))))


class LiftedILC:
    """
    reference : (N+1, 9) on the 35 Hz grid; U0 : (N, 2) trial-1 controls
    Q, Q_f, R : the task cost (weights.json), Q on samples 1..N-1, Q_f on N, R on u
    Qu_diag   : weight on the trial-to-trial step du (the quadruped's Qu)
    gain      : U <- U + gain du*
    model     : "linear" (default), "blend_linear" or "blend" -- what G is built from
    safeguard : a trial whose measured cost is above the best's x (1 + accept_tol) sends
                the next step from the best trial's controls and measurement with Qu x
                safeguard_growth; each improvement halves the scale back (backoff "qu")
    u_max     : |current slew rate|, |steer rate| limits (the iLQR forward pass's clip)
    """

    def __init__(self, reference, dt, U0, Q, Q_f, R, Qu_diag=(1e-4, 1.0), gain=1.0,
                 model="linear", safeguard=True, accept_tol=0.02, safeguard_growth=4.0,
                 qu_scale_max=256.0, car=None, u_max=(300.0, None), solver="osqp"):
        self.car = car or F110()
        self.reference = np.asarray(reference, float)
        self.dt = float(dt)
        self.N = len(U0)
        assert self.reference.shape == (self.N + 1, NX)
        self.U = np.asarray(U0, float).copy()
        self.Q, self.Q_f, self.R = (np.asarray(a, float) for a in (Q, Q_f, R))
        self.Qu = np.asarray(Qu_diag, float)
        self.gain = gain
        self.model = model
        self.safeguard = safeguard
        self.accept_tol = accept_tol
        self.safeguard_growth = safeguard_growth
        self.qu_scale_max = qu_scale_max
        self.qu_scale = 1.0
        self.u_max = np.array([u_max[0], u_max[1] or self.car['max_steer_vel']], float)
        self.solver = solver
        self.step_fn, self.jac_fn = make_model(model, self.car, self.dt)
        self.jac_map = self.jac_fn.map(self.N)
        self.trial = 0
        self.best = None
        self.history = []

    # -- model pieces ------------------------------------------------------------------
    def lifted_G(self, X, U):
        A, B = self.jac_map(X[:-1].T, U.T)
        A, B = np.asarray(A), np.asarray(B)              # (9, 9N), (9, 2N)
        A_list = [A[:, NX * t:NX * (t + 1)] for t in range(self.N)]
        B_list = [B[:, NU * t:NU * (t + 1)] for t in range(self.N)]
        return build_lifted_G(A_list, B_list)

    def rollout(self, x0, U):
        X = [np.asarray(x0, float)]
        for u in U:
            X.append(np.asarray(self.step_fn(X[-1], u)).ravel())
        return np.asarray(X)

    def _state_weights(self):
        W = np.tile(np.diag(self.Q), (self.N, 1))
        W[-1] = np.diag(self.Q_f)
        return W.reshape(-1)                              # rows of e_k[1:], time-major

    # -- the QP ------------------------------------------------------------------------
    def step(self, X, U, qu_scale=1.0):
        """du* (N, 2) from the trial (X, U): min |e - G du|^2_W + |du|^2_Qu + |U + du|^2_R
        s.t. rate limits, |steering command| <= max_steer, speed command >= 0."""
        N, dt, car = self.N, self.dt, self.car
        G = self.lifted_G(X, U)
        e = tracking_error(self.reference, X)[1:].reshape(-1)
        w = self._state_weights()
        r = np.tile(np.diag(self.R), N)
        s = np.tile(self.Qu * qu_scale, N)
        Uf = U.reshape(-1)
        H = G.T @ (w[:, None] * G) + np.diag(s + r)
        g = -G.T @ (w * e) + r * Uf

        # command-chain rows, exact (integrators the controller owns): steering command
        # after step t = steer_t + dt sum_{j<=t} du_d,j; speed command = double integral
        # of du_I, scaled by c
        c = car['gear_ratio'] * 1.5 * car['pole_pairs'] * car['lambda'] / (car['mass'] * car['rw'])
        L = np.tril(np.ones((N, N)))
        Sd = np.zeros((N, 2 * N)); Sd[:, 1::2] = dt * L
        SI = np.zeros((N, 2 * N)); SI[:, 0::2] = dt * L    # current after step t
        Sv = c * dt * L @ SI                               # speed command after step t
        chain = command_chain(U, X[0], car, dt)
        A_c = np.vstack([Sd, Sv])
        lba = np.r_[-car['max_steer'] - chain[:, 1], -chain[:, 2]]
        uba = np.r_[car['max_steer'] - chain[:, 1], np.full(N, np.inf)]
        umax = np.tile(self.u_max, N)

        qp = dict(h=ca.DM(H).sparsity(), a=ca.DM(A_c).sparsity())
        opts = dict(print_time=False, error_on_fail=False)
        if self.solver == "osqp":
            opts["osqp"] = dict(verbose=False, eps_abs=1e-8, eps_rel=1e-8, max_iter=40000,
                                polish=True)
        S = ca.conic('ilc', self.solver, qp, opts)
        sol = S(h=H, g=g, a=A_c, lba=lba, uba=uba, lbx=-umax - Uf, ubx=umax - Uf)
        stats = S.stats()
        du = np.asarray(sol['x']).reshape(N, NU)
        e_pred = e - G @ du.reshape(-1)
        info = dict(success=bool(stats.get('success', True)), status=str(stats.get('return_status', '')),
                    e_now=float(0.5 * w @ e**2), e_pred=float(0.5 * w @ e_pred**2),
                    du_max=np.abs(du).max(0).tolist())
        return du, info, G

    # -- per-trial update ----------------------------------------------------------------
    def update(self, X):
        """Score the trial that flew self.U (measured X) and set the next self.U."""
        X = np.asarray(X, float)
        U = self.U
        cost = trajectory_cost(X, U, self.reference, self.Q, self.Q_f, self.R)
        result = dict(trial=self.trial, cost=cost, position_rmse=position_rmse(self.reference, X))
        self.trial += 1
        is_best = self.best is None or cost <= self.best['cost'] * (1.0 + self.accept_tol)
        if is_best and (self.best is None or cost < self.best['cost']):
            self.best = dict(cost=cost, U=U.copy(), X=X.copy(), trial=result['trial'])
        if self.safeguard:
            if is_best:
                self.qu_scale = max(1.0, 0.5 * self.qu_scale)
            else:                                            # back to the best trial
                U, X = self.best['U'].copy(), self.best['X']
                self.qu_scale = min(self.qu_scale * self.safeguard_growth, self.qu_scale_max)
                result['rejected'] = True
            result['qu_scale'] = self.qu_scale
        du, info, _ = self.step(X, U, self.qu_scale)
        self.U = U + self.gain * du
        result['solver'] = info
        result['du_norm'] = float(np.linalg.norm(du))
        self.history.append(result)
        return result
