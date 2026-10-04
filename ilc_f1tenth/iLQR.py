"""
iLQR on the linearized Fiala (brush, combined-slip) dynamics, formulated like LLA-MPC's acados NMPC
(llampc/nmpc_gen_fiala_fixed.py):

  model        dynamics.continous_dynamics_model (the same Fiala model as nmpc_gen_fiala_fixed.export_model):
               x = [x, y, phi, vx, vy, omega, omega_w, current, delta], u = [current slew rate, steer rate],
               p = [Cf, Cr, muf, mur, Cro]
  integrator   explicit RK4 with N_SUB substeps per dt (acados: ERK, 4 stages, sim_method_num_steps substeps)
  linearized   A_k = d x_{k+1} / d x_k, B_k = d x_{k+1} / d u_k of that discrete map, at the current trajectory
               (what acados' SQP linearizes each iteration)
  cost         acados NONLINEAR_LS, 0.5 ||y||_W^2 per stage (scaled by dt, as acados scales stage costs), with
               y = [x - x_ref, y - y_ref, wrap(phi - phi_ref), vx - vx_ref, vy - vy_ref, omega - omega_ref,
                    current, delta, u]
               and the terminal y_e (no u); x_ref = the first 6 columns of the reference
  constraints  hard box on u (lbu/ubu), solved as a box QP in the backward pass (control-limited DDP);
               soft box on [vx, vy, omega, current, delta] (lbx/ubx with acados' L1 + L2 slack weights), as the
               exact penalty those slacks amount to
  regularized  Levenberg-Marquardt on Q_uu (acados levenberg_marquardt), increased when Q_uu is not positive
               definite

No STATE_MASK: the controls act only through current and delta (d current/dt = u0, d delta/dt = u1), so masking
those states removed the controls' effect. The masking was hiding the stiff wheel-speed mode (eigenvalue
~ -200 /s at 3 m/s, ~ -600 /s at 1 m/s), which a single RK4 step of 0.05 s cannot integrate stably; the substeps
do (stable for dt / N_SUB below ~2.8 / |eigenvalue|).

API (as before): backward(...) -> k, K, cost; forward(...) -> controls, trajectory, cost;
alpha_search(...) -> alpha, controls, trajectory, cost; plus solve(...), a full iLQR loop on the model.
"""
import casadi as ca
import numpy as np

from dynamics import continous_dynamics_model

# integrator substeps per dt (acados used 5 at dt = 0.05; 10 keeps RK4 stable down to ~1 m/s)
N_SUB = 10

# cost weights: llampc's "FAST" set
COST = dict(w_x=20.0, w_y=20.0, w_theta=0.0, w_vx=0.01, w_vy=0.0, w_omega=1.0, w_current=0.01, w_steer=0.5,
            w_slew=0.0, w_steer_v=0.5, w_xe=0.0, w_ye=0.0)

# soft state bounds on [vx, vy, omega, current, delta] (llampc idxbx), slack weights (L2, L1)
X_BOUND_IDX = np.array([3, 4, 5, 7, 8])
SLACK_L2, SLACK_L1 = 100.0, 10.0
# input bounds: |current slew rate| <= 300 A/s, |steer rate| <= max_steer_vel
SLEW_MAX = 300.0

LM = 1e-4            # Levenberg-Marquardt on Q_uu (acados levenberg_marquardt)
LM_MAX = 1e6
alpha_values = [1.0, 0.5, 0.25, 0.125, 0.0625]

# y = C x + (yaw wrapped): rows pick x, y, phi, vx, vy, omega, current, delta
_C = np.zeros((8, 9))
for _r, _c in enumerate([0, 1, 2, 3, 4, 5, 7, 8]):
    _C[_r, _c] = 1.0


def _weights():
    W = np.diag([COST["w_x"], COST["w_y"], COST["w_theta"], COST["w_vx"], COST["w_vy"], COST["w_omega"],
                 COST["w_current"], COST["w_steer"]])
    Wu = np.diag([COST["w_slew"], COST["w_steer_v"]])
    We = np.diag([COST["w_xe"], COST["w_ye"], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return W, Wu, We


def bounds(params_car):
    """(u_lb, u_ub, x_lb, x_ub) as in llampc's create_ocp"""
    u_lb = np.array([-SLEW_MAX, -params_car['max_steer_vel']])
    u_ub = np.array([SLEW_MAX, params_car['max_steer_vel']])
    x_lb = np.array([-0.5, -4.0, -2 * np.pi, -25.0, params_car['min_steer']])
    x_ub = np.array([params_car['max_v'], 4.0, 2 * np.pi, 50.0, params_car['max_steer']])
    return u_lb, u_ub, x_lb, x_ub


_FUNCS = {}


def functions(params_car, exact, dt):
    """The discrete Fiala map (RK4, N_SUB substeps) and its Jacobians, built once per (car, exact, dt)."""
    key = (id(params_car), bool(exact), round(float(dt), 9), N_SUB)
    if key not in _FUNCS:
        x = ca.SX.sym('x', 9)
        u = ca.SX.sym('u', 2)
        p = ca.SX.sym('p', 5)
        h = dt / N_SUB
        f = lambda x_: continous_dynamics_model(x_, u, p, params_car, exact)
        xn = x
        for _ in range(N_SUB):
            k1 = f(xn)
            k2 = f(xn + h / 2 * k1)
            k3 = f(xn + h / 2 * k2)
            k4 = f(xn + h * k3)
            xn = xn + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        step = ca.Function("fiala_step", [x, u, p], [xn])
        jac = ca.Function("fiala_jac", [x, u, p], [ca.jacobian(xn, x), ca.jacobian(xn, u)])
        _FUNCS[key] = (step, jac)
    return _FUNCS[key]


def _residual(x, xr):
    y = _C @ x
    y[:6] -= xr[:6]
    y[2] = np.arctan2(np.sin(y[2]), np.cos(y[2]))       # wrapped yaw error
    return y


def _slack(x, x_lb, x_ub):
    """the soft state bounds' exact penalty: value, gradient, Hessian (in x)"""
    xs = x[X_BOUND_IDX]
    over, under = np.maximum(xs - x_ub, 0.0), np.maximum(x_lb - xs, 0.0)
    v = over + under
    val = float(np.sum(0.5 * SLACK_L2 * v ** 2 + SLACK_L1 * v))
    g = np.zeros(9)
    H = np.zeros((9, 9))
    act = v > 0
    sgn = np.where(over > 0, 1.0, -1.0)
    g[X_BOUND_IDX] = np.where(act, sgn * (SLACK_L2 * v + SLACK_L1), 0.0)
    H[X_BOUND_IDX, X_BOUND_IDX] = np.where(act, SLACK_L2, 0.0)
    return val, g, H


def stage_cost(x, u, xr, dt, params_car):
    W, Wu, _ = _weights()
    _, _, x_lb, x_ub = bounds(params_car)
    y = _residual(x, xr)
    return dt * (0.5 * y @ W @ y + 0.5 * u @ Wu @ u + _slack(x, x_lb, x_ub)[0])


def terminal_cost(x, xr, params_car):
    _, _, We = _weights()
    _, _, x_lb, x_ub = bounds(params_car)
    y = _residual(x, xr)
    return 0.5 * y @ We @ y + _slack(x, x_lb, x_ub)[0]


def trajectory_cost(ref_trajectory, trajectory, controls, dt, params_car):
    c = sum(stage_cost(trajectory[i], controls[i], ref_trajectory[i], dt, params_car)
            for i in range(controls.shape[0]))
    return c + terminal_cost(trajectory[-1], ref_trajectory[-1], params_car)


def box_qp(H, g, lb, ub, x0=None, iters=50, tol=1e-10):
    """min 0.5 d'Hd + g'd, lb <= d <= ub (H positive definite), by projected Newton (Tassa et al. 2014).
    Returns d and the free set."""
    n = g.shape[0]
    d = np.clip(np.zeros(n) if x0 is None else x0, lb, ub)
    free = np.ones(n, bool)
    for _ in range(iters):
        grad = g + H @ d
        clamped = ((d <= lb + 1e-12) & (grad > 0)) | ((d >= ub - 1e-12) & (grad < 0))
        free = ~clamped
        if not free.any():
            break
        Hf = H[np.ix_(free, free)]
        step = np.zeros(n)
        step[free] = -np.linalg.solve(Hf, grad[free])
        if np.linalg.norm(step) < tol:
            break
        q = lambda z: 0.5 * z @ H @ z + g @ z      # Armijo line search on the projected step
        a, f0 = 1.0, q(d)
        while a > 1e-6:
            dn = np.clip(d + a * step, lb, ub)
            if q(dn) <= f0 + 0.1 * grad @ (dn - d):
                break
            a *= 0.5
        if np.linalg.norm(dn - d) < tol:
            d = dn
            break
        d = dn
    return d, free


def backward(ref_trajectory, cur_trajectory, cur_controls, p, params_car, exact, dt, mu=LM):
    """One backward pass along (cur_trajectory, cur_controls): the feedforward k (box-constrained) and gains K,
    and the trajectory's cost. Raises mu (Levenberg-Marquardt) until every Q_uu is positive definite."""
    _, jac = functions(params_car, exact, dt)
    W, Wu, We = _weights()
    u_lb, u_ub, x_lb, x_ub = bounds(params_car)
    N = cur_controls.shape[0]
    AB = [jac(cur_trajectory[i], cur_controls[i], p) for i in range(N)]
    AB = [(np.array(A), np.array(B)) for A, B in AB]
    cost = trajectory_cost(ref_trajectory, cur_trajectory, cur_controls, dt, params_car)
    while True:
        y = _residual(cur_trajectory[-1], ref_trajectory[-1])
        _, gs, Hs = _slack(cur_trajectory[-1], x_lb, x_ub)
        V_x = _C.T @ We @ y + gs
        V_xx = _C.T @ We @ _C + Hs
        k_list, K_list, ok = [], [], True
        for i in reversed(range(N)):
            x, u = cur_trajectory[i], cur_controls[i]
            A, B = AB[i]
            y = _residual(x, ref_trajectory[i])
            _, gs, Hs = _slack(x, x_lb, x_ub)
            # Gauss-Newton stage cost derivatives
            l_x = dt * (_C.T @ W @ y + gs)
            l_xx = dt * (_C.T @ W @ _C + Hs)
            l_u = dt * (Wu @ u)
            l_uu = dt * Wu
            Q_x = l_x + A.T @ V_x
            Q_u = l_u + B.T @ V_x
            Q_xx = l_xx + A.T @ V_xx @ A
            Q_uu = l_uu + B.T @ V_xx @ B
            Q_ux = B.T @ V_xx @ A
            Q_uu_r = Q_uu + mu * np.eye(2)
            try:
                np.linalg.cholesky(Q_uu_r)
            except np.linalg.LinAlgError:
                ok = False
                break
            # feedforward: the box QP on du with u + du within the input bounds
            k, free = box_qp(Q_uu_r, Q_u, u_lb - u, u_ub - u)
            K = np.zeros((2, 9))
            if free.any():                              # no feedback on clamped inputs
                K[free] = -np.linalg.solve(Q_uu_r[np.ix_(free, free)], Q_ux[free])
            V_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
            V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
            V_xx = 0.5 * (V_xx + V_xx.T)
            k_list.append(k)
            K_list.append(K)
        if ok:
            break
        mu *= 10.0
        if mu > LM_MAX:
            raise RuntimeError("iLQR backward: Q_uu not positive definite even with mu = %g" % LM_MAX)
    backward.mu = mu
    k_list.reverse()
    K_list.reverse()
    return np.array(k_list), np.array(K_list), cost


def _dx(x_new, x):
    d = x_new - x
    d[2] = np.arctan2(np.sin(d[2]), np.cos(d[2]))
    return d


def forward(k_list, K_list, ref_trajectory, cur_trajectory, cur_controls, p, params_car, exact, dt, alpha):
    """Roll the model out under u = clip(u_bar + alpha k + K (x - x_bar)) from cur_trajectory[0]."""
    step, _ = functions(params_car, exact, dt)
    u_lb, u_ub, _, _ = bounds(params_car)
    x_new = np.array(cur_trajectory[0], float)
    traj, ctrl = [x_new], []
    for i in range(cur_controls.shape[0]):
        u_new = np.clip(cur_controls[i] + alpha * k_list[i] + K_list[i] @ _dx(x_new, cur_trajectory[i]), u_lb, u_ub)
        x_new = np.array(step(x_new, u_new, p)).reshape(-1)
        ctrl.append(u_new)
        traj.append(x_new)
    traj, ctrl = np.array(traj), np.array(ctrl)
    if not np.all(np.isfinite(traj)):
        return ctrl, traj, np.inf
    return ctrl, traj, trajectory_cost(ref_trajectory, traj, ctrl, dt, params_car)


def alpha_search(k_list, K_list, ref_trajectory, cur_trajectory, cur_controls, cur_cost, p, params_car, exact, dt):
    """The largest alpha in alpha_values whose rollout lowers the cost: (alpha, controls, trajectory, cost);
    alpha = 0 (and the current controls) if none does."""
    for alpha in alpha_values:
        ctrl, traj, c = forward(k_list, K_list, ref_trajectory, cur_trajectory, cur_controls, p, params_car,
                                exact, dt, alpha)
        if c < cur_cost:
            return alpha, ctrl, traj, c
    return 0.0, cur_controls, cur_trajectory, cur_cost


def rollout(x0, controls, p, params_car, exact, dt):
    step, _ = functions(params_car, exact, dt)
    traj = [np.array(x0, float)]
    for u in controls:
        traj.append(np.array(step(traj[-1], u, p)).reshape(-1))
    return np.array(traj)


def solve(x0, ref_trajectory, controls, p, params_car, exact, dt, iters=50, tol=1e-6, verbose=False):
    """iLQR on the model from x0: Levenberg-Marquardt schedule on mu (down after a step, up after a failed one).
    Returns (controls, trajectory, cost history)."""
    u_lb, u_ub, _, _ = bounds(params_car)
    controls = np.clip(np.array(controls, float), u_lb, u_ub)
    traj = rollout(x0, controls, p, params_car, exact, dt)
    mu, hist = LM, []
    for it in range(iters):
        k, K, cost = backward(ref_trajectory, traj, controls, p, params_car, exact, dt, mu=mu)
        mu = backward.mu
        hist.append(cost)
        alpha, ctrl, tr, c = alpha_search(k, K, ref_trajectory, traj, controls, cost, p, params_car, exact, dt)
        if verbose:
            print(f"iLQR it {it}: cost {cost:.6g} -> {c:.6g} (alpha {alpha}, mu {mu:.1e})")
        if alpha == 0.0:
            mu *= 10.0
            if mu > LM_MAX:
                break
            continue
        controls, traj = ctrl, tr
        mu = max(mu / 10.0, 1e-8)
        if (cost - c) < tol * max(1.0, abs(cost)):
            hist.append(c)
            break
    return controls, traj, hist
