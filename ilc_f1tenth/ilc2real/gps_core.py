"""
gps_core.py (2026-11-06): mirror descent guided policy search (MDGPS; Montgomery & Levine, NeurIPS 2016, Algorithm 3 / Appendix B),
the GPS baseline of every figure that compares against FADA. Domain-agnostic numpy; the domain supplies per-iteration samples and
the simulator's linearizations. Defaults follow the authors' released code (github.com/cbfinn/gps): kl_step 0.2 per time step
(cons_per_step False: the total KL bound is kl_step * T), step multiplier in [0.01, 10], per-update multiplier clipped to [0.1, 5],
eta by geometric bracketing until |KL - eps| < 0.1 eps (<= 50 iterations), the 'laplace' step rule.

Deviations (paper/code -> here), all because every arm of the paper starts from the simulator (user 2026-11-06: "do b just like the
other ablation methods"):
- dynamics prior: the paper fits time-varying linear-Gaussian dynamics with a GMM prior over its own samples; here the prior is the
  SIMULATOR's linearization at the mean sample state (normal prior on [A B] worth n0 pseudo-samples per dimension), combined with the
  samples by Bayesian linear regression;
- the global policy's linearization is analytic (the paper allows "analytically differentiating the policy", Sec. 3.1);
- samples come from the global policy plus Gaussian action noise (the paper: "running either p_i or pi_theta").
"""
import numpy as np


def fit_dynamics(X, U, valid, A0, B0, n0, var_floor_x, var_floor_u):
    """X (S, T+1, dx), U (S, T, du), valid (S, T) bool; prior A0 (T, dx, dx), B0 (T, dx, du). Per t: Bayesian linear regression of
    x_{t+1} - mean on [x_t - mean, u_t - mean] toward [A0 B0] (prior precision n0 * diag(var floors)). Returns A, B, F (residual
    covariance, (T, dx, dx)), the mean trajectory xm (T+1, dx), um (T, du), and the number of samples used per t."""
    S, T1, dx = X.shape
    T, du = U.shape[1], U.shape[2]
    A, B = np.array(A0, float), np.array(B0, float)
    F = np.zeros((T, dx, dx))
    xm = np.zeros((T1, dx)); um = np.zeros((T, du)); ns = np.zeros(T, int)
    D = n0 * np.concatenate([var_floor_x, var_floor_u])
    for t in range(T):
        m = valid[:, t]
        ns[t] = m.sum()
        if m.sum() == 0:
            xm[t] = X[0, t]; um[t] = U[0, t]; F[t] = np.diag(var_floor_x)
            continue
        xm[t] = X[m, t].mean(0); um[t] = U[m, t].mean(0); xm[t + 1] = X[m, t + 1].mean(0)
        Z = np.concatenate([X[m, t] - xm[t], U[m, t] - um[t]], 1)
        Y = X[m, t + 1] - xm[t + 1]
        P = np.concatenate([A0[t], B0[t]], 1)
        Dz = np.diag(np.maximum(D, 1e-12))
        Th = (Y.T @ Z + P @ Dz) @ np.linalg.inv(Z.T @ Z + Dz)
        A[t], B[t] = Th[:, :dx], Th[:, dx:]
        R = Y - Z @ Th.T
        F[t] = (R.T @ R + n0 * np.diag(var_floor_x)) / (m.sum() + n0)
    xm[T] = X[valid[:, -1], T].mean(0) if valid[:, -1].any() else xm[T - 1]
    return A, B, F, xm, um, ns


def lqr_maxent(A, B, Cxx, cx, Kp, kp, Sp_inv, eta, lu_reg=1e-6):
    """Max-entropy LQR (Appendix B.1) in deviation coordinates on the surrogate cost (1/eta) l(x, u) - log pibar(u | x), with
    l = 0.5 x'Cxx x + cx'x (per t, no action cost) and pibar = N(Kp x + kp, Sp). Returns K (T, du, dx), k (T, du), Quu (T, du, du)
    (the local controller p = N(K x + k, Quu^-1)), or None if Quu is not positive definite."""
    T, dx, du = B.shape
    K = np.zeros((T, du, dx)); k = np.zeros((T, du)); Quu_all = np.zeros((T, du, du))
    Vxx = np.zeros((dx, dx)); vx = np.zeros(dx)
    for t in reversed(range(T)):
        # -log pibar = 0.5 (u - Kp x - kp)' Sp_inv (u - Kp x - kp)
        lxx = Cxx[t] / eta + Kp[t].T @ Sp_inv @ Kp[t]
        luu = Sp_inv + lu_reg * np.eye(du)
        lux = -Sp_inv @ Kp[t]
        lx = cx[t] / eta + Kp[t].T @ Sp_inv @ kp[t]
        lu = -Sp_inv @ kp[t]
        Qxx = lxx + A[t].T @ Vxx @ A[t]
        Quu = luu + B[t].T @ Vxx @ B[t]
        Qux = lux + B[t].T @ Vxx @ A[t]
        qx = lx + A[t].T @ vx
        qu = lu + B[t].T @ vx
        Quu = 0.5 * (Quu + Quu.T)
        try:
            L = np.linalg.cholesky(Quu)
        except np.linalg.LinAlgError:
            return None
        Kt = -np.linalg.solve(Quu, Qux)
        kt = -np.linalg.solve(Quu, qu)
        K[t], k[t], Quu_all[t] = Kt, kt, Quu
        Vxx = Qxx + Qux.T @ Kt
        Vxx = 0.5 * (Vxx + Vxx.T)
        vx = qx + Qux.T @ kt
    return K, k, Quu_all


def propagate(A, B, F, S0, K, k, Su):
    """Gaussian forward pass (Appendix B.3): state marginals (mu (T+1, dx), Sigma (T+1, dx, dx)) of x_{t+1} = A x + B u, u ~ N(K x + k, Su),
    deviation coordinates, x_0 ~ N(0, S0)."""
    T, dx, du = B.shape
    mu = np.zeros((T + 1, dx)); Sg = np.zeros((T + 1, dx, dx)); Sg[0] = S0
    for t in range(T):
        G = A[t] + B[t] @ K[t]
        mu[t + 1] = G @ mu[t] + B[t] @ k[t]
        Sg[t + 1] = G @ Sg[t] @ G.T + B[t] @ Su[t] @ B[t].T + F[t]
    return mu, Sg


def expected_cost(mu, Sg, Cxx, cx, c0):
    """E[sum_t l_t] under the state marginals, l_t = 0.5 x'Cxx x + cx'x + c0."""
    T = len(Cxx)
    return float(sum(0.5 * (mu[t] @ Cxx[t] @ mu[t] + np.trace(Cxx[t] @ Sg[t])) + cx[t] @ mu[t] + c0[t] for t in range(T)))


def kl_traj(mu, Sg, K, k, Su, Kp, kp, Sp, Sp_inv):
    """sum_t E_{x ~ p}[ KL( N(K x + k, Su) || N(Kp x + kp, Sp) ) ] (the trajectory KL of Eq. 3 / Appendix A)."""
    T, du = k.shape
    ld_p = np.linalg.slogdet(Sp)[1]
    tot = 0.0
    for t in range(T):
        dK, dk = K[t] - Kp[t], k[t] - kp[t]
        dm = dK @ mu[t] + dk
        tot += 0.5 * (np.trace(Sp_inv @ Su[t]) - du + ld_p - np.linalg.slogdet(Su[t])[1]
                      + dm @ Sp_inv @ dm + np.trace(dK.T @ Sp_inv @ dK @ Sg[t]))
    return float(tot)


def c_step(A, B, F, S0, Cxx, cx, c0, Kp, kp, Sp, eps, min_eta=1e-8, max_eta=1e16, eta0=1.0, max_iter=50):
    """The constrained C-step (Algorithm 3 line 5; Appendix B.1): eta by the released code's geometric bracketing until
    |KL - eps| < 0.1 eps. Returns dict(K, k, Quu, Su, eta, kl, cost (expected under the fitted dynamics), iters)."""
    Sp_inv = np.linalg.inv(Sp)
    eta, lo, hi = eta0, min_eta, max_eta
    best = None
    for it in range(max_iter):
        sol = lqr_maxent(A, B, Cxx, cx, Kp, kp, Sp_inv, eta)
        if sol is None:                               # not positive definite: larger eta (the code: eta + del, doubling)
            lo = eta; eta = min(eta * 10.0, max_eta); continue
        K, k, Quu = sol
        Su = np.linalg.inv(Quu)
        mu, Sg = propagate(A, B, F, S0, K, k, Su)
        kl = kl_traj(mu, Sg, K, k, Su, Kp, kp, Sp, Sp_inv)
        con = kl - eps
        best = dict(K=K, k=k, Quu=Quu, Su=Su, eta=eta, kl=kl, cost=expected_cost(mu, Sg, Cxx, cx, c0), iters=it + 1)
        if abs(con) < 0.1 * eps:
            break
        if con < 0:                                   # eta too big: the step is smaller than allowed
            hi = eta; eta = max(np.sqrt(lo * hi), 0.1 * hi)
        else:                                         # eta too small
            lo = eta; eta = min(np.sqrt(lo * hi), 10.0 * lo)
    return best


def policy_cost(A, B, F, S0, Kp, kp, Sp, Cxx, cx, c0):
    """The expected cost of the global policy's linearization under fitted dynamics (the 'laplace' estimate)."""
    T, du = kp.shape
    Su = np.repeat(Sp[None], T, 0)
    mu, Sg = propagate(A, B, F, S0, Kp, kp, Su)
    return expected_cost(mu, Sg, Cxx, cx, c0)


def new_step_mult(step_mult, prev_laplace, prev_predicted, cur_laplace, min_step_mult=0.01, max_step_mult=10.0):
    """The released code's step adjustment ('laplace' rule; algorithm.py _set_new_mult)."""
    predicted_impr = prev_laplace - prev_predicted
    actual_impr = prev_laplace - cur_laplace
    new_mult = predicted_impr / (2.0 * max(1e-4, predicted_impr - actual_impr))
    new_mult = max(0.1, min(5.0, new_mult))
    return max(min(new_mult * step_mult, max_step_mult), min_step_mult)
