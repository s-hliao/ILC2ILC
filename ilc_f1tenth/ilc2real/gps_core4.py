"""
gps_core4.py (2026-11-07 third pass: the global policy's linearization FITTED as the released code -- PolicyPriorGMM (GMM over
[x, pi(x)] of the last max_samples trajectories, mode 'add') + gauss_fit_joint_prior per t + the policy covariance -- so the policy's
covariance pol_S is time-varying: every function takes Sp as (T, du, du); gps_core3.py otherwise)
gps_core3.py (2026-11-07 second pass: c_step as traj_opt_lqr_python.update -- eta0 = the previous iteration's eta, a non-PD backward
pass raises eta by del (del0 1e-4, doubling) as backward(); gps_core2.py otherwise)
gps_core2.py (2026-11-07: gps_core.py with the residual-covariance prior fixed, see fit_dynamics; gps_core.py 2026-11-06): mirror descent guided policy search (MDGPS; Montgomery & Levine, NeurIPS 2016, Algorithm 3 / Appendix B),
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


def as_tv(S, T):
    """A covariance (du, du) or (T, du, du) -> (T, du, du)."""
    S = np.asarray(S, float)
    return np.repeat(S[None], T, 0) if S.ndim == 2 else S


def fit_dynamics(X, U, valid, A0, B0, n0, var_floor_x, var_floor_u, Fprior):
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
            xm[t] = X[0, t]; um[t] = U[0, t]; F[t] = Fprior
            continue
        xm[t] = X[m, t].mean(0); um[t] = U[m, t].mean(0); xm[t + 1] = X[m, t + 1].mean(0)
        Z = np.concatenate([X[m, t] - xm[t], U[m, t] - um[t]], 1)
        Y = X[m, t + 1] - xm[t + 1]
        P = np.concatenate([A0[t], B0[t]], 1)
        Dz = np.diag(np.maximum(D, 1e-12))
        Th = (Y.T @ Z + P @ Dz) @ np.linalg.inv(Z.T @ Z + Dz)
        A[t], B[t] = Th[:, :dx], Th[:, dx:]
        R = Y - Z @ Th.T
        # (2026-11-07 fix) the residual covariance's prior is the SIMULATOR model's one-step residual covariance pooled over the
        # samples (Fprior; the code's NIW prior Phi gives a pooled conditional covariance), not the across-sample state spread
        # (var_floor_x), which inflated the per-step noise ~100x (positions) and made the laplace estimate variance-dominated
        F[t] = (R.T @ R + n0 * Fprior) / (m.sum() + n0)
    xm[T] = X[valid[:, -1], T].mean(0) if valid[:, -1].any() else xm[T - 1]
    return A, B, F, xm, um, ns


def lqr_maxent(A, B, Cxx, cx, Kp, kp, Sp_inv_t, eta, lu_reg=1e-6):
    """Max-entropy LQR (Appendix B.1) in deviation coordinates on the surrogate cost (1/eta) l(x, u) - log pibar(u | x), with
    l = 0.5 x'Cxx x + cx'x (per t, no action cost) and pibar = N(Kp x + kp, Sp). Returns K (T, du, dx), k (T, du), Quu (T, du, du)
    (the local controller p = N(K x + k, Quu^-1)), or None if Quu is not positive definite."""
    T, dx, du = B.shape
    K = np.zeros((T, du, dx)); k = np.zeros((T, du)); Quu_all = np.zeros((T, du, du))
    Vxx = np.zeros((dx, dx)); vx = np.zeros(dx)
    for t in reversed(range(T)):
        Sp_inv = Sp_inv_t[t]
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


def kl_traj(mu, Sg, K, k, Su, Kp, kp, Sp_t, Sp_inv_t):
    """sum_t E_{x ~ p}[ KL( N(K x + k, Su) || N(Kp x + kp, Sp) ) ] (the trajectory KL of Eq. 3 / Appendix A)."""
    T, du = k.shape
    tot = 0.0
    for t in range(T):
        Sp, Sp_inv = Sp_t[t], Sp_inv_t[t]
        ld_p = np.linalg.slogdet(Sp)[1]
        dK, dk = K[t] - Kp[t], k[t] - kp[t]
        dm = dK @ mu[t] + dk
        tot += 0.5 * (np.trace(Sp_inv @ Su[t]) - du + ld_p - np.linalg.slogdet(Su[t])[1]
                      + dm @ Sp_inv @ dm + np.trace(dK.T @ Sp_inv @ dK @ Sg[t]))
    return float(tot)


def c_step(A, B, F, S0, Cxx, cx, c0, Kp, kp, Sp, eps, min_eta=1e-8, max_eta=1e16, eta0=1.0, max_iter=50, del0=1e-4):
    """The constrained C-step (traj_opt_lqr_python.update, cons_per_step False): eta starts at eta0 (the code: the previous iteration's
    eta, cur.eta = prev.eta); each backward pass that hits a non-PD Quu raises eta by del (del0 doubling: backward(): eta = eta0 + del_,
    del_ *= 2), up to max_eta; the bracketing on the KL constraint (con < 0: max_eta = eta, eta = max(sqrt(min max), 0.1 max); else
    min_eta = eta, eta = min(sqrt(min max), 10 min)) until |KL - eps| < 0.1 eps. Returns dict(K, k, Quu, Su, eta, kl, cost, iters)."""
    Sp = as_tv(Sp, len(B))
    Sp_inv = np.linalg.inv(Sp)
    eta, lo, hi = float(eta0), float(min_eta), float(max_eta)
    best = None
    for it in range(max_iter):
        e_try, del_ = eta, del0
        sol = lqr_maxent(A, B, Cxx, cx, Kp, kp, Sp_inv, e_try)
        while sol is None:                            # backward(): non-PD -> eta = eta0 + del_, del_ doubling
            e_try = eta + del_
            del_ *= 2
            if e_try >= 1e16:
                return best
            sol = lqr_maxent(A, B, Cxx, cx, Kp, kp, Sp_inv, e_try)
        eta = e_try
        K, k, Quu = sol
        Su = np.linalg.inv(Quu)
        mu, Sg = propagate(A, B, F, S0, K, k, Su)
        kl = kl_traj(mu, Sg, K, k, Su, Kp, kp, Sp, Sp_inv)
        con = kl - eps
        best = dict(K=K, k=k, Quu=Quu, Su=Su, eta=eta, kl=kl, cost=expected_cost(mu, Sg, Cxx, cx, c0), iters=it + 1)
        if abs(con) < 0.1 * eps:
            break
        if con < 0:                                   # eta too big
            hi = eta; eta = max(np.sqrt(lo * hi), 0.1 * hi)
        else:                                         # eta too small
            lo = eta; eta = min(np.sqrt(lo * hi), 10.0 * lo)
    return best


def policy_cost(A, B, F, S0, Kp, kp, Sp, Cxx, cx, c0):
    """The expected cost of the global policy's linearization under fitted dynamics (the 'laplace' estimate)."""
    T, du = kp.shape
    Su = as_tv(Sp, T)
    mu, Sg = propagate(A, B, F, S0, Kp, kp, Su)
    return expected_cost(mu, Sg, Cxx, cx, c0)


def new_step_mult(step_mult, prev_laplace, prev_predicted, cur_laplace, min_step_mult=0.01, max_step_mult=10.0):
    """The released code's step adjustment ('laplace' rule; algorithm.py _set_new_mult)."""
    predicted_impr = prev_laplace - prev_predicted
    actual_impr = prev_laplace - cur_laplace
    new_mult = predicted_impr / (2.0 * max(1e-4, predicted_impr - actual_impr))
    new_mult = max(0.1, min(5.0, new_mult))
    return max(min(new_mult * step_mult, max_step_mult), min_step_mult)


# ---------------- the released code's GMM (gps/utility/gmm.py, verbatim port) ----------------
def logsum(vec, axis=0, keepdims=True):
    maxv = np.max(vec, axis=axis, keepdims=keepdims)
    maxv[maxv == -float('inf')] = 0
    return np.log(np.sum(np.exp(vec - maxv), axis=axis, keepdims=keepdims)) + maxv


class GMM(object):
    """gps.utility.gmm.GMM (init_sequential False, eigreg False, warmstart True); np.random as the code (seeded by the caller)."""
    def __init__(self, warmstart=True):
        self.warmstart = warmstart
        self.sigma = None

    def inference(self, pts):
        logwts = self.clusterwts(pts)
        mu0, Phi = self.moments(logwts)
        m = self.N
        n0 = m - 2 - mu0.shape[0]
        m = float(m) / self.N
        n0 = float(n0) / self.N
        return mu0, Phi, m, n0

    def estep(self, data):
        import scipy.linalg
        N, D = data.shape
        K = self.sigma.shape[0]
        logobs = -0.5 * np.ones((N, K)) * D * np.log(2 * np.pi)
        for i in range(K):
            mu, sigma = self.mu[i], self.sigma[i]
            L = scipy.linalg.cholesky(sigma, lower=True)
            logobs[:, i] -= np.sum(np.log(np.diag(L)))
            diff = (data - mu).T
            soln = scipy.linalg.solve_triangular(L, diff, lower=True)
            logobs[:, i] -= 0.5 * np.sum(soln ** 2, axis=0)
        logobs += self.logmass.T
        return logobs

    def moments(self, logwts):
        wts = np.exp(logwts)
        mu = np.sum(self.mu * wts, axis=0)
        diff = self.mu - np.expand_dims(mu, axis=0)
        diff_expand = np.expand_dims(self.mu, axis=1) * np.expand_dims(diff, axis=2)
        wts_expand = np.expand_dims(wts, axis=2)
        sigma = np.sum((self.sigma + diff_expand) * wts_expand, axis=0)
        return mu, sigma

    def clusterwts(self, data):
        logobs = self.estep(data)
        logwts = logobs - logsum(logobs, axis=1)
        logwts = logsum(logwts, axis=0) - np.log(data.shape[0])
        return logwts.T

    def update(self, data, K, max_iterations=100):
        N = data.shape[0]
        Do = data.shape[1]
        if not self.warmstart or self.sigma is None or K != self.sigma.shape[0]:
            self.sigma = np.zeros((K, Do, Do))
            self.mu = np.zeros((K, Do))
            self.logmass = np.log(1.0 / K) * np.ones((K, 1))
            self.mass = (1.0 / K) * np.ones((K, 1))
            self.N = data.shape[0]
            N = self.N
            cidx = np.random.randint(0, K, size=(1, N))
            for i in range(K):
                cluster_idx = (cidx == i)[0]
                mu = np.mean(data[cluster_idx, :], axis=0)
                diff = (data[cluster_idx, :] - mu).T
                sigma = (1.0 / K) * (diff.dot(diff.T))
                self.mu[i, :] = mu
                self.sigma[i, :, :] = sigma + np.eye(Do) * 2e-6
        prevll = -float('inf')
        for itr in range(max_iterations):
            logobs = self.estep(data)
            ll = np.sum(logsum(logobs, axis=1))
            if ll < prevll:
                break
            if np.abs(ll - prevll) < 1e-5 * prevll:
                break
            prevll = ll
            logw = logobs - logsum(logobs, axis=1)
            logwn = logw - logsum(logw, axis=0)
            w = np.exp(logwn)
            self.logmass = logsum(logw, axis=0).T
            self.logmass = self.logmass - logsum(self.logmass, axis=0)
            self.mass = np.exp(self.logmass)
            w[:, (self.mass < (1.0 / K) * 1e-4)[:, 0]] = 1.0 / N
            w_expand = np.expand_dims(w, axis=2)
            data_expand = np.expand_dims(data, axis=1)
            self.mu = np.sum(w_expand * data_expand, axis=0)
            wdata = data_expand * np.sqrt(w_expand)
            for i in range(K):
                XX = wdata[:, i, :].T.dot(wdata[:, i, :])
                mu = self.mu[i, :]
                self.sigma[i, :, :] = XX - np.outer(mu, mu)
                sigma = self.sigma[i, :, :]
                self.sigma[i, :, :] = 0.5 * (sigma + sigma.T) + 1e-6 * np.eye(Do)


def gauss_fit_joint_prior(pts, mu0, Phi, m, n0, dwts, dX, dU, sig_reg):
    """gps.algorithm.algorithm_utils.gauss_fit_joint_prior, verbatim."""
    D = np.diag(dwts)
    mun = np.sum((pts.T * dwts).T, axis=0)
    diff = pts - mun
    empsig = diff.T.dot(D).dot(diff)
    empsig = 0.5 * (empsig + empsig.T)
    N = dwts.shape[0]
    mu = mun
    sigma = (N * empsig + Phi + (N * m) / (N + m) * np.outer(mun - mu0, mun - mu0)) / (N + n0)
    sigma = 0.5 * (sigma + sigma.T)
    sigma += sig_reg
    fd = np.linalg.solve(sigma[:dX, :dX], sigma[:dX, dX:dX + dU]).T
    fc = mu[dX:dX + dU] - fd.dot(mu[:dX])
    dynsig = sigma[dX:dX + dU, dX:dX + dU] - fd.dot(sigma[:dX, :dX]).dot(fd.T)
    dynsig = 0.5 * (dynsig + dynsig.T)
    return fd, fc, dynsig


class PolicyPriorGMM(object):
    """gps.algorithm.policy.policy_prior_gmm.PolicyPriorGMM with POLICY_PRIOR_GMM defaults (min_samples_per_cluster 20, max_clusters 50,
    max_samples 20, strength 1.0), mode 'add' (ALG_MDGPS policy_sample_mode). Deviation (declared): the samples have valid masks of
    different lengths (a car trial ends at a spin-out) -- only valid (x, u) rows enter the GMM, and fit() regresses each t on the samples
    valid at t (dwts = 1 / N_t); the code assumes equal-length samples."""
    def __init__(self, min_samp=20, max_samples=20, max_clusters=50, strength=1.0):
        self.X = self.obs = self.V = None
        self.gmm = GMM()
        self._min_samp, self._max_samples, self._max_clusters, self._strength = min_samp, max_samples, max_clusters, strength

    def update(self, X, obs, V, policy_mu_fn, mode='add'):
        """X (N, T, dX), obs (N, T, dO), V (N, T) valid; policy_mu_fn(obs (M, dO)) -> (M, dU) the CURRENT policy's mean."""
        if self.X is None or mode == 'replace':
            self.X, self.obs, self.V = X, obs, V
        elif mode == 'add' and X.size > 0:
            T = max(self.X.shape[1], X.shape[1])
            pad = lambda Z, T: np.concatenate([Z, np.repeat(Z[:, -1:], T - Z.shape[1], 1)], 1) if Z.shape[1] < T else Z
            padv = lambda Z, T: np.concatenate([Z, np.zeros((Z.shape[0], T - Z.shape[1]), bool)], 1) if Z.shape[1] < T else Z
            self.X = np.concatenate([pad(self.X, T), pad(X, T)], 0)
            self.obs = np.concatenate([pad(self.obs, T), pad(obs, T)], 0)
            self.V = np.concatenate([padv(self.V, T), padv(V, T)], 0)
            N = self.X.shape[0]
            if N > self._max_samples:
                start = N - self._max_samples
                self.X, self.obs, self.V = self.X[start:], self.obs[start:], self.V[start:]
        N, T = self.X.shape[:2]
        m = self.V.reshape(-1)
        U = np.asarray(policy_mu_fn(self.obs.reshape(N * T, -1)[m]))
        XU = np.concatenate([self.X.reshape(N * T, -1)[m], U], 1)
        K = int(max(2, min(self._max_clusters, np.floor(float(m.sum()) / self._min_samp))))
        self.gmm.update(XU, K)

    def eval(self, Ts, Ps):
        pts = np.concatenate((Ts, Ps), axis=1)
        mu0, Phi, m, n0 = self.gmm.inference(pts)
        n0 *= self._strength
        m *= self._strength
        Phi *= m
        return mu0, Phi, m, n0

    def fit(self, X, pol_mu, pol_sig, V):
        """X (N, T, dX), pol_mu (N, T, dU), pol_sig (dU, dU) (the policy's state-independent covariance), V (N, T).
        Returns pol_K (T, dU, dX), pol_k (T, dU), pol_S (T, dU, dU) in ABSOLUTE coordinates (u = pol_K x + pol_k)."""
        N, T, dX = X.shape
        dU = pol_mu.shape[2]
        pol_K = np.zeros([T, dU, dX]); pol_k = np.zeros([T, dU]); pol_S = np.zeros([T, dU, dU])
        for t in range(T):
            mt = V[:, t]
            if mt.sum() < 2:                           # the code raises on N == 1; a step with < 2 valid samples keeps the prior mean
                mt = np.ones(N, bool)
            Ts, Ps = X[mt, t, :], pol_mu[mt, t, :]
            Ys = np.concatenate([Ts, Ps], axis=1)
            mu0, Phi, mm, n0 = self.eval(Ts, Ps)
            sig_reg = np.zeros((dX + dU, dX + dU))
            if t == 0:
                sig_reg[:dX, :dX] = 1e-8
            dwts = (1.0 / mt.sum()) * np.ones(mt.sum())
            pol_K[t], pol_k[t], pol_S[t] = gauss_fit_joint_prior(Ys, mu0, Phi, mm, n0, dwts, dX, dU, sig_reg)
        pol_S += pol_sig
        return pol_K, pol_k, pol_S
