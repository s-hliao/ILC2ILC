"""
The closed-loop Gauss-Newton ILC step on a batch of measured (or simulated) runs, from the NOMINAL model's
Jacobians along the given states (sim_jax): G[t, j] = C_t Phi_cl(t, j+1) B_j with Phi_cl built from A_t + B_t K_t
(K_t the network's own feedback through its observation, or the LQR's, or none), errors scaled by ERR_SCALE,
step = -beta (G'G + delta tr(G'G)/n I)^-1 G'e with its rms capped. Rows are masked by `valid`.
jac_mode: 'closed' (default), 'open' (K = 0), 'flip' (the 180-degree control: G negated).
"""
import jax
import jax.numpy as jnp
import numpy as np

import car_model as cm
import sim_jax as sj


def make_gn(P, T, policy_fn, beta=0.5, delta=0.05, cap=0.05, jac_mode='closed', feedback='policy', err_scale=None):
    """Returns gn(params, xs (n, T+1, 10), idx (n, T), a (n, T, 2) normalized, pid (n,), valid (n, T), err (n, T, NE))
    -> step (n, T, 2), J, J_pred. feedback: 'policy' (K from policy_fn(params, obs)), 'expert' (the plan's LQR,
    for the per-track ILC baseline), 'none'."""
    p0 = sj.nominal_params(1)
    p1 = {k: v[0] for k, v in p0.items()}

    def lane(params, X, I, Ap, q, V, E):
        def one(x, i, a):
            A = jax.jacfwd(lambda xx: sj.step(xx, a * sj.A_SCALE + sj.A_OFF, p1))(x)
            B = jax.jacfwd(lambda aa: sj.step(x, aa * sj.A_SCALE + sj.A_OFF, p1))(a)
            C = jax.jacfwd(lambda xx: sj.features(P, q, i, xx)[1])(x)
            if jac_mode == 'open' or feedback == 'none':
                K = jnp.zeros((2, sj.NXS))
            elif feedback == 'expert':
                K = jax.jacfwd(lambda xx: (sj.expert_action(P, q, i, xx) - sj.A_OFF) / sj.A_SCALE)(x)
            else:
                K = jax.jacfwd(lambda xx: policy_fn(params, sj.features(P, q, i, xx)[0]))(x)
            return A, B, C, K
        A, B, C, K = jax.vmap(one)(X[:-1], I, Ap)
        Acl = A + jnp.einsum('tij,tjk->tik', B, K)

        def body(Vc, t):
            rows = jnp.einsum('ij,kjl->kil', C[t], Vc)
            Vn = jnp.einsum('ij,kjl->kil', Acl[t], Vc)
            Vn = Vn.at[t].set(B[t])
            return Vn, rows
        _, rows = jax.lax.scan(body, jnp.zeros((T, sj.NXS, 2)), jnp.arange(T))
        w = 1.0 / jnp.array(sj.ERR_SCALE if err_scale is None else err_scale)
        Vf = V.astype(jnp.float32)
        G = rows * w[None, None, :, None] * Vf[:, None, None, None] * Vf[None, :, None, None]
        G = jnp.transpose(G, (0, 2, 1, 3)).reshape(T * sj.NE, T * 2)
        if jac_mode == 'flip':
            G = -G
        e = (E * w[None, :] * Vf[:, None]).reshape(-1)
        H = G.T @ G
        n = H.shape[0]
        lam = delta * jnp.trace(H) / n + 1e-9
        step = -beta * jnp.linalg.solve(H + lam * jnp.eye(n), G.T @ e)
        rms = jnp.sqrt(jnp.sum(step ** 2) / (2 * jnp.maximum(jnp.sum(Vf), 1.0)))
        step = step * jnp.minimum(1.0, cap / (rms + 1e-12))
        e_pred = e + G @ step
        return step.reshape(T, 2) * Vf[:, None], 0.5 * jnp.sum(e ** 2), 0.5 * jnp.sum(e_pred ** 2)

    return jax.jit(jax.vmap(lane, in_axes=(None, 0, 0, 0, 0, 0, 0)))


def pad_runs(runs, T, nx=9):
    """Stack real runs (real_car.run dicts) to (n, T+1, ...) arrays with a validity mask (rows before the end of the
    run, minus 8 periods before a failure)."""
    n = len(runs)
    xs = np.zeros((n, T + 1, nx))
    idx = np.zeros((n, T), np.int32)
    a = np.zeros((n, T, 2))
    err = np.zeros((n, T, sj.NE))
    valid = np.zeros((n, T), bool)
    for k, r in enumerate(runs):
        m = min(len(r['x']), T)
        if m == 0:
            continue
        xs[k, :m] = r['x'][:m]
        xs[k, m:] = r['x'][m - 1]
        idx[k, :m] = r['idx'][:m]
        idx[k, m:] = r['idx'][m - 1]
        a[k, :m] = r['a'][:m]
        err[k, :m] = r['err'][:m]
        mv = m - 8 if r['failed'] else m
        valid[k, :max(mv, 0)] = True
    return xs, idx, a, err, valid
