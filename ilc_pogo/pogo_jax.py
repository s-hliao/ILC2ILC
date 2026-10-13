"""
pogo_jax.py: the NOMINAL pogo stick (common.substep: the 2-D SLIP with a pinned foot), batched in JAX on the GPU --
the model the method trains in and takes its Jacobians from. 0.2 ms steps (SUB per control period), no leg damping,
nominal mass / spring / actuators. Every lane carries its own parameter dict (nominal, or DR draws).
"""
import jax
import jax.numpy as jnp
import numpy as np

from common import (DT, NX, NOMINAL, NE, ERR_SCALE, features, errors, failed, expert_action, substep,
                    mlp as _mlp)

SUB = 50


def nominal_params(n):
    return {k: jnp.full((n,), v, jnp.float32) for k, v in NOMINAL.items()}


def dr_params(key, n, scale=1.0):
    """The domain-randomization family (the DR variants and the RL baselines): mass +-25 %, spring stiffness +-20 %,
    leg damping 0-30 N s/m, thrust gain +-20 %, leg-angle offset +-0.03 rad, rest length +-4 %."""
    k = jax.random.split(key, 6)
    u = lambda kk, lo, hi: lo + (hi - lo) * jax.random.uniform(kk, (n,))
    p = nominal_params(n)
    p['m'] = p['m'] * u(k[0], 1 - 0.25 * scale, 1 + 0.25 * scale)
    p['k'] = p['k'] * u(k[1], 1 - 0.2 * scale, 1 + 0.2 * scale)
    p['b'] = u(k[2], 0.0, 30.0 * scale)
    p['gth'] = u(k[3], 1 - 0.2 * scale, 1 + 0.2 * scale)
    p['phi_off'] = u(k[4], -0.03 * scale, 0.03 * scale)
    p['l0'] = p['l0'] * u(k[5], 1 - 0.04 * scale, 1 + 0.04 * scale)
    return p


def step(x, a, p):
    """One control period (SUB substeps) under the normalized action a."""
    h = DT / SUB

    def sub(xx, _):
        return substep(xx, a, p, h, jnp), None
    x, _ = jax.lax.scan(sub, x, None, length=SUB)
    return x


def init_mlp(key, sizes):
    ps = []
    for i in range(len(sizes) - 1):
        key, kk = jax.random.split(key)
        W = jax.random.normal(kk, (sizes[i], sizes[i + 1])) * np.sqrt(1.0 / sizes[i])
        if i == len(sizes) - 2:
            W = W * 0.01
        ps.append((W, jnp.zeros(sizes[i + 1])))
    return ps


def mlp(params, o):
    return _mlp(params, o, jnp)


def obs(x, g):
    return features(x, g, jnp)


def err(x, g):
    return errors(x, g, jnp)


def expert(x, g):
    return expert_action(x, g, jnp)


def start_states(key, goals, pert):
    """Drops: in flight at the goal's apex (z = h*, vx = v*), plus a perturbation of relative size `pert` (per lane):
    z +-5 cm, vx +-0.4 m/s, vz +-0.3 m/s."""
    goals = jnp.asarray(goals)
    n = goals.shape[0]
    pert = jnp.broadcast_to(jnp.asarray(pert, jnp.float32), (n,))
    d = jax.random.uniform(key, (n, 3), minval=-1, maxval=1) * pert[:, None]
    x0 = jnp.zeros((n, NX))
    return x0.at[:, 1].set(goals[:, 1] + 0.05 * d[:, 0]).at[:, 2].set(goals[:, 0] + 0.4 * d[:, 1]).at[:, 3].set(0.3 * d[:, 2])


def rollout(policy, T, x0, g, p, noise=None):
    """Batched rollout (lanes n): policy(o) -> normalized action mean (None: the Raibert expert); noise (n, T, 2) is
    added to the applied action. A lane freezes at its first failure (sticky flag)."""
    n = x0.shape[0]
    if noise is None:
        noise = jnp.zeros((n, T, 2))

    def lane(x0, gg, pp, nz):
        def body(c, t):
            x, f = c
            o = obs(x, gg)
            mu = expert(x, gg) if policy is None else policy(o)
            a = mu + nz[t]
            xn = step(x, a, pp)
            fn = f | failed(xn, jnp)
            xn = jnp.where(fn, x, xn)
            return (xn, fn), dict(x=x, obs=o, mu=mu, a=a, err=err(x, gg), failed=f)
        (xe, fe), out = jax.lax.scan(body, (x0, jnp.array(False)), jnp.arange(T))
        out['x_end'], out['failed_end'] = xe, fe
        out['x'] = jnp.concatenate([out['x'], xe[None]], 0)
        return out
    return jax.vmap(lane)(x0, g, p, noise)


def lane_jacobians(policy_fn, params, g, p, feedback='policy'):
    """Per step one-period Jacobians at (x, a): A = d step / dx, B = d step / da, C = d err / dx and the feedback
    K = d a / dx ('policy': the network through its observation; 'expert': the Raibert controller; 'none').
    Within-mode derivatives (the touchdown / liftoff switches are not differentiated: the leg force is ~0 at both)."""
    def one(x, a):
        A = jax.jacfwd(lambda xx: step(xx, a, p))(x)
        B = jax.jacfwd(lambda aa: step(x, aa, p))(a)
        C = jax.jacfwd(lambda xx: err(xx, g))(x)
        if feedback == 'policy':
            K = jax.jacfwd(lambda xx: policy_fn(params, obs(xx, g)))(x)
        elif feedback == 'expert':
            K = jax.jacfwd(lambda xx: expert(xx, g))(x)
        else:
            K = jnp.zeros((2, NX))
        return A, B, C, K
    return one


def gn_lane(A, B, C, K, E, valid, beta, delta, cap, err_scale=ERR_SCALE, flip=False, vectors=False):
    """The closed-loop Gauss-Newton ILC step of one lane / trial: G[t, j] = C_t Phi_cl(t, j+1) B_j (Phi_cl from
    A + B K), rows scaled by err_scale and masked by valid; step = -beta (G'G + delta tr(G'G)/n I)^-1 G'e, rms capped.
    Returns step (T, 2), J, J_pred (and, with vectors, the scaled error e and its predicted change G step)."""
    T = E.shape[0]
    Acl = A + jnp.einsum('tij,tjk->tik', B, K)

    def body(V, t):
        rows = jnp.einsum('ij,kjl->kil', C[t], V)
        Vn = jnp.einsum('ij,kjl->kil', Acl[t], V)
        Vn = Vn.at[t].set(B[t])
        return Vn, rows
    _, rows = jax.lax.scan(body, jnp.zeros((T, NX, 2)), jnp.arange(T))
    w = 1.0 / jnp.asarray(err_scale)
    vf = valid.astype(jnp.float32)
    G = rows * w[None, None, :, None] * vf[:, None, None, None] * vf[None, :, None, None]
    G = jnp.transpose(G, (0, 2, 1, 3)).reshape(T * NE, T * 2)
    if flip:
        G = -G
    e = (E * w[None, :] * vf[:, None]).reshape(-1)
    H = G.T @ G
    n = H.shape[0]
    lam = delta * jnp.trace(H) / n + 1e-9
    stp = -beta * jnp.linalg.solve(H + lam * jnp.eye(n), G.T @ e)
    rms = jnp.sqrt(jnp.sum(stp ** 2) / (2 * jnp.maximum(jnp.sum(vf), 1.0)))
    stp = stp * jnp.minimum(1.0, cap / (rms + 1e-12))
    de = G @ stp
    e_pred = e + de
    out = (stp.reshape(T, 2) * vf[:, None], 0.5 * jnp.sum(e ** 2), 0.5 * jnp.sum(e_pred ** 2))
    return out + (e, de) if vectors else out
