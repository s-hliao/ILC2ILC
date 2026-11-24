"""Step-0 positivity of the SRBD-linearized ILC step on the force-based Go1 trot (MJX "real" = a perturbed Go1).

Per walk (goal straight-ish ahead): fly the real robot; along the MEASURED control steps linearize the nominal SRBD
(A, B wrt the stance-foot forces at the measured feet, held) and the controller's feedback K = d pi / d s (autodiff), the
closed loop A + B K; rows = trunk xy error vs the reference / 5 cm and yaw / 0.1 rad at every control step; the lifted G
(columns: the 12 force offsets per control step, swing legs and saturated components masked); the damped GN step
du = -beta (G'G + lam I)^-1 G'e, lam = delta tr(G'G) / n (beta = delta = 0.5), rms-capped (--cap N). Re-fly the real robot
with du added to the stance forces; compare the predicted row change G du with the measured one (cosine, gain) and the
rms xy error before / after.

    python scripts/positivity.py --level full --walks 8 --out results/pos_full.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import jax                                           # noqa: E402
import jax.numpy as jnp                              # noqa: E402

from ilc_go1walk import walk as W                    # noqa: E402

LEVELS = dict(none={}, mild=dict(payload=1.0, torque_scale=0.95), full=dict(payload=2.0, torque_scale=0.85, friction=0.4))
POS_SCALE, YAW_SCALE = 0.05, 0.1


def rows_of(s, t, goal):
    pr = jax.vmap(lambda tt: W.reference(tt, goal, 0.0)[0])(t)
    e = jnp.concatenate([(s[:, :2] - pr[:, :2]) / POS_SCALE, (s[:, 5:6] - 0.0) / YAW_SCALE], 1)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', default='full', choices=list(LEVELS))
    ap.add_argument('--walks', type=int, default=8)
    ap.add_argument('--steps', type=int, default=175)
    ap.add_argument('--beta', type=float, default=0.5)
    ap.add_argument('--delta', type=float, default=0.5)
    ap.add_argument('--cap', type=float, default=15.0, help='rms cap of du (N)')
    ap.add_argument('--k-raib', type=float, default=0.17)
    ap.add_argument('--max-lat-deg', type=float, default=5.0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    W.K_RAIB = a.k_raib
    nom = W.Sim()
    rob = W.Robot(nom.qm)
    mi = (rob.mass, jnp.asarray(rob.inertia))
    real = W.Sim(**LEVELS[a.level])
    n = a.steps
    rng = np.random.default_rng(7)
    lin_s = jax.jit(jax.vmap(jax.jacfwd(W.srbd_step, argnums=(0, 1)), (0, 0, 0, None, None)))
    lin_k = jax.jit(jax.vmap(jax.jacfwd(W.controller, argnums=0), (0, 0, 0, 0, None, None, None, None)))
    res = []
    for w in range(a.walks):
        r = rng.uniform(0.6, 1.2)
        ang = np.radians(rng.uniform(-a.max_lat_deg, a.max_lat_deg))
        goal = jnp.asarray([r * np.cos(ang), r * np.sin(ang)])
        out, _ = real.rollout(goal, jnp.zeros((n, 4, 3)), mi, n)
        s, feet, c, u, t = (out[k] for k in ('s', 'feet', 'c', 'u', 't'))
        fell = bool(jnp.any(jnp.abs(s[:, 3:5]) > np.radians(30)) | (s[:, 2].min() < 0.15))
        e = rows_of(s, t, goal)                                    # (n, 3); rows at steps 1..n-1 (s[k] is the state at step k)
        A, B = lin_s(s[:-1], u[:-1], feet[:-1], mi[0], mi[1])     # (n-1, 12, 12), (n-1, 12, 4, 3)
        B = B.reshape(n - 1, 12, 12)
        K = lin_k(s[:-1], feet[:-1], c[:-1], t[:-1], goal, 0.0, mi[0], mi[1]).reshape(n - 1, 12, 12)
        A, B, K = (np.asarray(z, np.float64) for z in (A, B, K))
        bad = int(sum(np.isnan(z).sum() for z in (A, B, K)))
        if bad:
            print(f'!!! {bad} NaN entries in A, B, K', flush=True)
        cm = np.repeat(np.asarray(c[:-1], float), 3, axis=1)       # swing legs: no force columns
        uu = np.asarray(u[:-1]).reshape(n - 1, 12)
        sat = np.zeros_like(uu, bool)
        sat[:, 2::3] = (uu[:, 2::3] <= 0.5) | (uu[:, 2::3] >= W.FZ_MAX - 0.5)
        mask = cm * (~sat)
        Acl = A + np.einsum('tij,tjk->tik', B, K)
        T = n - 1
        C = np.zeros((3, 12)); C[0, 0] = C[1, 1] = 1 / POS_SCALE; C[2, 5] = 1 / YAW_SCALE
        G = np.zeros((T, 3, T, 12))                                # rows at state s[k+1], k = 0..T-1
        S = np.zeros((T, 12, 12))
        for k in range(T):
            S[:k] = np.einsum('ij,tjk->tik', Acl[k], S[:k]) if k else S[:k]
            S[k] = B[k] * mask[k][None]
            G[k, :, :k + 1] = np.einsum('ri,tik->rtk', C, S[:k + 1])
            # propagate to the next state: done at the next iteration's Acl[k+1]
        # note: S[j] after the loop body at step k = d s[k+1] / d du_j
        Gf = G.reshape(T * 3, T * 12)
        ef = np.asarray(e[1:]).reshape(-1)
        free = np.abs(Gf).sum(0) > 0
        Gq = Gf[:, free]
        N = Gq.T @ Gq
        lam = a.delta * np.trace(N) / max(N.shape[0], 1)
        dq = -a.beta * np.linalg.solve(N + lam * np.eye(N.shape[0]), Gq.T @ ef)
        rms = np.sqrt(np.mean(dq ** 2))
        if rms > a.cap:
            dq *= a.cap / rms
        du = np.zeros(T * 12); du[free] = dq
        pred = Gf @ du
        du_full = np.zeros((n, 4, 3)); du_full[:T] = du.reshape(T, 4, 3)
        out2, _ = real.rollout(goal, jnp.asarray(du_full, jnp.float32), mi, n)
        s2 = out2['s']
        fell2 = bool(jnp.any(jnp.abs(s2[:, 3:5]) > np.radians(30)) | (s2[:, 2].min() < 0.15))
        e2 = rows_of(s2, out2['t'], goal)
        meas = np.asarray(e2[1:] - e[1:]).reshape(-1)
        cos = float(pred @ meas / (np.linalg.norm(pred) * np.linalg.norm(meas) + 1e-12))
        gain = float(meas @ pred / (pred @ pred + 1e-12))
        xy1 = float(100 * POS_SCALE * np.sqrt(np.mean(np.sum(np.asarray(e[1:, :2]) ** 2, 1))))
        xy2 = float(100 * POS_SCALE * np.sqrt(np.mean(np.sum(np.asarray(e2[1:, :2]) ** 2, 1))))
        row = dict(walk=w, goal=np.round(np.asarray(goal), 3).tolist(), fell=fell, fell_after=fell2, cos=cos, gain=gain,
                   du_rms_N=float(np.sqrt(np.mean(du[free] ** 2))) if free.any() else 0.0,
                   xy_rms_cm_before=xy1, xy_rms_cm_after=xy2, pred_norm=float(np.linalg.norm(pred)),
                   meas_norm=float(np.linalg.norm(meas)))
        res.append(row)
        print(json.dumps(row), flush=True)
    ok = [r for r in res if not r['fell'] and not r['fell_after']]
    summ = dict(level=a.level, n=len(res), n_upright=len(ok),
                cos_median=float(np.median([r['cos'] for r in ok])) if ok else None,
                positive=float(np.mean([r['cos'] > 0 for r in ok])) if ok else None,
                gain_median=float(np.median([r['gain'] for r in ok])) if ok else None,
                xy_before=float(np.mean([r['xy_rms_cm_before'] for r in ok])) if ok else None,
                xy_after=float(np.mean([r['xy_rms_cm_after'] for r in ok])) if ok else None)
    print('SUMMARY', json.dumps(summ), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(summary=summ, rows=res, args=vars(a)), open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
