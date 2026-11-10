#!/usr/bin/env python3
"""
fada_quad_tf.py --teacher fig4/rmat_s<s> --out fig4/fadaT_s<s> (2026-11-07, src/ilc_f1tenth/ilc2real/runs/screen/FADA_AUDIT_1107.md FULL
FIDELITY, the quad): FADA's source pretraining (Xie et al. 2606.28476, Algorithm 1 S1-S5, Eq. 4.2 / 4.3, B.2) with fada_tf.py's transformers
(H 30, K 6), exactly as fada_car_tf.py does for the car.

  oracle      the privileged teacher rmat_s<s>/it300 (ppo_train --mode teacher: reads [obs, z], z the TRUE normalized dynamics), the paper's
              pi^o; suboptimal data from its intermediate checkpoints it25 ... it275 (B.2: 2x the optimal data per iteration)
  source      the figure-4 FADA's sim distribution (fada_train.py --dr --dr-async --priv --plane --est-window 0.025): JumpEnv DR dynamics +
              the robots' timing / mocap (sample_async), plane goals, the stance exploration (q_offset 0.08) -- one jump = one episode,
              the policy acting at its Nc = 60 contact samples
  proprio o_k [state error (6), time (1)] = 7 (2026-11-07 LEAK FIX: was + the previous executed action (4) = 11, so Y contained a_k); command c_k = the goal (2)
  S1          roll out the current source policy (iteration 0: the oracle -- declared) and keep (O^H_k, A^H_k, Y_traj = o_{k+1..k+K})
              -> a_k as executed; the future capped at the takeoff sample Nc (repeated; declared)
  S2          at every visited sample the oracle rolled K samples ahead from the SAME state (MJX state, estimator, command buffers):
              Y_orac; a*_k = the oracle's action there (as executed: the force clip)
  S3          the IDM on traj and oracle-shadow pairs 1 : 1 (declared), first-action MSE (Eq. 4.2), swing legs' unused action entries masked
  S4          the planner through the stop-gradient IDM to a*_k (Eq. 4.3);  S5  DAgger: data aggregated over iterations
Unspecified (declared, = fada_car_tf.py): 8 DAgger iterations, 1024 jumps per block, Adam 3e-4, batch 512, 2000 IDM + 2000 planner steps per
iteration. Fallen / non-finite jumps dropped (fada_train.py). Writes OUT/it<k>/policy (deploy format: policy.npz carries the pickled
transformers as plain arrays FP__* / FI__* (fada_tf.flatten), read by dilc_execute.NumpyActor) + OUT/it<k>.pkl (meta: sim jumps, seconds).
"""
import argparse
import glob
import json
import os
import pickle
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "/home/henry/ilc_ws/src/ilc_f1tenth/ilc2real")          # fada_tf.py (shared with the car)
ap = argparse.ArgumentParser()
ap.add_argument("--teacher", required=True, help="the teacher run dir (it<k>/policy/policy.npz)")
ap.add_argument("--final", type=int, default=300)
ap.add_argument("--o-prev", type=int, default=0, help="(legacy, LEAKY) 1: o = [err, time, prev_a] as before the 2026-11-07 leak fix (fada_tf.d_obs)")
ap.add_argument("--out", required=True)
ap.add_argument("--iters", type=int, default=8)
ap.add_argument("--jumps", type=int, default=1024)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--explore", type=float, default=0.08)
ap.add_argument("--est-window", type=float, default=0.025)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gpu", default="0")
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import fada_tf as F  # noqa: E402
from ilc_mjx.jump import Config, JumpEnv  # noqa: E402

H, K = F.H_HIST, F.K_FUT


class _Adam:
    """Adam (Kingma & Ba; beta 0.9 / 0.999, eps 1e-8): optax.adam's update, written out (the quad env has no optax)."""
    def __init__(self, lr):
        self.lr = lr

    def init(self, p):
        z = jax.tree_util.tree_map(jnp.zeros_like, p)
        return (z, z, jnp.zeros((), jnp.int32))

    def update(self, g, s, p):
        m, v, t = s
        t = t + 1
        m = jax.tree_util.tree_map(lambda m_, g_: 0.9 * m_ + 0.1 * g_, m, g)
        v = jax.tree_util.tree_map(lambda v_, g_: 0.999 * v_ + 0.001 * g_ * g_, v, g)
        u = jax.tree_util.tree_map(lambda m_, v_: -self.lr * (m_ / (1 - 0.9 ** t)) / (jnp.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8), m, v)
        return u, (m, v, t)


apply_updates = lambda p, u: jax.tree_util.tree_map(lambda a_, b_: a_ + b_, p, u)
DC, DA, OD = 2, 4, 31
DO = 11 if a.o_prev else 7                 # (2026-11-07 LEAK FIX) o = [err, time]; the executed actions only as the IDM's A^H tokens
pdir = os.path.join(a.teacher, f"it{a.final}", "policy")
bank = json.load(open(os.path.join(pdir, "bank.json")))
env = JumpEnv(bank, Config(est_window=a.est_window))
Nc, Ndc, N = env.Nc, env.Ndc, env.N
NZ = len(JumpEnv.DYN_KEYS)
env.priv_dim = NZ
MASK = np.ones((Nc, DA), np.float32)
MASK[Ndc:, :2] = 0.0
load_w = lambda p: {k: jnp.asarray(v, jnp.float32) for k, v in np.load(p).items() if k != "n_hidden"}
W_OR = load_w(os.path.join(pdir, "policy.npz"))
SUBOPT = sorted(int(os.path.basename(d)[2:]) for d in glob.glob(os.path.join(a.teacher, "it*")) if os.path.isdir(d)
                and os.path.exists(os.path.join(d, "policy", "policy.npz")) and 0 < int(os.path.basename(d)[2:]) < a.final)
os.makedirs(a.out, exist_ok=True)
logf = open(os.path.join(a.out, "train.log"), "a")


def say(**kw):
    print(json.dumps(kw), flush=True)
    logf.write(json.dumps(kw) + "\n")
    logf.flush()


def make_rollout(student):
    """One jitted rollout of B jumps. student True: the FADA (P, I) acts; False: the teacher weights wt act. Every visited contact sample
    also gets the oracle's action a* and its K-sample shadow (the oracle W_OR from the same state)."""
    def roll(P, I, wt, ref, d_stand, dyn):
        B = ref["x_ref"].shape[0]
        zp = dyn["z"][:, :NZ]
        dx0 = d_stand
        shift = jax.vmap(lambda r, d: r["x_ref"][0, :2] - env.pl.com(env.planar_state(d)[0]))(ref, dx0)
        srb = jax.vmap(env.srb)
        timed = "act_dl" in dyn
        Wb = env.Wb if timed else env.W

        def observe(c, k):
            dx, hist, prev_a, pbuf, du = c
            x = srb(dx, shift)
            if env.ekf_on:
                x_obs = env._ekf_obs(pbuf, shift)
            elif env.W and timed:
                pa = jnp.einsum("bw,bwi->bi", env.est_ad[dyn["pose_dl"]], pbuf)
                pb = jnp.einsum("bw,bwi->bi", env.est_bd[dyn["pose_dl"]], pbuf)
                x_obs = jnp.concatenate([pa[:, :2] + shift, pa[:, 2:3], pb], 1)
            elif env.W:
                pa = jnp.einsum("w,bwi->bi", env.est_a, pbuf)
                pb = jnp.einsum("w,bwi->bi", env.est_b, pbuf)
                x_obs = jnp.concatenate([pa[:, :2] + shift, pa[:, 2:3], pb], 1)
            else:
                x_obs = x
            err = jnp.clip((x_obs - ref["x_ref"][:, k]) / env.sx, -20.0, 20.0)
            hist = jnp.concatenate([err[:, None], hist[:, :-1]], 1) if env.H else err[:, None]
            o_t = jax.vmap(lambda h, pa_, g: env.obs(h, pa_, k, g))(hist, prev_a, ref["goal_obs"])
            o_t = jnp.concatenate([o_t, zp], 1)                                       # the teacher's observation
            o11 = jnp.concatenate([err, jnp.full((B, 1), 2.0 * k / Nc - 1.0)] + ([prev_a] if a.o_prev else []), 1)   # name kept; DO wide
            return hist, o_t, o11, x

        def advance(c, hist, act, k):
            dx, _, prev_a, pbuf, du = c
            u, a_st = jax.vmap(lambda a_, r: env._forces(a_, k, r))(act, ref)
            u_hold = ref["u_ref"][:, k] + jnp.where(env.swing[k], 0.0, du)
            du_next = u - ref["u_ref"][:, k]
            if timed:
                du_next = jnp.where(dyn["drop"][:, k, None], du, du_next)
            dx, h, ps = jax.vmap(lambda d, u_, r, p, uh: env._fly_sample(d, k, u_, r, True, p, True, uh))(dx, u, ref, dyn, u_hold)
            if env.ekf_on:
                pbuf = env._ekf_step(pbuf, ps, dyn, k, timed)
            elif env.W:
                pbuf = jnp.concatenate([pbuf, ps + dyn["pose_noise"][:, k] if timed else ps], 1)[:, -Wb:]
            return (dx, hist, a_st, pbuf, du_next), a_st

        teach = lambda w, o: jax.vmap(lambda o_: env.actor(w, o_))(o)
        sel = lambda f, x, y: jax.tree_util.tree_map(lambda p, q: jnp.where(f.reshape((-1,) + (1,) * (p.ndim - 1)) if p.ndim else f, p, q), x, y)

        def contact(carry, k):
            c, Oh, Ah, mh = carry
            hist, o_t, o11, x = observe(c, k)
            Oh = jnp.concatenate([Oh[:, 1:], o11[:, None]], 1)
            Ah = jnp.concatenate([Ah[:, 1:], c[2][:, None]], 1)                        # the action executed before this sample
            mh = jnp.concatenate([mh[:, 1:], jnp.ones((B, 1), bool)], 1)
            a_or = teach(W_OR, o_t)
            if student:
                Y = F.planner_fwd(P, Oh, ref["goal_obs"], mh, xp=jnp)
                act = F.idm_fwd(I, Oh, Ah, Y, mh, xp=jnp)[:, 0]
            else:
                act = teach(wt, o_t)
            _, astar = jax.vmap(lambda a_, r: env._forces(a_, k, r))(a_or, ref)           # a* as the force clip would execute it
            # S2: the oracle's K-sample shadow from this very state
            cs, hs, o_tj, o_prev, ys = c, hist, o_t, o11, []      # the shadow's carry, history, teacher obs, latest o (frozen at Nc)
            for j in range(K):
                kj = jnp.minimum(k + j, Nc - 1)
                live = (k + j) < Nc
                cs_new, _ = advance(cs, hs, teach(W_OR, o_tj), kj)
                hs_new, o_t_new, o11_new, _ = observe(cs_new, jnp.minimum(k + j + 1, Nc))
                cs = sel(jnp.full((B,), live), cs_new, cs)
                hs, o_tj, o_prev = (jnp.where(live, x1, x0) for x1, x0 in ((hs_new, hs), (o_t_new, o_tj), (o11_new, o_prev)))
                ys.append(o_prev)
            c, a_st = advance(c, hist, act, k)
            return (c, Oh, Ah, mh), (o11, a_st, astar, jnp.stack(ys, 1), x)

        def flight(carry, k):
            dx = carry
            x = srb(dx, shift)
            dx, _, _ = jax.vmap(lambda d, r, p: env._fly_sample(d, k, jnp.zeros(4), r, False, p, True))(dx, ref, dyn)
            return dx, x

        hist0 = jnp.zeros((B, env.H + 1, 6))
        pbuf0 = env._ekf_init(dx0, dyn, timed) if env.ekf_on else jnp.repeat(jax.vmap(env._pose3)(dx0)[:, None], max(Wb, 1), 1)
        c0 = (dx0, hist0, jnp.zeros((B, DA)), pbuf0, jnp.zeros((B, DA)))
        carry0 = (c0, jnp.zeros((B, H, DO)), jnp.zeros((B, H, DA)), jnp.zeros((B, H), bool))
        (c, _, _, _), (O, A, AS, YO, _) = jax.lax.scan(contact, carry0, jnp.arange(Nc))
        _, _, o_nc, _ = observe(c, Nc)
        dx = c[0]
        dx, _ = jax.lax.scan(flight, dx, jnp.arange(Nc, N))
        gl = env.cfg.gains[3]
        cmd = (env.q_stand, jnp.zeros(12), jnp.full(12, gl[0]), jnp.full(12, gl[1]), env.tau_stand)

        def land(carry, _):
            dx, fell = carry
            dx = jax.vmap(lambda d, p, r: env._step(d, cmd, p, r, True))(dx, dyn, ref)
            quat = dx.qpos[:, 3:7]
            tilt = jnp.arccos(jnp.clip(1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), -1.0, 1.0))
            return (dx, fell | (tilt > env.cfg.max_tilt) | (dx.qpos[:, 2] < env.cfg.min_height)), None
        from ilc_mjx.jump import TICK
        (dx, fell), _ = jax.lax.scan(land, (dx, jnp.zeros(B, bool)), None, length=int(round(env.cfg.land_time / TICK)))
        sw = lambda v: jnp.swapaxes(v, 0, 1)
        return dict(o=jnp.concatenate([sw(O), o_nc[:, None]], 1), a=sw(A), astar=sw(AS), Yo=sw(YO), fell=fell)
    return jax.jit(roll)


def windows(D, idx_n, idx_k):
    """(O^H, A^H, mask, Y_traj, Y_orac, a_exec, a*, c, swing mask) for jump rows idx_n at samples idx_k."""
    o, act = D["o"][idx_n], D["a"][idx_n]                         # (b, Nc+1, 11), (b, Nc, 4)
    b = len(idx_n)
    t = idx_k[:, None] - (H - 1) + np.arange(H)[None]             # history sample indices (b, H)
    valid = t >= 0
    tc = np.clip(t, 0, None)
    Oh = np.where(valid[..., None], np.take_along_axis(o, tc[..., None], 1), 0.0)
    ap = np.concatenate([np.zeros((b, 1, DA)), act], 1)           # ap[:, j] = the action executed before sample j
    Ah = np.where(valid[..., None], np.take_along_axis(ap, tc[..., None], 1), 0.0)
    tf_ = np.minimum(idx_k[:, None] + 1 + np.arange(K)[None], Nc)
    Yt = np.take_along_axis(o, tf_[..., None], 1)
    Yo = D["Yo"][idx_n, idx_k]
    return (Oh.astype(np.float32), Ah.astype(np.float32), valid, Yt.astype(np.float32), Yo.astype(np.float32),
            act[np.arange(b), idx_k].astype(np.float32), D["astar"][idx_n, idx_k].astype(np.float32),
            D["c"][idx_n].astype(np.float32), MASK[idx_k])


def export(it, P, I, meta):
    d = os.path.join(a.out, f"it{it}", "policy")
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "policy.npz"), fada_tf=np.array(1), n_hidden=np.array(2), **F.flatten(F.to_np(P), "FP"), **F.flatten(F.to_np(I), "FI"))
    for f in ("bank.json", "ilc_U.npz", "ilc_best.json"):
        if os.path.exists(os.path.join(pdir, f)):
            shutil.copy(os.path.join(pdir, f), d)
    json.dump(dict(best="policy", eval_goals=None, members=[dict(name="policy", group=0, variant="policy", tag="policy", episodes=0,
                                                                 updates=0, score=None)]), open(os.path.join(d, "manifest.json"), "w"))
    F.save(os.path.join(a.out, f"it{it}.pkl"), P, I, meta)


def main():
    rng = np.random.default_rng(a.seed)
    P = jax.tree_util.tree_map(jnp.asarray, F.init_planner(a.seed, DO, DC))
    I = jax.tree_util.tree_map(jnp.asarray, F.init_idm(a.seed + 1, DO, DA))
    roll_s, roll_t = make_rollout(True), make_rollout(False)
    oi, op = _Adam(a.lr), _Adam(a.lr)
    sI, sP = oi.init(I), op.init(P)

    @jax.jit
    def stepI(Iq, s, Oh, Ah, m, Y, tgt, mk):
        l, g = jax.value_and_grad(lambda q: jnp.mean(jnp.sum(((F.idm_fwd(q, Oh, Ah, Y, m, xp=jnp)[:, 0] - tgt) * mk) ** 2, -1)))(Iq)
        u, s = oi.update(g, s, Iq)
        return apply_updates(Iq, u), s, l

    @jax.jit
    def stepP(Pq, Iq, s, Oh, Ah, m, C, tgt, mk):
        def loss(q):
            Yh = F.planner_fwd(q, Oh, C, m, xp=jnp)
            return jnp.mean(jnp.sum(((F.idm_fwd(jax.lax.stop_gradient(Iq), Oh, Ah, Yh, m, xp=jnp)[:, 0] - tgt) * mk) ** 2, -1))
        l, g = jax.value_and_grad(loss)(Pq)
        u, s = op.update(g, s, Pq)
        return apply_updates(Pq, u), s, l

    data = []                                   # aggregated blocks (S5)
    t0, jumps = time.time(), 0
    B = 16 if a.smoke else a.jumps
    for it in range(a.iters + 1 if not a.smoke else 2):
        blocks = [("oracle" if it == 0 else "student", None)] + [("subopt", int(rng.choice(SUBOPT))) for _ in range(2)]
        alive = []
        for kind, ck in blocks:
            goals = env.sample_plane_goals(B, rng)
            ref = env.references(goals)
            dyn = env.dyn_arrays(env.sample_dyn(B, rng))
            dyn.update(env.sample_async(B, rng))
            q_off, _ = env.explore_noise(B, rng, a.explore, 0.0)
            d0 = env.stand_batch(q_off, dyn)
            wt = W_OR if kind == "oracle" else (load_w(os.path.join(a.teacher, f"it{ck}", "policy", "policy.npz")) if kind == "subopt" else W_OR)
            out = jax.tree_util.tree_map(np.asarray, (roll_s if kind == "student" else roll_t)(P, I, wt, ref, d0, dyn))
            ok = ~out["fell"] & np.isfinite(out["o"]).all((1, 2)) & np.isfinite(out["Yo"]).all((1, 2, 3)) & np.isfinite(out["a"]).all((1, 2))
            blk = {k: v[ok] for k, v in out.items() if k != "fell"}
            blk["c"] = np.asarray(ref["goal_obs"])[ok]
            data.append(blk)
            jumps += B
            alive.append(float(ok.mean()))
        D = {k: np.concatenate([d[k] for d in data]) for k in data[0]}
        n = len(D["o"])

        def sample(bs):
            i_n = rng.integers(0, n, bs); i_k = rng.integers(0, Nc, bs)
            return windows(D, i_n, i_k)
        li, lp = [], []
        for st in range(a.steps if not a.smoke else 20):
            Oh, Ah, m, Yt, Yo, ae, ast, C, mk = sample(a.batch)
            h = a.batch // 2                    # traj pairs : oracle-shadow pairs = 1 : 1
            Y = np.concatenate([Yt[:h], Yo[h:]]); tg = np.concatenate([ae[:h], ast[h:]])
            I, sI, l = stepI(I, sI, Oh, Ah, m, Y, tg, mk); li.append(float(l))
        for st in range(a.steps if not a.smoke else 20):
            Oh, Ah, m, Yt, Yo, ae, ast, C, mk = sample(a.batch)
            P, sP, l = stepP(P, I, sP, Oh, Ah, m, C, ast, mk); lp.append(float(l))
        meta = dict(it=it, sim_jumps=jumps, seconds=round(time.time() - t0, 1), teacher=os.path.abspath(a.teacher), final=a.final)
        export(it, P, I, meta)
        say(it=it, rollout=blocks[0][0], subopt=[b[1] for b in blocks[1:]], alive=alive, n_jumps_data=int(n),
            idm_loss=[round(float(np.mean(li[:20])), 5), round(float(np.mean(li[-20:])), 5)],
            planner_loss=[round(float(np.mean(lp[:20])), 5), round(float(np.mean(lp[-20:])), 5)], sim_jumps=jumps, t=round(time.time() - t0, 1))
    say(event="done", sim_jumps=jumps, t=round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
