#!/usr/bin/env python3
"""
ppo_train.py --init POLICY.npz --mode plain|teacher|student --out DIR: faithful RL baselines for the DR comparison --
PPO (clipped surrogate, GAE, a learned value) on the GPU sim under domain randomization, as RMA's teacher and FADA's
oracle are trained (our method never randomizes; dr_train.py is the same comparison with our own learner).

  plain    PPO under DR, no privileged input: one robust policy
  teacher  PPO under DR, the actor reading a latent l = E z + b_E of the robot's normalized dynamics z (RMA's
           extrinsics; the encoder linear, so the exported actor folds it in: a dilc/deploy actor with z appended)
  student  RMA phase 2: the teacher's base policy frozen, an adaptation module phi(o) -> l (o the deployable
           observation, which already carries 3 past errors and the last action) regressed onto E z on the
           student's own DR rollouts; exported as the composite actor (phi in the P_* slot), deployable as is

Policy: a = tanh(mu_theta(o)) + sigma * eps (sigma a learned per-channel std; the noise enters as the rollout's action
offset, so the likelihood is exact), swing-leg channels masked. Reward: at every contact sample -beta x the mean
squared normalized SRB error (dense shaping), and at the end -(0.01 r_scale e'Qe + 20 fall) of the landing (the
evaluation score). Starts from --init (our nominal sim policy: the same warm start as dr_train.py's baselines); the
value net warms up first with the actor frozen.
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True)
ap.add_argument("--mode", required=True, choices=("plain", "teacher", "student"))
ap.add_argument("--teacher", default="", help="student: the teacher's raw params (OUT/teacher_params.npz)")
ap.add_argument("--out", required=True)
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--minibatch", type=int, default=4096)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--vlr", type=float, default=3e-4)
ap.add_argument("--clip", type=float, default=0.2)
ap.add_argument("--gamma", type=float, default=0.99)
ap.add_argument("--lam", type=float, default=0.95)
ap.add_argument("--beta", type=float, default=0.01, help="dense shaping weight")
ap.add_argument("--sigma0", type=float, default=0.1)
ap.add_argument("--latent", type=int, default=8)
ap.add_argument("--value-warmup", type=int, default=10)
ap.add_argument("--explore-q", type=float, default=0.08, help="stance offsets (as every arm)")
ap.add_argument("--goals", type=float, nargs=2, default=[0.40, 0.60])
ap.add_argument("--eval-every", type=int, default=25)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gpu", default="0")
a = ap.parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ilc_mjx.jump import JumpEnv  # noqa: E402

os.makedirs(a.out, exist_ok=True)
pdir0 = os.path.dirname(os.path.abspath(a.init))
while not os.path.exists(os.path.join(pdir0, "bank.json")):     # (a teacher's params file sits beside its run)
    pdir0 = os.path.dirname(pdir0)
bank = json.load(open(os.path.join(pdir0, "bank.json")))
env = JumpEnv(bank)
Nc, Ndc, N = env.Nc, env.Ndc, env.N
qe, r_scale = np.asarray(bank["qe"], float), float(bank["r_scale"])
NZ, L, OD = len(JumpEnv.DYN_KEYS), a.latent, 31
mask = np.ones((Nc, 4))
mask[Ndc:, :2] = 0.0
mask_j = jnp.asarray(mask, jnp.float32)
logf = open(os.path.join(a.out, "train.log"), "a")


def say(msg):
    print(msg, flush=True)
    logf.write(msg + "\n")
    logf.flush()


rng = np.random.default_rng(a.seed)
key = jax.random.PRNGKey(a.seed)
he = lambda k, o, i, s=1.0: jax.random.normal(k, (o, i)) * jnp.sqrt(2.0 / i) * s
w0 = {k: jnp.asarray(v, jnp.float32) for k, v in np.load(a.init).items() if k != "n_hidden"}

# -- parameters --------------------------------------------------------------------------------------------------
if a.mode == "student":
    T = {k: jnp.asarray(v, jnp.float32) for k, v in np.load(a.teacher).items()}
    ks = jax.random.split(key, 3)
    phi = dict(P_W0=he(ks[0], 256, OD), P_b0=jnp.zeros(256), P_W1=he(ks[1], 256, 256), P_b1=jnp.zeros(256),
               P_Wo=he(ks[2], L, 256, 0.1), P_bo=jnp.zeros(L))
else:
    pi = dict(w0)
    if a.mode == "teacher":                         # the latent's inputs at zero weight: starts as the warm start
        pi["W0"] = jnp.concatenate([w0["W0"], jnp.zeros((w0["W0"].shape[0], L))], 1)
        ks = jax.random.split(key, 2)
        pi["E"], pi["bE"] = he(ks[0], L, NZ, 0.5), jnp.zeros(L)
    pi["log_sigma"] = jnp.full(4, np.log(a.sigma0), jnp.float32)
    vd = OD + (NZ if a.mode == "teacher" else 0)
    ks = jax.random.split(jax.random.PRNGKey(a.seed + 1), 3)
    vf = dict(W0=he(ks[0], 256, vd), b0=jnp.zeros(256), W1=he(ks[1], 256, 256), b1=jnp.zeros(256),
              Wo=he(ks[2], 1, 256, 0.1), bo=jnp.zeros(1))


def folded(p):
    """The deployable actor: the linear encoder folded into the first layer (input [o, z])."""
    w = {k: v for k, v in p.items() if k not in ("E", "bE", "log_sigma")}
    if "E" in p:
        Wo_, Wl = p["W0"][:, :OD], p["W0"][:, OD:]
        w["W0"] = jnp.concatenate([Wo_, Wl @ p["E"]], 1)
        w["b0"] = p["b0"] + Wl @ p["bE"]
    return w


def mean_act(p, o):
    return JumpEnv.actor(folded(p), o)


def value(v, o):
    h = jax.nn.relu(v["W0"] @ o + v["b0"])
    h = jax.nn.relu(v["W1"] @ h + v["b1"])
    return (v["Wo"] @ h + v["bo"])[0]


def adam_init(p):
    z = jax.tree_util.tree_map(jnp.zeros_like, p)
    return (z, z, 0)


def adam_step(p, g, st, lr):
    m, v, t = st
    t = t + 1
    m = jax.tree_util.tree_map(lambda m_, g_: 0.9 * m_ + 0.1 * g_, m, g)
    v = jax.tree_util.tree_map(lambda v_, g_: 0.999 * v_ + 0.001 * g_ ** 2, v, g)
    p = jax.tree_util.tree_map(lambda q, m_, v_: q - lr * (m_ / (1 - 0.9 ** t)) / (jnp.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8),
                               p, m, v)
    return p, (m, v, t)


def logp(p, o, act, mk):
    mu = jax.vmap(lambda o_: mean_act(p, o_))(o)
    ls = p["log_sigma"]
    z = (act - mu) / jnp.exp(ls)
    return ((-0.5 * z ** 2 - ls - 0.9189385) * mk).sum(-1)


@jax.jit
def ppo_update(p, v, sp, sv, O, Ov, A_, LP, ADV, RET, MK, key, train_actor):
    n = O.shape[0]
    nb = max(1, n // a.minibatch)

    def ploss(p_, o, act, lp0, adv, mk):
        r = jnp.exp(logp(p_, o, act, mk) - lp0)
        return -jnp.minimum(r * adv, jnp.clip(r, 1 - a.clip, 1 + a.clip) * adv).mean()

    def vloss(v_, o, ret):
        return ((jax.vmap(lambda o_: value(v_, o_))(o) - ret) ** 2).mean()

    def epoch(c, k):
        p, v, sp, sv = c
        perm = jax.random.permutation(k, n)

        def mb(c2, i):
            p, v, sp, sv = c2
            idx = jax.lax.dynamic_slice(perm, (i * a.minibatch,), (a.minibatch,))
            lp, gp = jax.value_and_grad(ploss)(p, O[idx], A_[idx], LP[idx], ADV[idx], MK[idx])
            lv, gv = jax.value_and_grad(vloss)(v, Ov[idx], RET[idx])
            p2, sp2 = adam_step(p, gp, sp, a.lr)
            p = jax.tree_util.tree_map(lambda x, y: jnp.where(train_actor, y, x), p, p2)
            sp = jax.tree_util.tree_map(lambda x, y: jnp.where(train_actor, y, x), sp, sp2)
            v, sv = adam_step(v, gv, sv, a.vlr)
            return (p, v, sp, sv), (lp, lv)
        c2, ls = jax.lax.scan(mb, (p, v, sp, sv), jnp.arange(nb))
        return c2, ls
    (p, v, sp, sv), ls = jax.lax.scan(epoch, (p, v, sp, sv), jax.random.split(key, a.epochs))
    return p, v, sp, sv, ls[0].mean(), ls[1].mean()


def targets_of(goals, ref):
    xr = np.asarray(ref["x_ref"])
    tg = xr[:, N].copy()
    tg[:, :2] = xr[:, 0, :2] + goals
    tg[:, 2] = 0.0
    return tg


def score(X, fell, tg):
    e = X[:, N] - tg
    return 0.01 * r_scale * (e * qe * e).sum(1) + 20.0 * fell


def export(w_, d, extra=None):
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "policy.npz"), **{k: np.asarray(v) for k, v in w_.items()}, n_hidden=np.array(2))
    for f in ("bank.json", "ilc_U.npz", "ilc_best.json"):
        if os.path.exists(os.path.join(pdir0, f)):
            shutil.copy(os.path.join(pdir0, f), d)
    json.dump(dict(best="policy", eval_goals=None, members=[dict(name="policy", group=0, variant="policy", tag="policy",
                                                                 episodes=0, updates=0, score=None)], **(extra or {})),
              open(os.path.join(d, "manifest.json"), "w"))


def deploy_weights():
    if a.mode == "student":
        return dict({k: v for k, v in T.items() if k not in ("E", "bE", "log_sigma")}, **phi)
    return folded(pi)


EB = 128
e_goals = np.stack([np.linspace(*a.goals, EB), np.zeros(EB)], 1)
e_ref = env.references(e_goals)
e_tg = targets_of(e_goals, e_ref)
e_dyn = env.dyn_arrays(env.sample_dyn(EB, np.random.default_rng(12345)))
n_goals = np.stack([np.linspace(*a.goals, 16), np.zeros(16)], 1)
n_ref = env.references(n_goals)
n_tg = targets_of(n_goals, n_ref)
n_dyn = env.dyn_arrays(np.tile([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0], (16, 1)))


def evaluate():
    w_ = deploy_weights()
    env.priv_dim = NZ if a.mode == "teacher" else 0
    o = env.rollout(w_, e_ref, jax.random.PRNGKey(0), stochastic=False, dyn=e_dyn)
    sd, fr = score(o["X"], o["fell"], e_tg).mean(), o["fell"].mean()
    o = env.rollout(w_, n_ref, jax.random.PRNGKey(0), stochastic=False, dyn=n_dyn)
    return float(sd), float(fr), float(score(o["X"], o["fell"], n_tg).mean())


if a.mode != "student":
    sp, sv = adam_init(pi), adam_init(vf)
else:
    sphi = adam_init(phi)

    @jax.jit
    def phi_fit(phi, st, O, Lt, k):
        n = O.shape[0]

        def one(c, kk):
            phi, st = c
            idx = jax.random.randint(kk, (a.minibatch,), 0, n)
            l, g = jax.value_and_grad(lambda q: ((jax.vmap(lambda o_: JumpEnv.planner(q, o_))(O[idx]) - Lt[idx]) ** 2).mean())(phi)
            phi, st = adam_step(phi, g, st, 3e-4)
            return (phi, st), l
        (phi, st), ls = jax.lax.scan(one, (phi, st), jax.random.split(k, 200))
        return phi, st, ls.mean()

t_start, t_train, jumps = time.time(), 0.0, 0
for it in range(a.iters + 1):
    if it % a.eval_every == 0 or it == a.iters:
        sd, fr, sn = evaluate()
        extra = dict(dyn_dim=NZ if a.mode == "teacher" else 0)
        export(deploy_weights(), os.path.join(a.out, f"it{it}", "policy"), extra)
        if a.mode == "teacher":
            np.savez(os.path.join(a.out, f"it{it}", "teacher_params.npz"), **{k: np.asarray(v) for k, v in pi.items()})
        say(f"it {it}: eval DR {sd:.2f} (falls {fr:.1%}), nominal {sn:.2f}  [{jumps} jumps, wall "
            f"{time.time() - t_start:.0f} s, train {t_train:.0f} s]")
    if it == a.iters:
        break
    t0 = time.time()
    B = a.batch
    goals = np.stack([rng.uniform(*a.goals, B), np.zeros(B)], 1)
    ref = env.references(goals)
    dyn = env.dyn_arrays(env.sample_dyn(B, rng))
    q_off, _ = env.explore_noise(B, rng, a.explore_q, 0.0)
    if a.mode == "student":                          # RMA phase 2: the student flies, phi regresses the latent
        env.priv_dim = 0
        out = env.rollout(deploy_weights(), ref, jax.random.PRNGKey(it), stochastic=False, q_offset=q_off, dyn=dyn)
        ok = np.isfinite(out["obs"]).all((1, 2))
        O = jnp.asarray(out["obs"][ok][:, :Nc].reshape(-1, OD), jnp.float32)
        lt = np.asarray(dyn["z"]) @ np.asarray(T["E"]).T + np.asarray(T["bE"])
        Lt = jnp.asarray(np.repeat(lt[ok], Nc, 0), jnp.float32)
        phi, sphi, lf = phi_fit(phi, sphi, O, Lt, jax.random.PRNGKey(10_000 + it))
        jumps += B
        t_train += time.time() - t0
        say(f"  it {it}: latent mse {float(lf):.4f} (latent var {float(Lt.var()):.3f})  [{time.time() - t0:.1f} s]")
        continue
    sig = np.exp(np.asarray(pi["log_sigma"]))
    a_off = rng.standard_normal((B, Nc, 4)) * sig * mask
    env.priv_dim = NZ if a.mode == "teacher" else 0
    out = env.rollout(folded(pi), ref, jax.random.PRNGKey(it), stochastic=False, a_offset=a_off, q_offset=q_off, dyn=dyn)
    ok = np.isfinite(out["X"]).all((1, 2)) & np.isfinite(out["obs"]).all((1, 2)) & np.isfinite(out["mu"]).all((1, 2))
    tg = targets_of(goals, ref)
    sc = np.where(ok, score(np.nan_to_num(out["X"]), out["fell"], tg), 0.0)
    Ob = np.nan_to_num(out["obs"])                                  # (B, Nc+1, od)
    err2 = (Ob[:, 1:Nc + 1, :6] ** 2).mean(-1)                      # the next sample's normalized error
    rew = -a.beta * err2
    rew[:, -1] -= sc
    act = out["mu"] + a_off                                         # the action taken (before the force clip)
    Ov = Ob[:, :Nc, :OD + (NZ if a.mode == "teacher" else 0)]
    V = np.asarray(jax.jit(jax.vmap(lambda o_: value(vf, o_)))(jnp.asarray(Ov.reshape(-1, Ov.shape[-1]), jnp.float32))
                   ).reshape(B, Nc)
    adv = np.zeros((B, Nc))
    g_ = np.zeros(B)
    for k in range(Nc - 1, -1, -1):
        nv = V[:, k + 1] if k < Nc - 1 else 0.0
        delta = rew[:, k] + a.gamma * nv - V[:, k]
        g_ = delta + a.gamma * a.lam * g_
        adv[:, k] = g_
    ret = adv + V
    adv = (adv - adv[ok].mean()) / (adv[ok].std() + 1e-8)
    f32 = lambda x: jnp.asarray(x[ok].reshape((-1,) + x.shape[2:]), jnp.float32)
    O_, Ov_, A_, ADV, RET = f32(Ob[:, :Nc]), f32(Ov), f32(act), f32(adv), f32(ret)
    MK = jnp.asarray(np.tile(mask, (int(ok.sum()), 1)), jnp.float32)
    LP = jax.jit(logp)(pi, O_, A_, MK)
    pi_new, vf_new, sp_new, sv_new, lp, lv = ppo_update(pi, vf, sp, sv, O_, Ov_, A_, LP, ADV, RET, MK,
                                                         jax.random.PRNGKey(20_000 + it), it >= a.value_warmup)
    if all(bool(jnp.isfinite(x).all()) for x in jax.tree_util.tree_leaves((pi_new, vf_new))):
        pi, vf, sp, sv = pi_new, vf_new, sp_new, sv_new
    else:
        say(f"  it {it}: non-finite update skipped")
    jumps += B
    t_train += time.time() - t0
    say(f"  it {it}: score {sc[ok].mean():.2f} (falls {int(out['fell'].sum())}), sigma "
        f"{' '.join(f'{s:.3f}' for s in np.exp(np.asarray(pi['log_sigma'])))}, L_pi {float(lp):+.4f} L_v {float(lv):.2f}"
        f"  [{time.time() - t0:.1f} s]")
say(f"done: {jumps} DR jumps, train time {t_train:.0f} s (wall {time.time() - t_start:.0f} s)")
