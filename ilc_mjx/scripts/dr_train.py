#!/usr/bin/env python3
"""
dr_train.py --init POLICY.npz --mode plain|teacher|student --out DIR: the domain-randomization BASELINES on the GPU
sim (our method never randomizes). Every robot of a batch draws its own dynamics from JumpEnv.sample_dyn -- the
dynamics part of the family the real-like robots are drawn from; delays, jitter, mocap and sensing stay out (the
GPU sim cannot model them), so they are the unseen part of the real robots.

  plain    the deep-ILC update of our sim stage (Stage III: each jump's Gauss-Newton step on its landing error
           through the closed-loop landing sensitivity from the FD Jacobians, the network regressing onto the
           stepped actions), under DR: one robust policy
  teacher  the same, the actor also reading the robot's normalized dynamics z (privileged: an oracle at test time)
  student  RMA-style: the deployable history-conditioned actor (its obs already carry 3 past errors and the last
           action) distilled from --teacher by DAgger -- the student flies, the teacher labels its states with z

Each starts from --init (our nominal sim policy; the teacher's extra inputs start at zero weight). Writes
OUT/it<k>/policy (dilc_execute / deploy.py format; the teacher's carries dyn_dim) and OUT/train.log, with the
GPU wall time (compute to report next to our sim stage's).
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True, help="policy npz to start from (our nominal sim policy)")
ap.add_argument("--bank", default="", help="bank.json (default: next to --init)")
ap.add_argument("--mode", required=True, choices=("plain", "teacher", "student"))
ap.add_argument("--teacher", default="", help="student: the teacher's policy npz")
ap.add_argument("--out", required=True)
ap.add_argument("--iters", type=int, default=100)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--explore", type=float, nargs=2, default=[0.08, 0.15])
ap.add_argument("--goals", type=float, nargs=2, default=[0.40, 0.60])
ap.add_argument("--gn-beta", type=float, default=0.5)
ap.add_argument("--gn-delta", type=float, default=0.1)
ap.add_argument("--step-rms", type=float, default=0.08)
ap.add_argument("--replay", type=int, default=8, help="batches kept for the regression")
ap.add_argument("--fit-steps", type=int, default=200)
ap.add_argument("--minibatch", type=int, default=4096)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--eval-every", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--nominal", action="store_true", help="no DR (a control: our sim stage's update, continued)")
ap.add_argument("--width", type=int, default=0, help="student: a fresh actor of this hidden width (the width ablation)")
ap.add_argument("--gpu", default="0")
ap.add_argument("--plane", action="store_true", help="2D goal plane: goals uniform over the bank's valid region (flat and box jumps), evals on its grid")
ap.add_argument("--est-window", type=float, default=0.0, help="s: the policy observes the robots' estimator (jump.Config)")
a = ap.parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ilc_mjx.jump import Config, JumpEnv  # noqa: E402

os.makedirs(a.out, exist_ok=True)
pdir0 = os.path.dirname(os.path.abspath(a.init))
bank = json.load(open(a.bank or os.path.join(pdir0, "bank.json")))
env = JumpEnv(bank, Config(est_window=a.est_window))
Nc, Ndc, N = env.Nc, env.Ndc, env.N
D = np.asarray(bank["sx"], float)
qe = np.asarray(bank["qe"], float)
r_scale = float(bank["r_scale"])
mask = np.ones((Nc, 4))
mask[Ndc:, :2] = 0.0
NZ = len(JumpEnv.DYN_KEYS)
logf = open(os.path.join(a.out, "train.log"), "a")


def say(msg):
    print(msg, flush=True)
    logf.write(msg + "\n")
    logf.flush()


def load(path, extra=0):
    w = {k: jnp.asarray(v, jnp.float32) for k, v in np.load(path).items() if k != "n_hidden"}
    if extra and w["W0"].shape[1] == 31:
        w["W0"] = jnp.concatenate([w["W0"], jnp.zeros((w["W0"].shape[0], extra), jnp.float32)], 1)
    return w


w = load(a.init, NZ if a.mode == "teacher" else 0)
wt = load(a.teacher) if a.mode == "student" else None
T_PRIV = wt is not None and wt["W0"].shape[1] > 31          # the teacher reads z (DR); else a plain distillation
if a.width:
    ks = jax.random.split(jax.random.PRNGKey(a.seed), 4)
    he = lambda k, o, i: jax.random.normal(k, (o, i)) * jnp.sqrt(2.0 / i)
    od = w["W0"].shape[1]
    w = dict(W0=he(ks[0], a.width, od), b0=jnp.zeros(a.width), W1=he(ks[1], a.width, a.width), b1=jnp.zeros(a.width),
             W_mu=he(ks[2], 4, a.width) * 0.1, b_mu=jnp.zeros(4), W_ls=he(ks[3], 4, a.width) * 0.1,
             b_ls=jnp.full(4, -3.0))
actor_b = jax.jit(jax.vmap(lambda w_, o: JumpEnv.actor(w_, o), (None, 0)))
dpi_b = jax.jit(jax.vmap(jax.jacfwd(lambda o, w_: JumpEnv.actor(w_, o)), (0, None)))

# the ballistic flight map takeoff -> touchdown (exact for the CoM): x_N = x_Nc + T v_Nc
T = (N - Nc) * env.dt
PHI = np.zeros((3, 6))
PHI[:, :3] = np.eye(3)
PHI[:, 3:] = T * np.eye(3)
Mrow = jnp.asarray((PHI * D[None]).T, jnp.float32)            # (6, 3): d x_N,i / d e_Nc (normalized)
mask_j = jnp.asarray(mask, jnp.float32)


@jax.jit
def landing_sens(An, Bn, K):
    """S (3, Nc, 4) of one jump: d (landing x, z, pitch) / d a_k, closed loop with the policy's feedback K."""
    Acl = An + jnp.einsum("kij,kjl->kil", Bn, K)               # (Nc, 6, 6): A_k + B_k K_k

    def back(m, k):
        m = jnp.where(k < Nc - 1, Acl[jnp.minimum(k + 1, Nc - 1)].T @ m, m)
        return m, (Bn[k].T @ m) * mask_j[k][:, None]
    _, S = jax.lax.scan(back, Mrow, jnp.arange(Nc - 1, -1, -1))
    return jnp.transpose(S[::-1], (2, 0, 1))                   # (3, Nc, 4)


@jax.jit
def gn_steps(A, B, K, e3):
    An = A * jnp.asarray(D)[None, None, None, :] / jnp.asarray(D)[None, None, :, None]
    Bn = B / jnp.asarray(D)[None, None, :, None]
    S = jax.vmap(landing_sens)(An, Bn, K).reshape(A.shape[0], 3, -1)
    M = jnp.einsum("bik,bjk->bij", S, S)
    lam = a.gn_delta * jnp.trace(M, axis1=1, axis2=2) / 3
    x = jnp.linalg.solve(M + lam[:, None, None] * jnp.eye(3)[None], e3[..., None])[..., 0]
    st = a.gn_beta * jnp.einsum("bik,bi->bk", S, x).reshape(A.shape[0], Nc, 4)
    rms = jnp.sqrt((st ** 2).sum((1, 2)) / mask.sum())
    return st * jnp.minimum(1.0, a.step_rms / jnp.maximum(rms, 1e-12))[:, None, None]


def targets_of(goals, ref):
    """The landing target (6,) of each jump: the goal's position, level, the plan's landing velocities."""
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


def make_fit():
    def loss(w_, O, Tt, Mk):
        return ((((jax.vmap(lambda o: JumpEnv.actor(w_, o))(O) - Tt) * Mk) ** 2).sum(-1)).mean()
    vg = jax.value_and_grad(loss)

    @jax.jit
    def fit(w_, mo, vo, step0, O, Tt, Mk, key):
        n = O.shape[0]

        def one(c, s):
            w_, mo, vo, key = c
            key, sub = jax.random.split(key)
            idx = jax.random.randint(sub, (a.minibatch,), 0, n)
            l, g = vg(w_, O[idx], Tt[idx], Mk[idx])
            t = step0 + s + 1
            mo = jax.tree_util.tree_map(lambda m_, g_: 0.9 * m_ + 0.1 * g_, mo, g)
            vo = jax.tree_util.tree_map(lambda v_, g_: 0.999 * v_ + 0.001 * g_ ** 2, vo, g)
            w_ = jax.tree_util.tree_map(lambda p, m_, v_: p - a.lr * (m_ / (1 - 0.9 ** t))
                                        / (jnp.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8), w_, mo, vo)
            return (w_, mo, vo, key), l
        (w_, mo, vo, key), ls = jax.lax.scan(one, (w_, mo, vo, key), jnp.arange(a.fit_steps))
        return w_, mo, vo, ls.mean()
    return fit


fit = make_fit()
mo = jax.tree_util.tree_map(jnp.zeros_like, w)
vo = jax.tree_util.tree_map(jnp.zeros_like, w)
rng = np.random.default_rng(a.seed)

# a fixed evaluation set: 128 DR robots x goals over the range, and the nominal robot at 16 goals
erng = np.random.default_rng(12345)
EB = 128
e_goals = (env.sample_plane_goals(EB, np.random.default_rng(777)) if a.plane else np.stack([np.linspace(a.goals[0], a.goals[1], EB), np.zeros(EB)], 1))
e_ref = env.references(e_goals)
e_tg = targets_of(e_goals, e_ref)
e_dyn = env.dyn_arrays(env.sample_dyn(EB, erng))
n_goals = (env.plane_grid(0.05) if a.plane else np.stack([np.linspace(a.goals[0], a.goals[1], 16), np.zeros(16)], 1))
n_ref = env.references(n_goals)
n_tg = targets_of(n_goals, n_ref)
n_dyn = env.dyn_arrays(np.tile([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0], (len(n_goals), 1)))


def evaluate(w_):
    env.priv_dim = NZ if a.mode == "teacher" else 0
    o = env.rollout(w_, e_ref, jax.random.PRNGKey(0), stochastic=False, dyn=e_dyn)
    sd = score(o["X"], o["fell"], e_tg)
    fd_ = float(o["fell"].mean())
    o = env.rollout(w_, n_ref, jax.random.PRNGKey(0), stochastic=False, dyn=n_dyn)
    sn = score(o["X"], o["fell"], n_tg)
    return float(sd.mean()), fd_, float(sn.mean())


nominal_dyn = lambda B: np.tile([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0], (B, 1))
replay = []
t_start = time.time()
gpu_s = 0.0
for it in range(a.iters + 1):
    if it % a.eval_every == 0 or it == a.iters:
        sd, fdr, sn = evaluate(w)
        export(w, os.path.join(a.out, f"it{it}", "policy"), dict(dyn_dim=NZ if a.mode == "teacher" else 0))
        say(f"it {it}: eval DR {sd:.2f} (falls {fdr:.1%}), nominal {sn:.2f}  [wall {time.time() - t_start:.0f} s, train {gpu_s:.0f} s]")
    if it == a.iters:
        break
    t0 = time.time()
    B = a.batch
    goals = env.sample_plane_goals(B, rng) if a.plane else np.stack([rng.uniform(*a.goals, B), np.zeros(B)], 1)
    ref = env.references(goals)
    P = nominal_dyn(B) if a.nominal else env.sample_dyn(B, rng)
    dyn = env.dyn_arrays(P)
    q_off, a_off = env.explore_noise(B, rng, *a.explore)
    key = jax.random.PRNGKey(int(rng.integers(2 ** 31)))
    if a.mode == "student":
        env.priv_dim = 0
        out = env.rollout(w, ref, key, stochastic=False, fd=False, a_offset=a_off, q_offset=q_off, dyn=dyn)
        O = jnp.asarray(out["obs"][:, :Nc], jnp.float32)                         # (B, Nc, 31)
        Oz = jnp.concatenate([O, jnp.repeat(dyn["z"][:, None], Nc, 1)], -1) if T_PRIV else O
        tgt = np.asarray(actor_b(wt, Oz.reshape(-1, Oz.shape[-1]))).reshape(B, Nc, 4) * mask
        keep = np.isfinite(np.asarray(O)).all((1, 2)) & np.isfinite(tgt).all((1, 2))
        info = f"teacher-labelled {B}"
    else:
        env.priv_dim = NZ if a.mode == "teacher" else 0
        out = env.rollout(w, ref, key, stochastic=False, fd=True, a_offset=a_off, q_offset=q_off, dyn=dyn)
        O = jnp.asarray(out["obs"][:, :Nc], jnp.float32)
        K = np.asarray(dpi_b(O.reshape(-1, O.shape[-1]), w)).reshape(B, Nc, 4, -1)[..., :6]
        tg = targets_of(goals, ref)
        e3 = out["X"][:, N, :3] - tg[:, :3]
        ok = np.isfinite(out["A"]).all((1, 2, 3)) & np.isfinite(out["B"]).all((1, 2, 3)) & ~out["fell"] \
            & np.isfinite(out["X"]).all((1, 2)) & np.isfinite(out["mu"]).all((1, 2)) & np.isfinite(out["obs"]).all((1, 2))
        A_ = np.where(ok[:, None, None, None], out["A"], 0.0)
        B_ = np.where(ok[:, None, None, None], out["B"], 0.0)
        e3 = np.where(ok[:, None], e3, 0.0)
        K = np.where(ok[:, None, None, None], np.nan_to_num(K), 0.0)
        st = np.asarray(gn_steps(jnp.asarray(A_, jnp.float32), jnp.asarray(B_, jnp.float32),
                                 jnp.asarray(K, jnp.float32), jnp.asarray(e3, jnp.float32)))
        ok &= np.isfinite(st).all((1, 2))
        tgt = np.clip((out["mu"] - st) * mask, -0.999, 0.999)
        keep = ok
        sc = score(out["X"], out["fell"], tg)
        info = f"train J {np.nanmean(sc):.2f} (falls {int(out['fell'].sum())}, used {ok.mean():.2f})"
    if keep.any():
        replay.append((np.asarray(O)[keep].reshape(-1, O.shape[-1]), tgt[keep].reshape(-1, 4)))
    replay = replay[-a.replay:]
    Ob = jnp.asarray(np.concatenate([r[0] for r in replay]), jnp.float32)
    Tb = jnp.asarray(np.concatenate([r[1] for r in replay]), jnp.float32)
    Mb = jnp.asarray(np.tile(mask, (len(Ob) // Nc, 1)), jnp.float32)
    w_new, mo_new, vo_new, lf = fit(w, mo, vo, it * a.fit_steps, Ob, Tb, Mb, jax.random.PRNGKey(it))
    if all(bool(jnp.isfinite(v).all()) for v in w_new.values()):          # never keep a non-finite update
        w, mo, vo = w_new, mo_new, vo_new
    else:
        say(f"  it {it}: non-finite update skipped")
    dt_ = time.time() - t0
    gpu_s += dt_
    say(f"  it {it}: {info}, fit {float(lf):.2e}  [{dt_:.1f} s]")
say(f"done: {a.iters} iterations x {a.batch} jumps = {a.iters * a.batch} DR jumps, train time {gpu_s:.0f} s "
    f"(wall {time.time() - t_start:.0f} s)")
