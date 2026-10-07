#!/usr/bin/env python3
"""
plane_train.py --bank BANK --out DIR: deep ILC over the 2D goal plane in the nominal GPU sim -- one network that jumps
to any goal (x, h) in the plans' hull (h = 0 flat, h > 0 onto a box), no domain randomization.

Every iteration (one ILC iteration amortized by the network, as dilc_train's coilc arm):
  1. a batch of goals uniform over the hull; part of the batch explores (its own stance, planar joints ~ N(0, q_off),
     and smooth action offsets of rms a_off: not DR, the dynamics stay nominal), the rest flies clean
  2. the batch flown by the network with the sim's one-step FD Jacobians A_k, B_k along every jump
  3. per jump the Gauss-Newton ILC step on its landing error e = (x, z, pitch) / sx through the CLOSED-loop landing
     sensitivity S = d x_N / d a (A_k + B_k K_k, K_k = d pi / d x the network's own feedback at the jump's states,
     and the ballistic flight): step = beta S'(SS' + delta tr/3 I)^-1 e, rms capped at cap; with --stages /
     --stage-floor also JumpILC's Stage I / II tracking rows (the SRB states against the plan's on the contact
     samples / from rear-leg contact through the flight), scheduled as its stages
  4. the targets a*_k = mu_k - step_k at the jump's observations (mu the network's mean: the noise is a disturbance
     the network learns to answer, not an action to copy); none from a fall, a non-finite jump, a jump whose leg or
     trunk hit the box before touchdown, or a landing error past max_err (all outside the step's linear regime)
  5. the network regresses onto the targets of the last `replay` iterations (Adam)
Optional warm start (--init-ilc): the network first cloned onto per-goal ILC solutions (plane_ilc.py --out), flown
open loop. Evaluated every --eval-every iterations on a goal grid over the hull (clean and from perturbed stances);
the snapshots go to DIR/policy as dilc_train exports them (npz + manifest + bank.json), for deploy.py / dilc_execute.
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--bank", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--init-ilc", default=None, help="plane_ilc.py --out json: clone its per-goal solutions first")
ap.add_argument("--init-npz", default=None, help="start from these network weights")
ap.add_argument("--iters", type=int, default=150)
ap.add_argument("--batch", type=int, default=96)
ap.add_argument("--explore-frac", type=float, default=0.5)
ap.add_argument("--q-off", type=float, default=0.08)
ap.add_argument("--a-off", type=float, default=0.05)
ap.add_argument("--beta", type=float, default=0.5)
ap.add_argument("--cap", type=float, default=0.1)
ap.add_argument("--delta", type=float, default=0.05)
ap.add_argument("--max-err", type=float, default=0.25, help="m: a landing error past this gives no landing rows")
ap.add_argument("--rate-w", type=float, default=0.0, help="rad/s: a 4th landing row, the touchdown pitch rate against "
                                                         "the plan's, this rate counting as the landing's 1 cm (0: none)")
ap.add_argument("--clear", type=float, default=0.0, help="m: the legs' and trunk's clearance to the box the ILC keeps "
                                                        "(0: no clearance rows)")
ap.add_argument("--stages", type=float, nargs=2, default=[0.0, 0.0],
                help="JumpILC's stage schedule (Nguyen et al.): the share of --iters where Stage II, then Stage III takes "
                     "over -- Stage I tracks the plan's SRB states on the contact samples, Stage II from rear-leg contact "
                     "through the flight, Stage III the landing against the goal (0 0: Stage III from the start)")
ap.add_argument("--stage-floor", type=float, default=0.0, help="row scale kept on the other stages' rows (0: none)")
ap.add_argument("--track-stride", type=int, default=2, help="a Stage I / II tracking row block every this many samples")
ap.add_argument("--vg", type=int, default=0,
                help="value-gradient critic for the landing rows: 0 none (the jump's own FD co-states); 1 the critic's "
                     "co-states on the flown batch; 2 also targets at the last --vg-pool iterations' states (the "
                     "critic's predicted landing error as the residual, the current network as the base)")
ap.add_argument("--vg-warm", type=int, default=3, help="iterations on the FD co-states before the critic's take over")
ap.add_argument("--vg-replay", type=int, default=10, help="iterations of co-state targets the critic fits")
ap.add_argument("--vg-steps", type=int, default=500)
ap.add_argument("--vg-lr", type=float, default=3e-4)
ap.add_argument("--vg-hidden", type=int, default=256)
ap.add_argument("--vg-pool", type=int, default=5, help="--vg 2: iterations of states that get critic targets")
ap.add_argument("--replay", type=int, default=6)
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--mb", type=int, default=2048)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--lr-final", type=float, default=None, help="the learning rate decays linearly to this by --iters")
ap.add_argument("--anchor-pool", type=int, default=0, help="iterations of observations the trust region holds (0: none)")
ap.add_argument("--anchor-w", type=float, default=1.0, help="the trust region's weight: on the pool's observations the "
                                                         "network held at its outputs before this iteration's fit")
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--eval-every", type=int, default=10)
ap.add_argument("--eval-grid", type=float, default=0.025)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--est-window", type=float, default=0.0, help="s: the policy observes the robots' estimator (jump.Config.est_window); 0: the true state")
ap.add_argument("--gpu", default=None)
a = ap.parse_args()
if a.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ilc_mjx.jump import Config, JumpEnv  # noqa: E402

os.makedirs(os.path.join(a.out, "policy"), exist_ok=True)
LOG = open(os.path.join(a.out, "train.log"), "a")


def log(s):
    s = f"[{time.strftime('%H:%M:%S')}] {s}"
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


bank = json.load(open(a.bank))
shutil.copy(a.bank, os.path.join(a.out, "policy", "bank.json"))
shutil.copy(a.bank, os.path.join(a.out, "bank.json"))
env = JumpEnv(bank, Config(est_window=a.est_window))
N, Nc = env.N, env.Nc
sx = np.asarray(bank["sx"], float)
swing = np.asarray(env.swing)
amask = (~swing).astype(np.float32)                                   # (Nc, 4)
od = 9 + 6 * env.H + (4 if env.PA else 0)
rng = np.random.default_rng(a.seed)
Gp = env.goals
lo, hi = Gp.min(0), Gp.max(0)
Tf = env.Nfl * env.dt
Phi = np.eye(6)
Phi[:3, 3:] = Tf * np.eye(3)
Pn = Phi[:3] / sx[:3, None]                                           # d e_n / d x_Nc


def in_hull(G):
    return env.weights(G).min(1) > -1e-9


def sample_goals(n):
    out = []
    while len(out) < n:
        c = lo + rng.random((4 * n, 2)) * (hi - lo)
        out += list(c[in_hull(c)])
    return np.array(out[:n])


def targets_of(G, ref):
    xr = np.asarray(ref["x_ref"])
    tg = xr[:, N, :3].copy()
    tg[:, :2] = xr[:, 0, :2] + G
    tg[:, 2] = 0.0
    return tg


# -- the network (dilc_train's actor: 2 relu layers -> tanh mean) ---------------------------------------------------
def init_w(key):
    k = jax.random.split(key, 3)
    H = a.hidden
    return {"W0": jax.random.normal(k[0], (H, od)) / np.sqrt(od), "b0": jnp.zeros(H),
            "W1": jax.random.normal(k[1], (H, H)) / np.sqrt(H), "b1": jnp.zeros(H),
            "W_mu": 1e-3 * jax.random.normal(k[2], (4, H)), "b_mu": jnp.zeros(4),
            "W_ls": jnp.zeros((4, H)), "b_ls": jnp.full(4, -3.0)}


if a.init_npz:
    w = {k: jnp.asarray(v, jnp.float32) for k, v in np.load(a.init_npz).items() if k != "n_hidden"}
else:
    w = init_w(jax.random.PRNGKey(a.seed))
TRAIN = ("W0", "b0", "W1", "b1", "W_mu", "b_mu")
mean_fn = jax.jit(jax.vmap(lambda w_, o: JumpEnv.actor(w_, o), (None, 0)))
jac_fn = jax.jit(jax.vmap(jax.jacfwd(lambda o, w_: JumpEnv.actor(w_, o)), (0, None)))


def loss_fn(p, w_, o, t, m):
    ww = dict(w_, **p)
    y = jax.vmap(lambda x: JumpEnv.actor(ww, x))(o)
    return (((y - t) ** 2) * m).sum() / jnp.maximum(m.sum(), 1.0)


@jax.jit
def adam_step(p, s, w_, o, t, m, lr):
    l, g = jax.value_and_grad(loss_fn)(p, w_, o, t, m)
    n = s["n"] + 1
    mo = jax.tree_util.tree_map(lambda m_, g_: 0.9 * m_ + 0.1 * g_, s["m"], g)
    v = jax.tree_util.tree_map(lambda v_, g_: 0.999 * v_ + 0.001 * g_ * g_, s["v"], g)
    p = jax.tree_util.tree_map(lambda p_, m_, v_: p_ - lr * (m_ / (1 - 0.9 ** n)) / (jnp.sqrt(v_ / (1 - 0.999 ** n)) + 1e-8),
                               p, mo, v)
    return p, dict(n=n, m=mo, v=v), l


params = {k: w[k] for k in TRAIN}
opt = dict(n=0, m=jax.tree_util.tree_map(jnp.zeros_like, params), v=jax.tree_util.tree_map(jnp.zeros_like, params))


def fit(O, T, M, steps, lr=None, anchor=None):
    """Adam on the targets (O, T, M); anchor (obs, outputs, mask): the trust region, half of every minibatch from it,
    weighted --anchor-w (the network held where this iteration brings no targets)."""
    global params, opt
    O, T, M = (jnp.asarray(x, jnp.float32) for x in (O, T, M))
    if anchor is not None:
        AO, AT, AM = (jnp.asarray(x, jnp.float32) for x in anchor)
        AM = AM * a.anchor_w
    ls = []
    for _ in range(steps):
        i = jnp.asarray(rng.integers(0, len(O), min(a.mb, len(O))))
        o_, t_, m_ = O[i], T[i], M[i]
        if anchor is not None:
            j = jnp.asarray(rng.integers(0, len(AO), min(a.mb, len(AO))))
            o_, t_, m_ = jnp.concatenate([o_, AO[j]]), jnp.concatenate([t_, AT[j]]), jnp.concatenate([m_, AM[j]])
        p_new, o_new, l = adam_step(params, opt, w, o_, t_, m_, a.lr if lr is None else lr)
        if not np.isfinite(float(l)) or not all(bool(jnp.isfinite(v).all()) for v in p_new.values()):
            continue                                  # a non-finite step is skipped, the network kept
        params, opt = p_new, o_new
        ls.append(float(l))
    w.update(params)
    return float(np.mean(ls[-20:])) if ls else float("nan")


# -- the value-gradient critic (--vg): at a sample's observation, the predicted landing error e_n (scaled) and its
# closed-loop co-states d e_n / d (x_k / sx) (3 x 6) -- the value and its gradient, over states, phases and goals.
# Fitted to the flown jumps' exact closed-loop co-states (the FD chain backward: Lam_k = Lam_k+1 (A_k + B_k K_k), Lam_Nc
# the ballistic flight's) and landing errors -- Monte Carlo targets, never its own outputs (no bootstrapping loop).
def crit_init(key):
    k = jax.random.split(key, 3)
    H = a.vg_hidden
    return {"W0": jax.random.normal(k[0], (H, od)) / np.sqrt(od), "b0": jnp.zeros(H),
            "W1": jax.random.normal(k[1], (H, H)) / np.sqrt(H), "b1": jnp.zeros(H),
            "W2": 1e-3 * jax.random.normal(k[2], (21, H)), "b2": jnp.zeros(21)}


def crit(p, o):
    h = jax.nn.relu(p["W0"] @ o + p["b0"])
    h = jax.nn.relu(p["W1"] @ h + p["b1"])
    return p["W2"] @ h + p["b2"]


crit_fn = jax.jit(jax.vmap(crit, (None, 0)))


def crit_loss(p, o, t, m):
    d = jax.vmap(lambda x: crit(p, x))(o) - t            # in the targets' robust units (cscale): Huber
    hub = jnp.where(jnp.abs(d) < 1.0, 0.5 * d ** 2, jnp.abs(d) - 0.5)
    return (hub * m).sum() / jnp.maximum(m.sum(), 1.0)


@jax.jit
def crit_step(p, s, o, t, m, lr):
    l, g = jax.value_and_grad(crit_loss)(p, o, t, m)
    n = s["n"] + 1
    mo = jax.tree_util.tree_map(lambda m_, g_: 0.9 * m_ + 0.1 * g_, s["m"], g)
    v = jax.tree_util.tree_map(lambda v_, g_: 0.999 * v_ + 0.001 * g_ * g_, s["v"], g)
    p = jax.tree_util.tree_map(lambda p_, m_, v_: p_ - lr * (m_ / (1 - 0.9 ** n)) / (jnp.sqrt(v_ / (1 - 0.999 ** n)) + 1e-8),
                               p, mo, v)
    return p, dict(n=n, m=mo, v=v), l


cparams = crit_init(jax.random.PRNGKey(a.seed + 1000))
copt = dict(n=0, m=jax.tree_util.tree_map(jnp.zeros_like, cparams), v=jax.tree_util.tree_map(jnp.zeros_like, cparams))


cscale = None          # per output, the targets' robust scale (1.4826 MAD), fixed at the first fit
CO_OUT = 30.0          # a sample whose co-states pass this many robust scales: its gradient targets dropped (a chain
                       # exploding backward through an unstable closed loop -- a few such swamp the fit)


def crit_fit(O, T, M):
    """The critic on its replay, targets in robust units -> (loss, share of samples whose gradient targets were dropped)."""
    global cparams, copt, cscale
    if cscale is None:
        sc = []
        for j in range(T.shape[1]):
            t_ = T[M[:, j] > 0, j]
            sc.append(1.4826 * np.median(np.abs(t_ - np.median(t_))) if len(t_) else 1.0)
        cscale = np.maximum(np.array(sc), 1e-3)
    T = T / cscale
    out = (np.abs(T[:, 3:]) * M[:, 3:]).max(1) > CO_OUT
    M = M.copy()
    M[out, 3:] = 0.0
    O, T, M = (jnp.asarray(x, jnp.float32) for x in (O, T, M))
    ls = []
    for _ in range(a.vg_steps):
        i = jnp.asarray(rng.integers(0, len(O), min(a.mb, len(O))))
        p_new, o_new, l = crit_step(cparams, copt, O[i], T[i], M[i], a.vg_lr)
        if not np.isfinite(float(l)):
            continue
        cparams, copt = p_new, o_new
        ls.append(float(l))
    return (float(np.mean(ls[-20:])) if ls else float("nan")), float(out.mean())


def crit_pred(O):
    """(n, Nc+1, od) -> the predicted landing error (n, Nc+1, 3) and co-states d e_n / d x_k (n, Nc+1, 3, 6)."""
    C = np.asarray(crit_fn(cparams, jnp.asarray(O.reshape(-1, od), jnp.float32))).reshape(O.shape[:2] + (21,))
    C = C * (1.0 if cscale is None else cscale)
    return C[..., :3], C[..., 3:].reshape(O.shape[:2] + (3, 6)) / sx[None, None, None, :]


def land_rows(Lm, Bm):
    """Co-states (n, Nc+1, 3, 6) and control Jacobians (n, Nc, 6, 4) -> landing rows d e_n / d a (n, 3, Nc*4):
    d e_n / d a_k = Lam_k+1 B_k."""
    r = np.einsum("bkij,bkjl->bkil", Lm[:, 1:], Bm) * amask[None, :, None, :]
    return r.transpose(0, 2, 1, 3).reshape(len(Lm), 3, Nc * 4)


def gn_step(rows, res, keep):
    """The damped Gauss-Newton step on the stacked rows (n, r, Nc*4) -> (step (n, Nc, 4), row mask (n, r))."""
    R = np.concatenate(rows, 1)
    r_ = np.concatenate(res, 1)
    kp = np.concatenate(keep, 1).astype(float)
    R, r_ = R * kp[..., None], np.nan_to_num(r_) * kp
    Mm = R @ R.transpose(0, 2, 1)
    nr = np.maximum(kp.sum(1), 1.0)
    # Levenberg-Marquardt as before (delta x the mean active diagonal); an inactive row gets 1 on its diagonal (its
    # row and residual are 0, so it takes no part)
    Mm = Mm + a.delta * np.trace(Mm, axis1=1, axis2=2)[:, None, None] / nr[:, None, None] * np.eye(R.shape[1]) \
        + 1e-9 * np.eye(R.shape[1]) + np.einsum("bi,ij->bij", 1.0 - kp, np.eye(R.shape[1]))
    stp = (R.transpose(0, 2, 1) @ np.linalg.solve(Mm, r_[..., None]))[..., 0].reshape(len(R), Nc, 4) * a.beta
    rms = np.sqrt((stp ** 2).sum((1, 2)) / amask.sum())
    stp *= np.minimum(1.0, a.cap / np.maximum(rms, 1e-12))[:, None, None]
    return stp, kp


def export(tag):
    path = os.path.join(a.out, "policy", f"g0_plane_{tag}.npz")
    np.savez(path, **{k: np.asarray(v) for k, v in w.items()}, n_hidden=np.array(2))
    return os.path.basename(path)[:-4]


# -- evaluation: a goal grid over the hull --------------------------------------------------------------------------
xs = np.arange(lo[0], hi[0] + 1e-9, a.eval_grid)
hs = np.arange(0.0, hi[1] + 1e-9, a.eval_grid)
GE = np.array([(x, h) for h in hs for x in xs])
GE = GE[in_hull(GE)]
REF_E = env.references(GE)
TG_E = targets_of(GE, REF_E)
QOFF_E = env.explore_noise(len(GE), np.random.default_rng(5), 0.08, 0.0)[0]


def evaluate():
    res = {}
    for name, q in (("clean", None), ("perturbed", QOFF_E)):
        o = env.rollout(w, REF_E, jax.random.PRNGKey(0), stochastic=False, q_offset=q)
        e = o["X"][:, N, :3] - TG_E
        f = np.asarray(o["fell"], bool) | ~np.isfinite(e).all(1)
        e = np.nan_to_num(e, nan=1.0)
        ok = (~f) & (np.abs(e[:, 0]) <= 0.05) & (np.abs(e[:, 1]) <= 0.03)
        res[name] = dict(ok=float(ok.mean()), fell=float(f.mean()), hit=float(np.asarray(o["hit"], bool).mean()), ex=float(np.abs(e[~f, 0]).mean()) if (~f).any() else 1.0,
                         ez=float(np.abs(e[~f, 1]).mean()) if (~f).any() else 1.0,
                         flat_ok=float(ok[GE[:, 1] < 0.005].mean()), box_ok=float(ok[GE[:, 1] >= 0.005].mean()))
    return res


# -- warm start: clone per-goal ILC solutions -----------------------------------------------------------------------
if a.init_ilc:
    sol = json.load(open(a.init_ilc))["rows"]
    sol = [r for r in sol if not r["hist"][-1]["fell"] and r["best"] < 20]
    G0 = np.array([r["goal"] for r in sol])
    U0 = np.array([r["a"] for r in sol])
    ref0 = env.references(G0)
    zero = {k: (jnp.zeros_like(v) if k in ("W_mu", "b_mu") else v) for k, v in w.items()}
    O, T, M = [], [], []
    for q in (None, *[env.explore_noise(len(G0), rng, a.q_off, 0.0)[0] for _ in range(3)]):
        o = env.rollout(zero, ref0, jax.random.PRNGKey(0), stochastic=False, a_offset=U0, q_offset=q)
        O.append(o["obs"][:, :Nc].reshape(-1, od))
        T.append(U0.reshape(-1, 4))
        M.append(np.broadcast_to(amask, U0.shape).reshape(-1, 4))
    l = fit(np.concatenate(O), np.concatenate(T), np.concatenate(M), 4000)
    log(f"warm start: cloned {len(sol)} per-goal ILC solutions (4 stances each), loss {l:.2e}")

hist_json = os.path.join(a.out, "evals.json")
evals = json.load(open(hist_json)) if os.path.exists(hist_json) else []
replay = []
crep, vgpool = [], []                         # --vg: the critic's co-state targets; --vg 2: the states it steps
pool = []
members = []
t0 = time.time()
for it in range(a.iters + 1):
    if it % a.eval_every == 0:
        r = evaluate()
        name = export(f"it{it}")
        members.append(dict(name=name, it=it, eval=r))
        evals.append(dict(it=it, **r))
        json.dump(evals, open(hist_json, "w"), indent=0)
        sc = lambda m: m["eval"]["clean"]["ok"] + m["eval"]["perturbed"]["ok"]
        man = dict(best=max(members, key=sc)["name"], members=members, eval_goals=GE.tolist())
        json.dump(man, open(os.path.join(a.out, "policy", "manifest.json"), "w"), indent=1)
        c, p = r["clean"], r["perturbed"]
        log(f"eval it {it}: clean ok {c['ok']:.0%} (flat {c['flat_ok']:.0%} box {c['box_ok']:.0%}) fell {c['fell']:.0%} hit {c['hit']:.0%} "
            f"|ex| {c['ex'] * 100:.1f} |ez| {c['ez'] * 100:.1f} cm; perturbed ok {p['ok']:.0%} fell {p['fell']:.0%} "
            f"|ex| {p['ex'] * 100:.1f} cm  ({len(GE)} goals, {time.time() - t0:.0f} s)")
    if it == a.iters:
        break
    B = a.batch
    G = sample_goals(B)
    ref = env.references(G)
    tg = targets_of(G, ref)
    nx = int(round(a.explore_frac * B))
    q_off, a_off = env.explore_noise(B, rng, a.q_off, a.a_off)
    q_off[nx:], a_off[nx:] = 0.0, 0.0
    o = env.rollout(w, ref, jax.random.PRNGKey(it), stochastic=False, fd=True, a_offset=a_off, q_offset=q_off)
    e = o["X"][:, N, :3] - tg
    fell = np.asarray(o["fell"], bool)
    en = np.nan_to_num(e) / sx[:3]
    obs = o["obs"][:, :Nc]                                             # (B, Nc, od)
    K = np.asarray(jac_fn(jnp.asarray(obs.reshape(-1, od), jnp.float32), w)).reshape(B, Nc, 4, od)[..., :6]
    K = K / sx[None, None, None, :] * amask[None, :, :, None]         # d a / d x (SRB units)
    A_, B_ = np.nan_to_num(o["A"]), np.nan_to_num(o["B"])
    Acl = A_ + B_ @ K                                                  # (B, Nc, 6, 6)
    # forward closed-loop sensitivities D_k = d x_k / d a (B, 6, Nc*4), every contact sample and takeoff
    D = np.zeros((B, Nc + 1, 6, Nc * 4))
    for k in range(Nc):
        D[:, k + 1] = Acl[:, k] @ D[:, k]
        D[:, k + 1, :, 4 * k:4 * k + 4] += B_[:, k] * amask[k][None, None]
    rows, res, keep = [], [], []
    # the stage's row scales (--stages): 1 on this iteration's stage, --stage-floor on the other two
    p_it = it / max(a.iters, 1)
    stage = 0 if p_it < a.stages[0] else 1 if p_it < a.stages[1] else 2
    cst = np.full(3, a.stage_floor)
    cst[stage] = 1.0
    # landing rows: e_n through the ballistic flight (none for a jump that hit the box: its landing is the hit's)
    hit = np.asarray(o["hit"], bool)
    land_ok = (~fell) & (~hit) & np.isfinite(e).all(1) & (np.abs(np.nan_to_num(e[:, 0], nan=9.0)) <= a.max_err) \
        & (np.abs(np.nan_to_num(e[:, 1], nan=9.0)) <= a.max_err)
    # a jump far off (past max_err, or fallen with its landing past it) and not hit: its landing as the takeoff state
    # predicts it (the CoM is ballistic in flight: exact for x, z whatever the landing met), x and z rows only
    Xnc = np.asarray(o["X"])[:, Nc]
    pred = np.stack([Xnc[:, 0] + Tf * Xnc[:, 3], Xnc[:, 1] + Tf * Xnc[:, 4] - 0.5 * 9.81 * Tf ** 2], 1) - tg[:, :2]
    far = (~hit) & (~land_ok) & np.isfinite(pred).all(1) & (np.abs(np.nan_to_num(e[:, :2], nan=9.0)).max(1) > a.max_err)
    e_use = np.where(far[:, None], np.concatenate([np.nan_to_num(pred) / sx[:2], np.zeros((B, 1))], 1), en)
    R_land = np.einsum("ij,bjk->bik", Pn, D[:, Nc])                       # (B, 3, Nc*4): the jump's own FD co-states
    use_vg = a.vg > 0 and it >= a.vg_warm
    if a.vg:
        Lam = np.zeros((B, Nc + 1, 3, 6))                                  # exact closed-loop co-states, backward
        Lam[:, Nc] = Pn
        for k in range(Nc - 1, -1, -1):
            Lam[:, k] = Lam[:, k + 1] @ Acl[:, k]
        obs_full = np.asarray(o["obs"])                                    # (B, Nc+1, od)
        _, Lh = crit_pred(obs_full)                                        # the critic's, before it sees this batch
        R_crit = land_rows(Lh, B_)
        if use_vg:
            R_land = R_crit
    rows.append(cst[2] * R_land)                                           # (B, 3, Nc*4)
    res.append(cst[2] * e_use)
    keep.append(np.stack([land_ok | far, land_ok | far, land_ok], 1) & (cst[2] > 0))
    if a.rate_w > 0:   # the touchdown pitch rate against the plan's (constant in the ballistic flight), per rate_w rad/s
        rows.append(cst[2] * D[:, Nc, 5:6] / a.rate_w)
        res.append(cst[2] * ((np.asarray(o["X"])[:, N, 5] - np.asarray(ref["x_ref"])[:, N, 5]) / a.rate_w)[:, None])
        keep.append(land_ok[:, None] & (cst[2] > 0))
    # tracking rows (JumpILC's Stages I / II): the jump's SRB state against the plan's, (x_k - x_ref,k) / sx, every
    # --track-stride samples -- Stage I on the contact samples, Stage II from rear-leg contact through the flight
    # (ballistic from takeoff) -- up to the jump's first hit, scaled by the stage's weight
    n_trk = 0
    if cst[0] > 0 or cst[1] > 0:
        Xs, xr = np.asarray(o["X"]), np.asarray(ref["x_ref"])
        hg = np.asarray(o["hit_geom"])
        first = np.where((hg[:, :N - 3] >= 0).any(1), np.argmax(hg[:, :N - 3] >= 0, 1), N - 3)
        fin = np.isfinite(o["A"]).all((1, 2, 3)) & np.isfinite(o["B"]).all((1, 2, 3)) & np.isfinite(Xs).all((1, 2))
        for s_, ks in ((0, range(a.track_stride, Nc + 1, a.track_stride)), (1, range(env.Ndc, N, a.track_stride))):
            if cst[s_] <= 0:
                continue
            for k in ks:
                Pk = np.eye(6)
                Pk[:3, 3:] = max(k - Nc, 0) * env.dt * np.eye(3)
                rows.append(cst[s_] * (Pk @ D[:, min(k, Nc)]) / sx[None, :, None])
                res.append(cst[s_] * np.nan_to_num(Xs[:, k] - xr[:, k]) / sx[None])
                act = fin & (k < first)
                keep.append(np.repeat(act[:, None], 6, 1))
                n_trk += 6 * int(act.sum())
    if it == 0:
        d_ = np.abs(pred - e[:, :2])[land_ok]
        log(f"ballistic landing prediction vs measured on clean jumps: |dx| {d_[:, 0].mean() * 100:.2f} cm, "
            f"|dz| {d_[:, 1].mean() * 100:.2f} cm")
    # clearance rows (--clear > 0): the legs' and trunk's least clearance to the box before the jump's first hit, in
    # single stance and in flight: below --clear, a row pushing it back up to --clear (scaled like the landing's cm)
    n_clr = 0
    if a.clear > 0 and "clr" in o:
        hg = np.asarray(o["hit_geom"])
        first = np.where((hg[:, :N - 3] >= 0).any(1), np.argmax(hg[:, :N - 3] >= 0, 1), N - 3)
        clr, cg = np.nan_to_num(np.asarray(o["clr"]), nan=1.0), np.nan_to_num(np.asarray(o["clr_grad"]))
        for lo_, hi_ in ((env.Ndc, Nc), (Nc, N - 3)):
            ks = np.arange(lo_, hi_)
            c_ = np.where(ks[None] <= first[:, None], clr[:, lo_:hi_], 1.0)
            j = np.argmin(c_, 1)
            k = ks[j]
            cmin = c_[np.arange(B), j]
            # a hit in this window: its row at the hit sample, the clearance there taken as at most 0 (the sample's
            # start can still read clear: the contact comes within the sample)
            hw = (first >= lo_) & (first < hi_) & (first < N - 3)
            k = np.where(hw, first, k)
            cmin = np.where(hw, np.minimum(clr[np.arange(B), np.minimum(first, N - 1)], 0.0), cmin)
            act = ((cmin < a.clear) | hw) & np.isfinite(o["A"]).all((1, 2, 3))
            Dc = D[np.arange(B), np.minimum(k, Nc)]                          # (B, 6, Nc*4)
            tf = (np.maximum(k - Nc, 0) * env.dt)[:, None, None]               # in flight: ballistic from takeoff
            Dk = Dc[:, :3] + tf * Dc[:, 3:]
            g = cg[np.arange(B), k]                                       # (B, 3)
            rows.append((np.einsum("bi,bik->bk", g, Dk) / 0.01)[:, None])
            res.append(((cmin - a.clear) / 0.01)[:, None])
            keep.append(act[:, None])
            n_clr += int(act.sum())
    stp, kp = gn_step(rows, res, keep)
    vg_msg = ""
    if a.vg:
        lv = (land_ok | far) & np.isfinite(o["A"]).all((1, 2, 3)) & np.isfinite(o["B"]).all((1, 2, 3))
        # how well the critic (not yet fitted to this batch: held-out states and goals) predicts the jumps' landing rows,
        # and how close its step comes to the exact one
        Rx = np.einsum("ij,bjk->bik", Pn, D[:, Nc])
        if it == 0:
            log(f"co-state check: backward Lam rows vs forward D rows, max |diff| "
                f"{np.abs(land_rows(Lam, B_) - Rx)[lv].max():.2e} (|rows| max {np.abs(Rx[lv]).max():.2e})")
        r_rows = float(np.corrcoef(R_crit[lv].ravel(), Rx[lv].ravel())[0, 1]) if lv.sum() > 1 else float("nan")
        stp_ex, _ = gn_step([cst[2] * Rx] + rows[1:], res, keep) if use_vg else (stp, None)
        num, den = (stp * stp_ex).sum((1, 2)), np.linalg.norm(stp, axis=(1, 2)) * np.linalg.norm(stp_ex, axis=(1, 2))
        cos = float(np.mean((num / np.maximum(den, 1e-12))[lv])) if lv.any() else float("nan")
        # the critic's targets from this batch: value (the landing error) and gradient (Lam_k sx), every sample
        tv = np.repeat(np.clip(e_use, -10.0, 10.0)[:, None], Nc + 1, 1)
        tl = (Lam * sx[None, None, None, :]).reshape(B, Nc + 1, 18)
        mv = np.stack([land_ok | far, land_ok | far, land_ok], 1)[:, None].repeat(Nc + 1, 1).astype(float)
        mg = np.repeat(lv[:, None, None], Nc + 1, 1) * np.ones((1, 1, 18))
        fin = np.isfinite(obs_full).all((1, 2)) & np.isfinite(tl).all((1, 2))
        crep.append((obs_full[fin].reshape(-1, od), np.concatenate([tv, tl], 2)[fin].reshape(-1, 21),
                     np.concatenate([mv, mg], 2)[fin].reshape(-1, 21)))
        crep = crep[-a.vg_replay:]
        cl_, cout = crit_fit(*(np.concatenate([c[j] for c in crep]) for j in range(3)))
        if a.vg == 2:
            vgpool.append((obs_full[fin & lv], B_[fin & lv]))
            vgpool = vgpool[-a.vg_pool:]
        vg_msg = f" | vg {'on' if use_vg else 'warm'}: crit loss {cl_:.3f} (dropped {cout:.1%}), rows r {r_rows:.2f}, step cos {cos:.2f}"
    # targets from a jump with any row: its landing (clean jumps) or its clearance (also a jump that hit the box)
    good = (kp.sum(1) > 0) & np.isfinite(o["A"]).all((1, 2, 3)) & np.isfinite(o["B"]).all((1, 2, 3))
    tgt = np.clip(o["mu"] - stp, -0.999, 0.999) * amask
    good &= np.isfinite(tgt).all((1, 2)) & np.isfinite(obs).all((1, 2))     # a non-finite jump never reaches the fit
    gi = np.where(good)[0]
    replay.append((obs[gi].reshape(-1, od), tgt[gi].reshape(-1, 4), np.broadcast_to(amask, (len(gi), Nc, 4)).reshape(-1, 4)))
    replay = replay[-a.replay:]
    lr_it = a.lr if a.lr_final is None else a.lr + (a.lr_final - a.lr) * it / max(a.iters - 1, 1)
    anc = None
    if a.anchor_pool > 0:
        pool.append(obs.reshape(-1, od))
        pool = pool[-a.anchor_pool:]
        PO = np.concatenate(pool)
        PO = PO[rng.choice(len(PO), min(len(PO), 60000), replace=False)]
        PT = np.asarray(mean_fn(w, jnp.asarray(PO, jnp.float32)))
        tcol = PO[:, 6]                                               # the obs' time: 2 k / Nc - 1
        kk = np.clip(np.round((tcol + 1) * Nc / 2).astype(int), 0, Nc - 1)
        anc = (PO, PT, amask[kk])
    data = list(replay)
    n_pool = 0
    if a.vg == 2 and use_vg:
        # value-gradient targets at the last --vg-pool iterations' states: the critic's co-states for the rows, its
        # predicted landing error at the jump's start as the residual, the CURRENT network's mean as the base
        for Op, Bp in vgpool:
            ev, Lp = crit_pred(Op)
            stp_p, _ = gn_step([cst[2] * land_rows(Lp, Bp)], [cst[2] * np.clip(ev[:, 0], -10.0, 10.0)],
                               [np.ones((len(Op), 3), bool) & (cst[2] > 0)])
            mu_p = np.asarray(mean_fn(w, jnp.asarray(Op[:, :Nc].reshape(-1, od), jnp.float32))).reshape(len(Op), Nc, 4)
            tg_p = np.clip(mu_p - stp_p, -0.999, 0.999) * amask
            ok_p = np.isfinite(tg_p).all((1, 2))
            data.append((Op[ok_p, :Nc].reshape(-1, od), tg_p[ok_p].reshape(-1, 4),
                         np.broadcast_to(amask, (int(ok_p.sum()), Nc, 4)).reshape(-1, 4)))
            n_pool += int(ok_p.sum())
        vg_msg += f", pool targets {n_pool}"
    l = fit(*(np.concatenate([r_[j] for r_ in data]) for j in range(3)), a.steps, lr_it, anc)
    # how much of this iteration's ILC step the network took: the projection of its change onto the step
    if len(gi):
        mu_new = np.asarray(mean_fn(w, jnp.asarray(obs[gi].reshape(-1, od), jnp.float32))).reshape(len(gi), Nc, 4) * amask
        d_ = (o["mu"][gi] - mu_new)
        real = float((d_ * stp[gi]).sum() / max((stp[gi] ** 2).sum(), 1e-12))
    else:
        real = float("nan")
    ok = (~fell) & (np.abs(np.nan_to_num(e[:, 0], nan=9)) <= 0.05) & (np.abs(np.nan_to_num(e[:, 1], nan=9)) <= 0.03)
    cl = slice(nx, None)
    log(f"it {it:3d}: stage {'I' * (stage + 1)} batch ok {ok.mean():.0%} (clean {ok[cl].mean():.0%}) fell {fell.mean():.0%} hit {hit.mean():.0%} targets {good.mean():.0%} clr rows {n_clr} trk rows {n_trk} far {int(far.sum())} "
        f"|ex| {np.abs(np.nan_to_num(e[~fell, 0])).mean() * 100:.1f} cm  step rms {np.sqrt((stp[gi] ** 2).sum((1, 2)) / amask.sum()).mean():.3f} "
        f"loss {l:.2e} step taken {real:.0%}{vg_msg}  {time.time() - t0:.0f} s")
log("done")
