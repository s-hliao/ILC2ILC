#!/usr/bin/env python3
"""
quad_real_ppo.py --policy DIR --out OUT: the RL baselines trained ON THE ROBOTS with privileged data (the data-matching
study, user 2026-10-09: how many real jumps does a baseline need to match our best model after the hardware stage --
our learner + DR (B) + 24 jumps, 43.8 % on the reserved test goals, robots r1 s1 r5?). From the baseline's sim-trained
policy, PPO on each CPU robot itself:

  rollouts  the executor's serve workers (dilc_execute.py serve, --workers in the container, the box-fix overlay):
            each jump flies the robot's current policy STOCHASTICALLY -- the executor's own tanh-Gaussian,
            a = tanh(mu(o) + sigma eps) -- on one of the 6 training goals (the hardware stage's; never a test goal)
  reward    per contact sample -c_dense x its stage cost, and at the end -(0.01 x landing cost + 20 x fall): the
            evaluation's score
  critic    privileged: the observation, the robot's TRUE normalized dynamics z (jump.JumpEnv.dyn_from_cond of its
            condition string) and the sample's phase
  PPO       clipped surrogate, GAE; per robot its own actor / critic; sigma per channel, learned

--policy a PPO+DR deploy dir (base/ppo_plain/itK/policy): fine-tuned as is.
--policy an RMA teacher deploy dir (base/ppo_teacher/itK/policy: input [o, z]): each robot's TRUE z folded into the first
layer -- the oracle latent, the best RMA's adaptation could estimate -- then fine-tuned the same way.
Checkpoints at --checkpoints jumps per robot -> OUT/ck<K>/<robot>/policy (deploy format), then each evaluated on the
reserved test goals (planefinal: 8 goals x --eval-episodes, seed 701, deterministic) -> OUT/summary.json.
Run from src/ilc_mjx with the ilcmjx env; the ilc_quad container must be up.
"""
import argparse
import io
import json
import os
import queue
import shutil
import struct
import subprocess
import sys
import threading
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--robots", nargs="+", default=["real_r1", "real_s1", "real_r5"])
ap.add_argument("--goals", nargs="+", default=["0.45", "0.60", "0.50,0.15", "0.60,0.10", "0.50,0.05", "0.575,0.15"])
ap.add_argument("--per-iter", type=int, default=24, help="jumps per robot per PPO iteration")
ap.add_argument("--checkpoints", type=int, nargs="+", default=[24, 48, 96, 192, 384, 768, 1536, 2016])
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--sigma", type=float, default=0.1, help="initial exploration std (pre-tanh), per channel")
ap.add_argument("--c-dense", type=float, default=0.01)
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--mb", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--lr-critic", type=float, default=1e-3)
ap.add_argument("--critic-warmup", type=int, default=300)
ap.add_argument("--gamma", type=float, default=0.99)
ap.add_argument("--lam", type=float, default=0.95)
ap.add_argument("--clip", type=float, default=0.2)
ap.add_argument("--eval-episodes", type=int, default=4)
ap.add_argument("--seed", type=int, default=6060)
ap.add_argument("--overlay", default="/ilc_ws/log/dilc/plane/fixbox/overlay")
ap.add_argument("--eval-only", action="store_true", help="skip training: evaluate the checkpoints in OUT")
a = ap.parse_args()
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                       # noqa: E402
import jax.numpy as jnp                          # noqa: E402


class _Adam:
    """Adam (the ilcmjx env has no optax), with optional global-norm clipping."""

    def __init__(self, lr, clip=None, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.clip, self.b1, self.b2, self.eps = lr, clip, b1, b2, eps

    def init(self, p):
        z = jax.tree_util.tree_map(jnp.zeros_like, p)
        return (z, z, jnp.zeros((), jnp.int32))

    def update(self, g, st, p):
        m, v, t = st
        if self.clip:
            n = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g)))
            g = jax.tree_util.tree_map(lambda x: x * jnp.minimum(1.0, self.clip / (n + 1e-9)), g)
        t = t + 1
        m = jax.tree_util.tree_map(lambda m_, g_: self.b1 * m_ + (1 - self.b1) * g_, m, g)
        v = jax.tree_util.tree_map(lambda v_, g_: self.b2 * v_ + (1 - self.b2) * g_ ** 2, v, g)
        mh = jax.tree_util.tree_map(lambda m_: m_ / (1 - self.b1 ** t), m)
        vh = jax.tree_util.tree_map(lambda v_: v_ / (1 - self.b2 ** t), v)
        u = jax.tree_util.tree_map(lambda a_, b_: -self.lr * a_ / (jnp.sqrt(b_) + self.eps), mh, vh)
        return u, (m, v, t)


def apply_updates(p, u):
    return jax.tree_util.tree_map(lambda a_, b_: a_ + b_, p, u)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "ilc_quad", "scripts"))
import dilc_execute as dx                        # noqa: E402  (the robots' condition strings; no ROS at import)
from ilc_mjx.jump import JumpEnv                 # noqa: E402

WS, CWS = "/home/henry/ilc_ws", "/ilc_ws"
cpath = lambda p: CWS + os.path.abspath(p)[len(WS):]
gvec = lambda g: tuple(float(v) for v in g.split(",")) if "," in g else (float(g), 0.0)
GOALS = [gvec(g) for g in a.goals]
TEST = [(0.4875, 0.0), (0.6125, 0.0), (0.52, 0.12), (0.54, 0.14), (0.57, 0.11), (0.51, 0.17), (0.4875, 0.035),
        (0.6375, 0.09)]
XC = (f"cd /ilc_ws && source install/setup.bash && export ROS_LOCALHOST_ONLY=1 OMP_NUM_THREADS=1 && "
      f"export PYTHONPATH={a.overlay}:$PYTHONPATH && python3 install/ilc_quad/lib/ilc_quad/dilc_execute.py")
BANK = json.load(open(os.path.join(a.policy, "bank.json")))
NDC, NSC = BANK["phases"][:2]
NC = NDC + NSC
OD = 31
MASK = np.ones((NC, 4), np.float32)
MASK[NDC:, :2] = 0.0


def true_z(robot):
    raw = JumpEnv.dyn_from_cond(None, dx.NAMED[robot])
    return ((raw - JumpEnv.DYN_CENTER) / JumpEnv.DYN_SCALE).astype(np.float32)


def load_actor(robot):
    d = dict(np.load(os.path.join(a.policy, "policy.npz")))
    w = {k: np.asarray(d[k], np.float32) for k in ("W0", "b0", "W1", "b1", "W_mu", "b_mu")}
    if w["W0"].shape[1] > OD:                    # an RMA teacher, input [o, z]: fold this robot's true z
        w["b0"] = w["b0"] + w["W0"][:, OD:] @ true_z(robot)
        w["W0"] = w["W0"][:, :OD]
    return {k: jnp.array(v) for k, v in w.items()}


def mu_fn(w, o):
    h = jax.nn.relu(o @ w["W0"].T + w["b0"])
    h = jax.nn.relu(h @ w["W1"].T + w["b1"])
    return h @ w["W_mu"].T + w["b_mu"]


def export(w, log_sigma, d):
    """A deploy dir: the executor's NumpyActor (relu MLP, tanh-Gaussian: W_ls = 0, b_ls = log sigma)."""
    os.makedirs(d, exist_ok=True)
    arr = {k: np.asarray(v, np.float64) for k, v in w.items()}
    arr.update(W_ls=np.zeros_like(arr["W_mu"]), b_ls=np.asarray(log_sigma, np.float64), n_hidden=np.array(2))
    np.savez(os.path.join(d, "policy.npz"), **arr)
    shutil.copy(os.path.join(a.policy, "bank.json"), d)
    json.dump(dict(best="policy", eval_goals=None, dyn_dim=0, members=[dict(name="policy", group=0, variant="policy",
                                                                             tag="policy", episodes=0, updates=0,
                                                                             score=None)]),
              open(os.path.join(d, "manifest.json"), "w"))
    return arr


# ---- the serve workers ------------------------------------------------------------------------------------------
def _msg(header, payload=None):
    hb = json.dumps(header).encode()
    pb = b""
    if payload is not None:
        buf = io.BytesIO()
        np.savez(buf, **payload)
        pb = buf.getvalue()
    return struct.pack("<I", len(hb)) + hb + struct.pack("<Q", len(pb)) + pb


def _read(f):
    h = f.read(4)
    if len(h) < 4:
        return None, None
    (n,) = struct.unpack("<I", h)
    header = json.loads(f.read(n))
    (m,) = struct.unpack("<Q", f.read(8))
    pb = f.read(m) if m else b""
    if not pb:
        return header, None
    with np.load(io.BytesIO(pb), allow_pickle=False) as d:
        return header, {k: d[k] for k in d.files}


class Workers:
    def __init__(self, n, bank_path):
        self.procs = []
        for i in range(n):
            cmd = f"{XC} serve --bank {cpath(bank_path)}"
            p = subprocess.Popen(["sg", "docker", "-c", f"docker exec -i ilc_quad bash -lc '{cmd}'"],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self.procs.append(p)

    def run(self, jobs):
        """jobs: [(header, payload)] -> [(header, episode)] in order, the workers in parallel."""
        q, out = queue.Queue(), [None] * len(jobs)
        for i, j in enumerate(jobs):
            q.put((i, j))

        def loop(p):
            while True:
                try:
                    i, (h, pl) = q.get_nowait()
                except queue.Empty:
                    return
                p.stdin.write(_msg(h, pl))
                p.stdin.flush()
                out[i] = _read(p.stdout)
        th = [threading.Thread(target=loop, args=(p,)) for p in self.procs]
        for t in th:
            t.start()
        for t in th:
            t.join()
        return out

    def close(self):
        for p in self.procs:
            try:
                p.stdin.write(_msg(dict(cmd="quit")))
                p.stdin.close()
            except Exception:
                pass
        for p in self.procs:
            p.wait(timeout=60)


def train():
    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    actors = {r: load_actor(r) for r in a.robots}
    log_sigma = {r: jnp.full(4, np.log(a.sigma), jnp.float32) for r in a.robots}
    zs = {r: true_z(r) for r in a.robots}

    def init_critic(k):
        k1, k2, k3 = jax.random.split(k, 3)
        nin = OD + len(JumpEnv.DYN_CENTER) + 1
        return dict(W0=jax.random.normal(k1, (256, nin)) / np.sqrt(nin), b0=jnp.zeros(256),
                    W1=jax.random.normal(k2, (256, 256)) / 16.0, b1=jnp.zeros(256),
                    Wo=jax.random.normal(k3, (1, 256)) / 16.0, bo=jnp.zeros(1))

    def v_fn(c, x):
        h = jnp.tanh(x @ c["W0"].T + c["b0"])
        h = jnp.tanh(h @ c["W1"].T + c["b1"])
        return (h @ c["Wo"].T + c["bo"])[..., 0]

    critics = {}
    for r in a.robots:
        key, k = jax.random.split(key)
        critics[r] = init_critic(k)
    opt_a = _Adam(a.lr, clip=1.0)
    opt_c = _Adam(a.lr_critic)
    st_a = {r: opt_a.init((actors[r], log_sigma[r])) for r in a.robots}
    st_c = {r: opt_c.init(critics[r]) for r in a.robots}

    @jax.jit
    def critic_step(c, st, x, ret):
        l, g = jax.value_and_grad(lambda q: jnp.mean((v_fn(q, x) - ret) ** 2))(c)
        u, st = opt_c.update(g, st, c)
        return apply_updates(c, u), st, l

    def logp(w, ls, o, u, m):
        mu = mu_fn(w, o)
        return jnp.sum(m * (-0.5 * ((u - mu) / jnp.exp(ls)) ** 2 - ls), -1)

    @jax.jit
    def actor_step(q, st, o, u, m, lp_old, adv):
        def loss(q):
            ratio = jnp.exp(logp(q[0], q[1], o, u, m) - lp_old)
            return -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - a.clip, 1 + a.clip) * adv))
        l, g = jax.value_and_grad(loss)(q)
        u_, st = opt_a.update(g, st, q)
        return apply_updates(q, u_), st, l

    for r in a.robots:
        export({k: np.asarray(v) for k, v in actors[r].items()}, np.asarray(log_sigma[r]), os.path.join(a.out, "ck0", r, "policy"))
    wk = Workers(a.workers, os.path.join(a.policy, "bank.json"))
    jumps = {r: 0 for r in a.robots}
    falls = {r: 0 for r in a.robots}
    hist = []
    done = {0}
    it = 0
    t0 = time.time()
    logf = open(os.path.join(a.out, "train.log"), "a")
    try:
        while min(jumps.values()) < max(a.checkpoints):
            jobs, who = [], []
            for r in a.robots:
                arr = {k: np.asarray(v, np.float64) for k, v in actors[r].items()}
                arr.update(W_ls=np.zeros_like(arr["W_mu"]), b_ls=np.asarray(log_sigma[r], np.float64), n_hidden=np.array(2))
                for k in range(a.per_iter):
                    g = GOALS[(it * a.per_iter + k) % len(GOALS)]
                    jobs.append((dict(goal=list(g), seed=int(rng.integers(1, 1 << 30)), cond=r, stochastic=True,
                                      id=len(jobs)), arr))
                    who.append(r)
            res = wk.run(jobs)
            for r in a.robots:
                eps = [ep for (h, ep), w_ in zip(res, who) if w_ == r and h is not None and h.get("ok") and ep is not None]
                jumps[r] += sum(1 for w_ in who if w_ == r)
                O, X, U, M, R = [], [], [], [], []
                scores = []
                for ep in eps:
                    obs = np.asarray(ep["obs"], np.float32)[:NC, :OD]
                    act = np.clip(np.asarray(ep["act"], np.float32)[:NC], -0.999, 0.999)
                    rc = np.asarray(ep["rc"], np.float32)
                    fell = float(ep["fell"])
                    rew = -a.c_dense * rc[:NC, 0]
                    final = 0.01 * float(rc[:, 2].sum()) + 20.0 * fell
                    rew[-1] -= final
                    scores.append(final)
                    falls[r] += int(fell)
                    O.append(obs)
                    X.append(np.c_[obs, np.broadcast_to(zs[r], (NC, len(zs[r]))), np.arange(NC)[:, None] / NC])
                    U.append(np.arctanh(act))
                    M.append(MASK)
                    R.append(rew)
                if not O:
                    continue
                advs, rets = [], []
                for x, rew in zip(X, R):                 # GAE within each jump (terminal at its end)
                    v = np.asarray(v_fn(critics[r], jnp.array(x, jnp.float32)))
                    vn = np.r_[v[1:], 0.0]
                    delta = rew + a.gamma * vn - v
                    adv, gg = np.zeros_like(delta), 0.0
                    for t in range(len(delta) - 1, -1, -1):
                        gg = delta[t] + a.gamma * a.lam * gg
                        adv[t] = gg
                    advs.append(adv)
                    rets.append(adv + v)
                Oj, Xj, Uj, Mj = (jnp.array(np.concatenate(z_), jnp.float32) for z_ in (O, X, U, M))
                ADV, RET = np.concatenate(advs), np.concatenate(rets)
                ADVj = jnp.array((ADV - ADV.mean()) / (ADV.std() + 1e-8), jnp.float32)
                RETj = jnp.array(RET, jnp.float32)
                if it == 0:
                    for s in range(a.critic_warmup):
                        b = jnp.array(rng.integers(0, len(RET), min(a.mb, len(RET))))
                        critics[r], st_c[r], _ = critic_step(critics[r], st_c[r], Xj[b], RETj[b])
                else:
                    lp_old = logp(actors[r], log_sigma[r], Oj, Uj, Mj)
                    n = len(RET)
                    for e in range(a.epochs):
                        perm = rng.permutation(n)
                        for s in range(0, n, a.mb):
                            b = jnp.array(perm[s:s + a.mb])
                            (actors[r], log_sigma[r]), st_a[r], _ = actor_step((actors[r], log_sigma[r]), st_a[r], Oj[b],
                                                                               Uj[b], Mj[b], lp_old[b], ADVj[b])
                            critics[r], st_c[r], _ = critic_step(critics[r], st_c[r], Xj[b], RETj[b])
                hist.append(dict(it=it, robot=r, jumps=jumps[r], score=float(np.mean(scores)), falls=int(sum(
                    float(ep["fell"]) for ep in eps)), n=len(eps), sigma=np.exp(np.asarray(log_sigma[r])).round(3).tolist()))
            it += 1
            line = dict(it=it, t=round(time.time() - t0), jumps=jumps, falls=falls,
                        score={h["robot"]: round(h["score"], 2) for h in hist if h["it"] == it - 1})
            print(json.dumps(line), flush=True)
            logf.write(json.dumps(line) + "\n")
            logf.flush()
            for k in a.checkpoints:
                if k not in done and min(jumps.values()) >= k:
                    for r in a.robots:
                        export({kk: np.asarray(v) for kk, v in actors[r].items()}, np.asarray(log_sigma[r]),
                               os.path.join(a.out, f"ck{k}", r, "policy"))
                    done.add(k)
    finally:
        wk.close()
    json.dump(dict(hist=hist, jumps=jumps, falls=falls, seconds=time.time() - t0),
              open(os.path.join(a.out, "train_summary.json"), "w"), indent=1)


def evaluate():
    """Every checkpoint on the reserved test goals, deterministic, as fada_adapt.py / deploy.py's planefinal."""
    jumps = " ".join(f"--jump {g[0]} {g[1]}" for g in TEST)
    res = {}
    for k in [0] + a.checkpoints:
        for r in a.robots:
            pdir = os.path.join(a.out, f"ck{k}", r, "policy")
            js = os.path.join(a.out, f"ck{k}", r, "eval_planefinal.json")
            if not os.path.exists(pdir):
                continue
            if not os.path.exists(js):
                cmd = (f"{XC} run --policy {cpath(pdir)} --member policy --conds {r} {jumps} --episodes {a.eval_episodes} "
                       f"--seed 701 --jobs {a.workers} --json {cpath(js)}")
                subprocess.run(["sg", "docker", "-c", f"docker exec ilc_quad bash -lc '{cmd}'"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600)
            rr = json.load(open(js)) if os.path.exists(js) else []
            ok = [(not x["fell"]) and abs(x["ex"]) <= 0.05 and abs(x["ez"]) <= 0.03 for x in rr]
            res.setdefault(str(k), {})[r] = dict(n=len(rr), success=float(np.mean(ok)) if rr else None,
                                                 falls=int(sum(x["fell"] for x in rr)))
        tot = [v for v in res.get(str(k), {}).values() if v["success"] is not None]
        if tot:
            p = sum(v["success"] * v["n"] for v in tot) / sum(v["n"] for v in tot)
            print(f"ck{k}: success {100 * p:.1f}% over {sum(v['n'] for v in tot)} test jumps", flush=True)
        json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    if not a.eval_only:
        train()
    evaluate()
