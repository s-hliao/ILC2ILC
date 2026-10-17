#!/usr/bin/env python3
"""
ppo_real.py --mode ppo|rma_oracle: the data-matching study (the car's ppo_real_car.py / the quadruped's
quad_real_ppo.py analog) -- how many REAL trials does RL need, trained ON each real pogo stick with privileged data,
to match our method?
  every iteration, per robot (all of real_pogo.DEFAULT_ROBOTS in one batched numpy simulation): --per-iter trials of
  --T periods from drops at goals uniform in the goal box (not the test goals), the robot's own stochastic actor;
  reward per period = ppo_dr's all-phase reward (1 - 0.1 min(|e / ERR_SCALE|^2, 10), e = [predicted apex from the
  vertical energy incl. the spring's, vx - v*]) computed with the ROBOT'S OWN parameters (privileged); a fall ends
  the trial. PPO (clip, GAE) per robot; the critic is privileged: it sees [obs, the robot's true z] (rma_np.true_z).
  --mode ppo         start: PPO+DR's actor and critic (runs/ppo_dr_s0; the critic's z inputs zero-initialized)
  --mode rma_oracle  start: the RMA teacher with the robot's TRUE z folded into its first-layer bias (the oracle
                     latent; then a plain observation->action network), and the teacher's critic
At checkpoints (--checkpoints, real trials per robot) each robot's deterministic actor is evaluated on the reserved
test goals (real_pogo.evaluate: 8 goals x 4 drops x 10 s of the POGO_EVAL_SET set) ->
runs/hw/match/{mode}_ep{N}/summary.json (+ <robot>_policy.npz, plain MLPs: policies.load_any).
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='ppo', choices=('ppo', 'rma_oracle'))
ap.add_argument('--ppo-dir', default='runs/ppo_dr_s0')
ap.add_argument('--rma-dir', default='runs/rma_s0')
ap.add_argument('--robots', nargs='+', default=None)
ap.add_argument('--checkpoints', type=int, nargs='+', default=[24, 96, 384, 768, 1536, 2000])
ap.add_argument('--per-iter', type=int, default=48, help='trials per robot per PPO iteration')
ap.add_argument('--T', type=int, default=300, help='periods per trial (3 s)')
ap.add_argument('--epochs', type=int, default=4)
ap.add_argument('--mb', type=int, default=4096)
ap.add_argument('--lr', type=float, default=3e-5)
ap.add_argument('--gamma', type=float, default=0.98)
ap.add_argument('--lam', type=float, default=0.95)
ap.add_argument('--clip', type=float, default=0.2)
ap.add_argument('--kl-stop', type=float, default=0.02, help='stop a robot\'s epochs once the approx KL (old -> new) of a minibatch exceeds this')
ap.add_argument('--log-std', type=float, default=-2.3, help='initial exploration log-std (None: the agent\'s)')
ap.add_argument('--eval-runs', type=int, default=4)
ap.add_argument('--eval-seconds', type=float, default=10.0)
ap.add_argument('--out', default='runs/hw/match')
ap.add_argument('--tag', default='', help='suffix of the output dirs ({mode}{tag}_ep{N})')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--gpu', default=None)
a = ap.parse_args()
if a.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax                                       # noqa: E402
import jax.numpy as jnp                          # noqa: E402
import numpy as np                               # noqa: E402
import optax                                     # noqa: E402

import common as C                               # noqa: E402
import real_pogo as rp                           # noqa: E402
import rma_np                                    # noqa: E402

ROBOTS = a.robots or rp.DEFAULT_ROBOTS
NR = len(ROBOTS)
rng = np.random.default_rng(a.seed)
T0 = time.time()
RMA_PATH = os.path.join(a.rma_dir, 'policy_final.npz')
PR = {r: rp.robot_params(r) for r in ROBOTS}
Z = {r: rma_np.true_z(PR[r], RMA_PATH) for r in ROBOTS}
NZ = len(next(iter(Z.values())))
NAME = f'{a.mode}{a.tag}'
LOG = open(os.path.join('logs', f'match_{NAME}.jsonl'), 'a')


def log(**kw):
    kw.update(t=round(time.time() - T0, 1))
    print(json.dumps(kw), flush=True)
    LOG.write(json.dumps(kw) + '\n')
    LOG.flush()


def critic_from(path, n_in):
    d = np.load(path, allow_pickle=True)
    n = len([k for k in d.files if k.startswith('cW')])
    cr = [(np.asarray(d[f'cW{i}']), np.asarray(d[f'cb{i}'])) for i in range(n)]
    W0 = cr[0][0]
    if W0.shape[0] < n_in:                        # PPO+DR's critic: zero rows for the privileged z inputs
        cr[0] = (np.concatenate([W0, np.zeros((n_in - W0.shape[0], W0.shape[1]))]), cr[0][1])
    return cr, np.asarray(d['log_std'])


def start_params(r):
    if a.mode == 'ppo':
        actor = C.load_policy(os.path.join(a.ppo_dir, 'policy_final.npz'))
        critic, ls = critic_from(os.path.join(a.ppo_dir, 'agent_final.npz'), C.NOBS + NZ)
    else:                                         # fold the robot's true z into the teacher's first layer
        t = C.load_policy(RMA_PATH)
        W0, b0 = t[0]
        actor = [(W0[:C.NOBS], b0 + Z[r] @ W0[C.NOBS:])] + t[1:]
        critic, ls = critic_from(os.path.join(a.rma_dir, 'agent_final.npz'), C.NOBS + NZ)
    if a.log_std is not None:
        ls = np.full(2, a.log_std)
    j = lambda ps: [(jnp.array(W, jnp.float32), jnp.array(b, jnp.float32)) for W, b in ps]
    return dict(actor=j(actor), critic=j(critic), log_std=jnp.array(ls, jnp.float32))


params = {r: start_params(r) for r in ROBOTS}
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
opt_state = {r: opt.init(params[r]) for r in ROBOTS}


def np_actor(r):
    return [(np.asarray(W, np.float64), np.asarray(b, np.float64)) for W, b in params[r]['actor']]


class PerRobot:
    """One batched simulation, each lane flown by its robot's actor."""
    def __init__(self, lane_robot):
        self.lr = lane_robot
        self.acts = [np_actor(r) for r in ROBOTS]

    def reset(self, n):
        pass

    def __call__(self, o, x, g):
        out = np.zeros((len(o), 2))
        for k in range(NR):
            m = self.lr == k
            out[m] = C.mlp(self.acts[k], o[m], np)
        return out


def reward_np(x, G, p):
    """ppo_dr's all-phase reward with the lanes' own parameters (incl. the progressive spring's energy)."""
    st = x[..., 5]
    L = C.leg_length(x, np)
    c = np.maximum(p['l0'][:, None] - L, 0.0) * st
    espr = p['k'][:, None] * (0.5 * c ** 2 + p['kp'][:, None] * c ** 3 / (3 * p['l0'][:, None]))
    apex = x[..., 1] + 0.5 * x[..., 3] ** 2 / C.GRAV + espr / (p['m'][:, None] * C.GRAV)
    e = np.stack([apex - G[:, None, 1], x[..., 2] - G[:, None, 0]], -1) / C.ERR_SCALE
    return 1.0 - 0.1 * np.minimum(np.sum(e ** 2, -1), 10.0)


mlp = lambda ps, o: C.mlp(ps, o, jnp)


@jax.jit
def gae(critic, obs_in, r, alive):
    v = mlp(critic, obs_in)[..., 0]
    v_next = jnp.concatenate([v[:, 1:], v[:, -1:]], 1) * jnp.concatenate([alive[:, 1:], alive[:, -1:]], 1)

    def body(adv, t):
        d = r[:, t] + a.gamma * v_next[:, t] - v[:, t]
        adv = d + a.gamma * a.lam * adv * alive[:, t]
        return adv, adv
    _, advs = jax.lax.scan(body, jnp.zeros(r.shape[0]), jnp.arange(r.shape[1])[::-1])
    adv = advs[::-1].T
    return adv, adv + v


@jax.jit
def update(ps, st, o, oc, act, logp_old, adv, ret, w):
    def loss(ps):
        mu = mlp(ps['actor'], o)
        logp = jnp.sum(-0.5 * ((act - mu) / jnp.exp(ps['log_std'])) ** 2 - ps['log_std'], -1)
        ratio = jnp.exp(logp - logp_old)
        an = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
        pl = -jnp.minimum(ratio * an, jnp.clip(ratio, 1 - a.clip, 1 + a.clip) * an)
        vl = (mlp(ps['critic'], oc)[:, 0] - ret) ** 2
        lr_ = logp - logp_old
        return jnp.sum(w * (pl + 0.5 * vl)) / jnp.sum(w), jnp.sum(w * (jnp.exp(lr_) - 1 - lr_)) / jnp.sum(w)
    (l, kl), g = jax.value_and_grad(loss, has_aux=True)(ps)
    u, st = opt.update(g, st, ps)
    return optax.apply_updates(ps, u), st, l, kl


def checkpoint(n_done):
    t = time.time()
    G = C.eval_goals(8)
    pols = {r: np_actor(r) for r in ROBOTS}
    res = rp.evaluate(pols, G, ROBOTS, runs=a.eval_runs, seconds=a.eval_seconds)
    out = os.path.join(a.out, f'{NAME}_ep{n_done}')
    os.makedirs(out, exist_ok=True)
    for r in ROBOTS:
        C.save_policy(os.path.join(out, f'{r}_policy.npz'), pols[r], kind='mlp')
    summ = {r: rp.summarize(m) for r, m in res.items()}
    json.dump(dict(args=vars(a), eval=summ, runs=a.eval_runs, real_trials={r: n_done for r in ROBOTS},
                   goals_eval=G.tolist(), seconds=time.time() - T0, per_run=res), open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    log(event='eval', trials=n_done, mean_success=float(np.mean([summ[r]['success'] for r in ROBOTS])),
        success={r: round(summ[r]['success'], 3) for r in ROBOTS}, eval_s=round(time.time() - t, 1))


checkpoint(0)
done = 0
it = 0
for cp in sorted(a.checkpoints):
    while done < cp:
        m = min(a.per_iter, cp - done)
        lane_robot = np.repeat(np.arange(NR), m)
        n = len(lane_robot)
        G = C.GOAL_LO + (C.GOAL_HI - C.GOAL_LO) * rng.uniform(size=(n, 2))
        x0 = rp.start_states(G, rng.uniform(0, 1, n), rng)
        p = {k: np.array([PR[ROBOTS[i]][k] for i in lane_robot]) for k in PR[ROBOTS[0]]}
        std = np.stack([np.exp(np.asarray(params[ROBOTS[i]]['log_std'])) for i in lane_robot])
        noise = rng.standard_normal((n, a.T, 2)) * std[:, None, :]
        res = rp.simulate(PerRobot(lane_robot), G, x0, p, a.T, noise=noise)
        alive = ~res['failed']
        r_all = reward_np(res['x'][:, :-1], G, p) * alive
        stats = {}
        for k, rob in enumerate(ROBOTS):
            sel = lane_robot == k
            ps = params[rob]
            O = jnp.asarray(res['obs'][sel], jnp.float32)
            OC = jnp.concatenate([O, jnp.broadcast_to(jnp.asarray(Z[rob], jnp.float32), O.shape[:2] + (NZ,))], -1)
            A_ = jnp.asarray(res['a'][sel], jnp.float32)
            MU = jnp.asarray(res['mu'][sel], jnp.float32)
            W = jnp.asarray(alive[sel], jnp.float32)
            adv, ret = gae(ps['critic'], OC, jnp.asarray(r_all[sel], jnp.float32), W)
            logp = jnp.sum(-0.5 * ((A_ - MU) / jnp.exp(ps['log_std'])) ** 2 - ps['log_std'], -1)
            flat = lambda v: v.reshape((-1,) + v.shape[2:])
            fO, fOC, fA, fLP, fAD, fRT, fW = map(flat, (O, OC, A_, logp, adv, ret, W))
            N = fO.shape[0]
            stop, nup = False, 0
            for ep in range(a.epochs):
                perm = jnp.array(rng.permutation(N))
                for s in range(0, N, a.mb):
                    b = perm[s:s + a.mb]
                    ps_n, st_n, l, kl = update(ps, opt_state[rob], fO[b], fOC[b], fA[b], fLP[b], fAD[b], fRT[b], fW[b])
                    if float(kl) > a.kl_stop:        # this minibatch's pre-update policy is already too far: stop
                        stop = True
                        break
                    ps, opt_state[rob], nup = ps_n, st_n, nup + 1
                if stop:
                    break
            params[rob] = ps
            stats[rob] = (round(float(r_all[sel].sum(1).mean()), 1), round(float(res['failed_end'][sel].mean()), 3),
                          round(float(kl), 4), nup)
        done += m
        it += 1
        if it % 5 == 0 or done == cp:
            log(event='ppo', it=it, trials=done, stats=stats)
    checkpoint(done)
log(event='done', seconds=round(time.time() - T0, 1))
