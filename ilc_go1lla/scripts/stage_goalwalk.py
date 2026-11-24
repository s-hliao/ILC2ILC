"""DistILC stage for the GOAL-CONDITIONED walker (2026-11-24; goalwalk.py / goalstage.py), from stage_goal_nv.py:
- the sim robot walks the goal walker (no command law); goals in ANY direction at --dist m, random bearings for practice, 8 fixed
  bearings (0, +-45, +-90, +-135, 180) for evaluation; one walk = GS.walk_steps(goal); no speed ladder, no hand-over
- rows (--rows): time | progress | goal (goalstage.py); time/progress compare with the BASE network's walk to the same goal on the
  NOMINAL sim robot (the reference, fixed); goal = every window's position error to the goal
- evaluation reports every row type's rms (so arms are compared on the same yardsticks) + time to goal, path, final error, falls
Everything else as stage_goal_nv.py: SRB G with measured saturation and load stance, --max-lag truncation, damped GN step with an
rms cap, targets pi(o) + du, anchored refit of the ONE network (anchors: the base network on nominal walks), rollback.
    python scripts/stage_goalwalk.py --params runs/goalwalk/L_vel2_s0/params.pkl --case payload3R --rows time --out runs/gstage/...
"""
import argparse, json, os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ilc_go1lla import walkenv as WE, srb as S, goalstage as GS, goalwalk as GW   # noqa: E402
import jax                                               # noqa: E402
import jax.numpy as jnp                                  # noqa: E402
import optax                                             # noqa: E402
from zeroshot_cases import CASES, parse_case             # noqa: E402

FALL_MARGIN = 10
ap = argparse.ArgumentParser()
ap.add_argument('--params', required=True)
ap.add_argument('--case', required=True)
ap.add_argument('--variant', default='vel2')
ap.add_argument('--rows', choices=['time', 'progress', 'goal'], required=True)
ap.add_argument('--iters', type=int, default=10)
ap.add_argument('--practice', type=int, default=4, help='practice walks per iteration (real trials)')
ap.add_argument('--eval-walks', type=int, default=8)
ap.add_argument('--dist', type=float, default=1.5)
ap.add_argument('--beta', type=float, default=0.5)
ap.add_argument('--delta', type=float, default=0.05)
ap.add_argument('--cap', type=float, default=0.03)
ap.add_argument('--max-lag', type=int, default=25)
ap.add_argument('--window', type=int, default=10)
ap.add_argument('--anchor-w', type=float, default=10.0)
ap.add_argument('--age-decay', type=float, default=0.5)
ap.add_argument('--fit-steps', type=int, default=800)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--rollback', type=float, default=1.3, help='revert if the practice objective > this x the best so far')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
log = open(os.path.join(a.out, 'progress.jsonl'), 'a')
T0 = time.time()


def emit(d):
    d = dict(d, wall_s=round(time.time() - T0, 1))
    log.write(json.dumps(d) + '\n'); log.flush(); print(json.dumps(d), flush=True)


CASE_P = parse_case(a.case)
real = GS.GoalSimWalker(a.params, a.variant, **CASE_P)
nom = GS.GoalSimWalker(a.params, a.variant)
KIN = S.LegKin(real.env.mj_model)
W = a.window
NR = GS.NRS[a.rows]
norm_p, pol0 = real.params[0], real.params[1]
net = real.net
BEAR = [0, 45, -45, 90, -90, 135, -135, 180]


def goal_at(b):
    return np.array([a.dist * np.cos(np.radians(b)), a.dist * np.sin(np.radians(b))], np.float32)


def goals_for(rng, k):
    return [goal_at(b) for b in rng.uniform(-180, 180, k)]


EVAL_GOALS = [goal_at(b) for b in (BEAR * 4)[:a.eval_walks]]
_REF = {}


def ref_for(g):
    """The base network's walk to g on the nominal sim robot (cached; the base never changes, so neither does it)."""
    key = tuple(np.round(g, 5))
    if key not in _REF:
        _REF[key] = GS.reference(nom.rollout(g, GS.walk_steps(g, W)))
    return _REF[key]


def rows_of(out, g, kind):
    return S.measured_rows(out, g, W=W, row_fn=GS.row_fn_for(kind, g, ref_for(g)), nr=GS.NRS[kind])


def set_policy(walker, pol):
    walker.params = (norm_p, pol) + tuple(real.params[2:])
    walker._roll = jax.jit(walker._rollout, static_argnames=('n',))


def fly(walker, g):
    n = GS.walk_steps(g, W)
    out = walker.rollout(g, n)
    kf = S.fall_index_feet(out, KIN)
    return out, kf, rows_of(out, g, a.rows), n


def objective(e, kf, n):
    J = float(np.mean(e ** 2))
    if kf is not None:
        J += 100.0 * (1.0 - kf / n)
    return J


def score(walker, goals):
    rows = []
    for g in goals:
        out, kf, e, n = fly(walker, g)
        end = n if kf is None else kf
        p = out['p'][:end, :2]
        d = np.linalg.norm(p - g[:2], axis=1)
        arr = np.where(d < GW.ARRIVE)[0]
        u = g[:2] / np.linalg.norm(g[:2])
        cross = np.abs(u[0] * p[:, 1] - u[1] * p[:, 0])
        k_end = arr[0] + 1 if len(arr) else end
        cut = {k: v[:end] for k, v in out.items()}
        rms = {f'rms_{k}': (float(np.sqrt(np.mean(rows_of(cut, g, k) ** 2))) if end > 2 * W + 1 else None) for k in GS.NRS}
        rows.append(dict(fell=kf is not None, t_goal=None if (kf is not None or not len(arr)) else float(arr[0] * 0.02),
                         path_cm=float(100 * np.sqrt(np.mean(cross[:k_end] ** 2))),
                         final_cm=None if kf is not None else float(100 * d[-1]), **rms))
    up = [r for r in rows if not r['fell']]
    tg = [r['t_goal'] for r in rows if r['t_goal'] is not None]
    agg = lambda k: float(np.mean([r[k] for r in up if r[k] is not None])) if up else None
    return dict(falls=sum(r['fell'] for r in rows), n=len(rows), arrived=len(tg), t_goal=float(np.mean(tg)) if tg else None,
                path_cm=float(np.mean([r['path_cm'] for r in rows])), final_cm=float(np.median([r['final_cm'] for r in up])) if up else None,
                **{f'rms_{k}': agg(f'rms_{k}') for k in GS.NRS})


# anchors: the base network's actions on nominal-robot walks (sim, nominal model), 6 random bearings
rng_a = np.random.default_rng(1)
anc_obs, anc_act = [], []
for g in goals_for(rng_a, 6):
    out = nom.rollout(g, GS.walk_steps(g, W))
    anc_obs.append(out['obs_state']); anc_act.append(out['a'])
anc_obs, anc_act = np.concatenate(anc_obs), np.concatenate(anc_act)


def pi(pol, obs):
    logits = net.policy_network.apply(norm_p, pol, {'state': obs})
    return jnp.tanh(jnp.split(logits, 2, axis=-1)[0])


@jax.jit
def fit_loss(pol, obs, tgt, wts, aobs, aact):
    lt = jnp.sum(wts[:, None] * (pi(pol, obs) - tgt) ** 2) / jnp.sum(wts)
    la = jnp.mean((pi(pol, aobs) - aact) ** 2)
    return lt + a.anchor_w * la


def refit(pol, buf):
    obs = jnp.asarray(np.concatenate([b['obs'] for b in buf]))
    tgt = jnp.asarray(np.concatenate([b['tgt'] for b in buf]))
    wts = jnp.asarray(np.concatenate([np.full(len(b['obs']), a.age_decay ** b['age']) for b in buf]), jnp.float32)
    opt = optax.adam(a.lr)
    st = opt.init(pol)
    gfn = jax.jit(jax.value_and_grad(fit_loss))
    key = np.random.default_rng(a.seed)
    for i in range(a.fit_steps):
        ia = key.integers(0, len(anc_obs), 4096)
        l, gr = gfn(pol, obs, tgt, wts, jnp.asarray(anc_obs[ia]), jnp.asarray(anc_act[ia]))
        up, st = opt.update(gr, st, pol)
        pol = optax.apply_updates(pol, up)
    return pol, float(l)


rng_p = np.random.default_rng(100 + a.seed)
pol = pol0
best = dict(J=np.inf, pol=pol0)
cap = a.cap
buf = []
emit(dict(it=-1, kind='eval_zero_shot', case=a.case, rows=a.rows, eval=score(real, EVAL_GOALS)))
emit(dict(it=-1, kind='eval_nominal_robot', eval=score(nom, EVAL_GOALS)))
for it in range(a.iters):
    set_policy(real, pol)
    goals = goals_for(rng_p, a.practice)
    targets, Js, falls = [], [], 0
    for g in goals:
        out, kf, e, n = fly(real, g)
        Js.append(objective(e, kf, n)); falls += kf is not None
        n_use = n if kf is None else min(n, ((kf - FALL_MARGIN - 1) // W) * W + 1)
        if n_use < 2 * W + 1:
            continue
        cut = {k: val[:n_use] for k, val in out.items()}
        rows, G = S.make_G(real, cut, g, W=W, cmd_fn=GS.cmd_fn_for(g), row_fn=GS.row_fn_for(a.rows, g, ref_for(g)), nr=NR)
        G = np.array(G)
        if a.max_lag > 0:
            nr = G.shape[0] // NR
            Gr = G.reshape(nr, NR, n_use, 12)
            for j in range(nr):
                Gr[j, :, :max(0, (j + 1) * W - a.max_lag)] = 0.0
            G = Gr.reshape(G.shape)
        N = G.T @ G
        lam = a.delta * np.trace(N) / N.shape[0]
        du = -a.beta * np.linalg.solve(N + lam * np.eye(N.shape[0]), G.T @ rows)
        r = np.sqrt(np.mean(du ** 2))
        if r > cap:
            du *= cap / r
        du = du.reshape(n_use, 12)
        targets.append(dict(obs=cut['obs_state'], tgt=np.clip(cut['a'] + du, -1, 1)))
    J = float(np.mean(Js))
    rolled = False
    if J > a.rollback * best['J'] and np.isfinite(best['J']):
        pol, cap, rolled = best['pol'], cap * 0.7, True
        buf = [b for b in buf if b['age'] > 0]
    else:
        if J < best['J']:
            best = dict(J=J, pol=pol)
        for b in buf:
            b['age'] += 1
        buf += [dict(b, age=0) for b in targets]
        pol, loss = refit(pol, buf) if buf else (pol, None)
    set_policy(real, pol)
    emit(dict(it=it, practice_falls=falls, practice_J=J, rolled_back=rolled, cap=cap,
              n_targets=sum(len(b['obs']) for b in buf), eval=score(real, EVAL_GOALS)))
    pickle.dump(dict(norm=norm_p, pol=pol), open(os.path.join(a.out, f'policy_it{it}.pkl'), 'wb'))
emit(dict(done=True, real_walks=a.iters * a.practice))
