"""DistILC stage for the Go1 walker on a perturbed ("real") robot, with a speed curriculum.

Each iteration: fly K practice walks at the current speed; per walk the SRB-linearized G (srb.py: measured-saturation motors,
rows cross/along/heading/roll/pitch per 0.2 s, G limited to --max-lag steps), a walk that falls used up to FALL_MARGIN
steps before the fall; the damped GN step du = -beta (G'G + lam I)^-1 G'e (lam = delta tr/n, rms cap); targets
pi(o) + du on the measured observations; anchored refit of the ONE network (targets age-decayed, anchors = the original
policy on nominal-robot walks). Rollback: if the practice objective got worse than rollback x the best, revert the network
and shrink the cap x 0.7. Curriculum: the speed steps up the --speeds ladder once a practice round has no falls.
Held-out evaluation walks (other goals) after every iteration at the current and the target speed.

    ~/venvs/pg0/bin/python scripts/stage.py --params runs/walker_nom_s0/params.pkl --case lockRL --out runs/stage_lockRL_s0
"""
import argparse, json, os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ilc_go1lla import walkenv as WE, srb as S          # noqa: E402  (walkenv installs the jax shim)
import jax                                               # noqa: E402
import jax.numpy as jnp                                  # noqa: E402
import optax                                             # noqa: E402
from zeroshot_cases import CASES                         # noqa: E402

FALL_MARGIN = 10
ap = argparse.ArgumentParser()
ap.add_argument('--params', required=True)
ap.add_argument('--case', required=True)
ap.add_argument('--speeds', type=float, nargs='+', default=[0.3, 0.4, 0.5, 0.6, 0.8])
ap.add_argument('--iters', type=int, default=12)
ap.add_argument('--practice', type=int, default=4, help='practice walks per iteration (real trials)')
ap.add_argument('--eval-walks', type=int, default=6)
ap.add_argument('--dist', type=float, default=2.5)
ap.add_argument('--beta', type=float, default=0.5)
ap.add_argument('--delta', type=float, default=0.05)
ap.add_argument('--cap', type=float, default=0.06)
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


real = WE.Walker(a.params, **CASES[a.case])
nom = WE.Walker(a.params)
NR, W = S.NR, a.window
norm_p, pol0 = real.params[0], real.params[1]
net = real.net


def walk_steps(v):
    n = int((WE.T0 + a.dist / v + v + 1.5) / 0.02)
    return int(np.ceil(n / W) * W) + 1


def goals_for(rng, k, v):
    th = np.radians(rng.uniform(-15, 15, k))
    return [np.array([a.dist * np.cos(t), a.dist * np.sin(t), v], np.float32) for t in th]


def set_policy(walker, pol):
    walker.params = (norm_p, pol) + tuple(real.params[2:])
    walker._roll = jax.jit(walker._rollout, static_argnames=('n',))


def fly(walker, g, n):
    out = walker.rollout(g, n=n)
    kf = S.fall_index(out)
    e = S.measured_rows(out, g, W=W)
    return out, kf, e


def objective(e, kf, n):
    """Practice objective: mean squared rows over the upright part, plus a fall penalty growing with how early it fell."""
    J = float(np.mean(e ** 2))
    if kf is not None:
        J += 100.0 * (1.0 - kf / n)
    return J


def score(walker, goals, n):
    rows = []
    for g in goals:
        out, kf, e = fly(walker, g, n)
        m = len(e) if kf is None else max(0, ((kf - 1) // W)) * NR
        ee = e[:m] if m else e[:NR]
        track = float(5 * np.sqrt(np.mean(ee[0::NR] ** 2 + ee[1::NR] ** 2)))
        att = float(np.degrees(0.1) * np.sqrt(np.mean(ee[3::NR] ** 2 + ee[4::NR] ** 2)))
        dist = float(np.linalg.norm(out['p'][-1, :2] - g[:2]))
        rows.append(dict(fell=kf is not None, t_fall=None if kf is None else kf * 0.02, track_cm=track, att_deg=att,
                         final_cm=100 * dist))
    up = [r for r in rows if not r['fell']]
    return dict(falls=sum(r['fell'] for r in rows), n=len(rows),
                track_cm=float(np.mean([r['track_cm'] for r in up])) if up else None,
                att_deg=float(np.mean([r['att_deg'] for r in up])) if up else None,
                final_cm=float(np.median([r['final_cm'] for r in rows])),
                t_fall_mean=float(np.mean([r['t_fall'] for r in rows if r['fell']])) if len(up) < len(rows) else None)


# anchors: the original policy's actions on nominal-robot walks (sim, nominal model)
rng_a = np.random.default_rng(1)
anc_obs, anc_act = [], []
for v in a.speeds:
    for g in goals_for(rng_a, 3, v):
        out = nom.rollout(g, n=walk_steps(v))
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
rng_e = np.random.default_rng(200)
v_idx = 0
v_tgt = a.speeds[-1]
eval_goals_tgt = goals_for(np.random.default_rng(300), a.eval_walks, v_tgt)
pol = pol0
best = dict(J=np.inf, pol=pol0)
cap = a.cap
buf = []
emit(dict(it=-1, kind='eval_zero_shot', case=a.case, **{f'v{v}': score(real, goals_for(np.random.default_rng(300), a.eval_walks, v), walk_steps(v)) for v in a.speeds}))
for it in range(a.iters):
    v = a.speeds[v_idx]
    n = walk_steps(v)
    set_policy(real, pol)
    goals = goals_for(rng_p, a.practice, v)
    targets, Js, falls, coss = [], [], 0, []
    for g in goals:
        out, kf, e = fly(real, g, n)
        Js.append(objective(e, kf, n)); falls += kf is not None
        n_use = n if kf is None else ((kf - FALL_MARGIN - 1) // W) * W + 1
        if n_use < 2 * W + 1:
            continue
        cut = {k: val[:n_use] for k, val in out.items()}
        rows, G = S.make_G(real, cut, g, W=W)
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
        pol, cap, rolled = best['pol'], cap * 0.7, True          # the last refit made practice worse: revert, smaller steps
        buf = [b for b in buf if b['age'] > 0]
    else:
        if J < best['J']:
            best = dict(J=J, pol=pol)
        for b in buf:
            b['age'] += 1
        buf += [dict(b, age=0) for b in targets]
        pol, loss = refit(pol, buf) if buf else (pol, None)
    advanced = False
    if falls == 0 and not rolled and v_idx < len(a.speeds) - 1:
        v_idx += 1; advanced = True
        best = dict(J=np.inf, pol=pol)      # objectives at a new speed are not comparable with the old speed's best
    set_policy(real, pol)
    ev = dict(cur=score(real, goals_for(np.random.default_rng(300), a.eval_walks, v), n))
    if v != v_tgt:
        ev['target'] = score(real, eval_goals_tgt, walk_steps(v_tgt))
    emit(dict(it=it, speed=v, practice_falls=falls, practice_J=J, rolled_back=rolled, cap=cap, advanced=advanced,
              next_speed=a.speeds[v_idx], n_targets=sum(len(b['obs']) for b in buf), eval=ev))
    pickle.dump(dict(norm=norm_p, pol=pol), open(os.path.join(a.out, f'policy_it{it}.pkl'), 'wb'))
emit(dict(done=True, real_walks=a.iters * a.practice))
