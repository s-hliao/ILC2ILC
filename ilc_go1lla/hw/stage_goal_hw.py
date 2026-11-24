"""DistILC stage for the Go1 goal walker on HARDWARE (2026-11-21): scripts/stage_goal_nv.py (no re-walk, rollback, nominal-model G,
anchored refit of the one network) with the practice walks flown by a backend instead of the perturbed MJX walker:
  --backend files    real robot. Iteration k: OUT/it<k>/policy.npz (hw/walker_core.Policy) + REQUEST.json {goals, policy}; then wait
                     for OUT/it<k>/walks/walk_<j>.npz (walker_node.py's logs, one per goal, in order). --remote USER@HOST:DIR mirrors
                     it<k>/ there (rsync over ssh: push the request, pull the walks). OUT/ABORT (or REMOTE/ABORT) stops the stage.
  --backend cpusim   hw/cpu_robot.py's CPU-MuJoCo robot with --case (the pipeline test: same logs, same conversion).
Same math and defaults as the sim arms (goalnv_*_1121.sh: --cap 0.03, 10 iterations x 4 walks) unless given. Per walk: the log is
converted to the simulated rollout's fields (hardware tauEst includes the motor's -Kd dq, which MuJoCo's passive joint damping does
not: tau = tauEst + KD dq, by the log's meta tau_includes_kd); a walk walker_node aborted counts as fallen at its abort step.
No per-iteration evaluation walks on hardware (evaluate the saved networks separately).

    # matcha (GPU side):
    ~/venvs/pg0/bin/python hw/stage_goal_hw.py --params runs/walker_fulldr5_foamlow/ck/params_00255590400.pkl --backend files \
        --speeds 1.0 --iters 10 --practice 4 --cap 0.03 --out runs/hw/stage_X --remote nsh1609@172.26.79.206:/home/nsh1609/go1_walk/stage_X
"""
import argparse, glob, json, os, pickle, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(HERE, '..', 'scripts')); sys.path.insert(0, HERE)
from ilc_go1lla import walkenv as WE, srb as S          # noqa: E402
import jax                                               # noqa: E402
import jax.numpy as jnp                                  # noqa: E402
import optax                                             # noqa: E402
import walker_core as C                                  # noqa: E402
from export_policy import to_npz                         # noqa: E402

FALL_MARGIN = 10
ap = argparse.ArgumentParser()
ap.add_argument('--params', required=True)
ap.add_argument('--backend', required=True, choices=('files', 'cpusim'))
ap.add_argument('--case', default='nominal', help='cpusim only: nominal | payload:m,dy,dz | foam:tc,width')
ap.add_argument('--remote', default='', help='files only: USER@HOST:DIR mirrored with rsync')
ap.add_argument('--poll', type=float, default=3.0)
ap.add_argument('--speeds', type=float, nargs='+', default=[1.0])
ap.add_argument('--iters', type=int, default=10)
ap.add_argument('--practice', type=int, default=4, help='practice walks per iteration (real walks)')
ap.add_argument('--dist', type=float, default=2.5)
ap.add_argument('--beta', type=float, default=0.5)
ap.add_argument('--delta', type=float, default=0.05)
ap.add_argument('--cap', type=float, default=0.03)
ap.add_argument('--max-lag', type=int, default=25)
ap.add_argument('--window', type=int, default=10)
ap.add_argument('--anchor-w', type=float, default=10.0)
ap.add_argument('--age-decay', type=float, default=0.5)
ap.add_argument('--fit-steps', type=int, default=800)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--rollback', type=float, default=1.3)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
log = open(os.path.join(a.out, 'progress.jsonl'), 'a')
json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
T0 = time.time()


def emit(d):
    d = dict(d, wall_s=round(time.time() - T0, 1))
    log.write(json.dumps(d) + '\n'); log.flush(); print(json.dumps(d), flush=True)


nom = WE.Walker(a.params, stand_at_goal=True)           # nominal model: G (its mj_model) and the anchors
KIN = S.LegKin(nom.env.mj_model)
NR, W = S.NR, a.window
norm_p, pol0 = nom.params[0], nom.params[1]
net = nom.net
SETTLE = 15


def handover(v):
    return int(np.ceil((float(WE.ref_end_time(np.array([a.dist, 0.0, v], np.float32))) + WE.STAND_DELAY) / 0.02))


def walk_steps(v):
    return int(np.ceil((handover(v) + SETTLE) / W) * W) + 1


def walk_part(v):
    return ((handover(v) - 1) // W) * W + 1


def goals_for(rng, k, v):
    th = np.radians(rng.uniform(-15, 15, k))
    return [np.array([a.dist * np.cos(t), a.dist * np.sin(t), v], np.float32) for t in th]


def set_policy(walker, pol):
    walker.params = (norm_p, pol) + tuple(nom.params[2:])
    walker._roll = jax.jit(walker._rollout, static_argnames=('n',))


# ---------------------------------------------------------------- walk logs -> the simulated rollout's fields
def log_to_out(z):
    meta = json.loads(str(z['meta']))
    qd, tau_now = z['qd'], z['tau_now']
    tau_act = tau_now + (C.KD * qd if meta.get('tau_includes_kd', True) else 0.0)   # the PD's spring part, as MuJoCo's actuator_force
    tau = np.concatenate([tau_act[1:], tau_act[-1:]])                                  # read after the step (WalkLog convention)
    out = dict(p=z['p'], quat=z['quat'], qj=z['qj'], v=z['v'], qd=qd, cmd=z['cmd'], a=z['a'], contact=z['contact'],
               obs_state=z['obs_state'], tau=tau, q_after=z['q_after'])
    out = {k: np.asarray(v, np.float32) for k, v in out.items()}
    abort = meta.get('abort')
    return out, (abort['step'] if abort else None), meta


def fly_logged(out, abort_step, g):
    kf = S.fall_index_feet(out, KIN)
    if abort_step is not None:
        kf = abort_step if kf is None else min(kf, abort_step)
    n_have = len(out['p'])
    e = S.measured_rows({k: val[:min(walk_part(g[2]), n_have)] for k, val in out.items()}, g, W=W)
    return kf, e


# ---------------------------------------------------------------- backends
class CpuSim:
    def __init__(self):
        from cpu_robot import CpuRobot
        self.robot = CpuRobot(case=a.case, seed=1000 + a.seed)

    def fly(self, k, pol_npz, goals):
        outs = []
        pol = C.Policy(pol_npz)
        wd = os.path.join(a.out, f'it{k}', 'walks'); os.makedirs(wd, exist_ok=True)
        for j, g in enumerate(goals):
            ctl, status, fq, ft = self.robot.walk(pol, g, meta=dict(iter=k, walk=j))
            ctl.log.save(os.path.join(wd, f'walk_{j:02d}.npz'), final_q=fq, final_tau=ft)
            outs.append(log_to_out(np.load(os.path.join(wd, f'walk_{j:02d}.npz'))))
        return outs


class Files:
    def rsync(self, src, dst):
        r = subprocess.run(['rsync', '-a', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10', src, dst], capture_output=True, text=True)
        if r.returncode:
            print(f'!!! rsync {src} -> {dst}: {r.stderr.strip()[:200]}', flush=True)
        return r.returncode == 0

    def aborted(self):
        if os.path.exists(os.path.join(a.out, 'ABORT')):
            return True
        if a.remote:
            host, d = a.remote.split(':', 1)
            r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, f'test -e {d}/ABORT'], capture_output=True)
            return r.returncode == 0
        return False

    def fly(self, k, pol_npz, goals):
        it = os.path.join(a.out, f'it{k}')
        wd = os.path.join(it, 'walks'); os.makedirs(wd, exist_ok=True)
        req = dict(iter=k, policy='policy.npz', goals=[[float(x) for x in g] for g in goals], walks='walks',
                   walk_steps=[C.walk_steps(g) for g in goals], created=time.time())
        json.dump(req, open(os.path.join(it, 'REQUEST.json'), 'w'), indent=1)
        if a.remote:
            host, d = a.remote.split(':', 1)
            subprocess.run(['ssh', '-o', 'BatchMode=yes', host, f'mkdir -p {d}/it{k}/walks'], check=False)
            self.rsync(it + '/', f'{a.remote}/it{k}/')
        print(f'>>> iteration {k}: REQUEST.json written ({len(goals)} walks{", mirrored to " + a.remote if a.remote else ""}); waiting',
              flush=True)
        want = [os.path.join(wd, f'walk_{j:02d}.npz') for j in range(len(goals))]
        while not all(os.path.exists(f) for f in want):
            if self.aborted():
                emit(dict(aborted=True, iter=k)); sys.exit(1)
            time.sleep(a.poll)
            if a.remote:
                self.rsync(f'{a.remote}/it{k}/walks/', wd + '/')
        time.sleep(1.0)
        return [log_to_out(np.load(f)) for f in want]


backend = CpuSim() if a.backend == 'cpusim' else Files()


def objective(e, kf, n):
    J = float(np.mean(e ** 2))
    if kf is not None:
        J += 100.0 * (1.0 - kf / n)
    return J


def walk_metrics(out, kf, g):
    """time to goal (within 0.15 m), RMS distance from the start->goal line, final distance; fall."""
    end = len(out['p']) if kf is None else kf
    p = out['p'][:end, :2]
    d = np.linalg.norm(p - g[:2], axis=1)
    arr = np.where(d < 0.15)[0]
    u = g[:2] / np.linalg.norm(g[:2])
    cross = np.abs(u[0] * p[:, 1] - u[1] * p[:, 0])
    k_end = arr[0] + 1 if len(arr) else max(end, 1)
    return dict(fell=kf is not None, t_goal=None if (kf is not None or not len(arr)) else float(arr[0] * 0.02),
                path_cm=float(100 * np.sqrt(np.mean(cross[:k_end] ** 2))) if len(p) else None,
                final_cm=None if kf is not None else float(100 * d[-1]))


# ---------------------------------------------------------------- anchors (sim, nominal model) and the refit: as stage_goal_nv
rng_a = np.random.default_rng(1)
anc_obs, anc_act = [], []
for v in a.speeds:
    for g in goals_for(rng_a, 3, v):
        out = nom.rollout(g, n=walk_steps(v))
        anc_obs.append(out['obs_state'][:walk_part(v)]); anc_act.append(out['a'][:walk_part(v)])
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
v_idx = 0
pol = pol0
best = dict(J=np.inf, pol=pol0)
cap = a.cap
buf = []
emit(dict(it=-1, kind='start', backend=a.backend, case=a.case if a.backend == 'cpusim' else 'real', params=a.params))
for it in range(a.iters):
    v = a.speeds[v_idx]
    n = walk_steps(v)
    set_policy(nom, pol)
    pol_npz = os.path.join(a.out, f'it{it}', 'policy.npz')
    to_npz(norm_p, pol, pol_npz)
    goals = goals_for(rng_p, a.practice, v)
    flown = backend.fly(it, pol_npz, goals)
    targets, Js, falls, walks = [], [], 0, []
    for g, (out, abort_step, meta) in zip(goals, flown):
        kf, e = fly_logged(out, abort_step, g)
        Js.append(objective(e, kf, n)); falls += kf is not None
        walks.append(dict(walk_metrics(out, kf, g), kf=kf, abort=meta.get('abort'), steps=len(out['p'])))
        n_use = min(walk_part(v), len(out['p'])) if kf is None else min(walk_part(v), ((kf - FALL_MARGIN - 1) // W) * W + 1)
        n_use = ((n_use - 1) // W) * W + 1
        if n_use < 2 * W + 1:
            continue
        cut = {k: val[:n_use] for k, val in out.items()}
        rows, G = S.make_G(nom, cut, g, W=W)
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
    advanced = False
    if falls == 0 and not rolled and v_idx < len(a.speeds) - 1:
        v_idx += 1; advanced = True
        best = dict(J=np.inf, pol=pol)
    tg = [w['t_goal'] for w in walks if w['t_goal'] is not None]
    emit(dict(it=it, speed=v, practice_falls=falls, practice_J=J, rolled_back=rolled, cap=cap, advanced=advanced, next_speed=a.speeds[v_idx],
              n_targets=sum(len(b['obs']) for b in buf), walks=walks,
              practice=dict(t_goal=float(np.mean(tg)) if tg else None, arrived=len(tg),
                            path_cm=float(np.mean([w['path_cm'] for w in walks if w['path_cm'] is not None])))))
    pickle.dump(dict(norm=norm_p, pol=pol), open(os.path.join(a.out, f'policy_it{it}.pkl'), 'wb'))
final_npz = os.path.join(a.out, 'policy_final.npz')
to_npz(norm_p, pol, final_npz)
emit(dict(done=True, real_walks=a.iters * a.practice, final=final_npz))
