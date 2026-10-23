#!/usr/bin/env python3
"""flows_car.py: the closed-loop vector field of the F1TENTH drift car, its corruption on the "real" (multi-body) cars and
its re-correction over the hardware-stage iterations (log/dilc/flows/SPEC.md).

The map. F^s_pi(x, plan): the car started at the planar state x (single-track layout [X, Y, psi, vx, vy, r, omega_w, I,
delta]) flies H control periods (H = 4: 0.1 s) under policy pi against the plan, on system s; the field is taken in the
plan-relative error coordinates the stage and the policy use (sim_jax.features: e = [e_y, e_psi, vx, vy, r] minus the plan
at the car's own arc length, plus the speed error |V| - |V_plan|):  V^s_pi(x) = e(F^s_pi(x)) - e(x).
  s = sim:  the nominal Fiala single-track model (sim_jax.step, RK4 0.25 ms), no delays.
  s = car:  real_car's multi-body car (mb_car.step) started at mb_from_planar(x) (suspension at its static equilibrium),
            its command delays / steering offset / servo applied, the command history warm-started with pi's action at x;
            no sensing noise (the deterministic map).
  RMA:      z_hat is held at the value the policy's OWN trajectory on that system had at the nearest plan index (the
            adapted context; H 0.1 s << the 1 s history), recomputed from the measured history as hw_stage.rma_zhat.
Field error |dV| = RMS over the 5 motion coordinates of (V^car_{pi_i} - V^sim_{pi_0}) / DEV_SCALE[:5] (0.1 m, 0.2 rad,
0.5, 0.5 m/s, 1 rad/s), averaged over the tube states. Also against the plan's own flow (its periodic LQR in the nominal
sim), and in the task coordinates only (e_y / 0.05 m, speed / 0.3 m/s: the stage's TASK_SCALE).

The tube (per car and plan): the states visited by sim pi_0 and by the car under every pi_i (i = 0..N; 3 runs x 2 chained
laps each from the VALIDATION starts, seed 301 as reeval.py), every 40th period, each with one perturbed copy of relative
size 0.25 (+-0.025 m lateral, +-0.025 rad, +-0.075 m/s vx / vy, +-0.125 rad/s r: sim_jax.start_states' pattern x 0.25);
"off-tube" (IN plans only): the same base states perturbed at relative size 1.0. The tube is the same for every i.
Goal sets (each plan on the ladder rung the car settled on for its track, as the stage's adaptive evaluation):
  IN   = the stage's training goals, mocap_{square2fast,figfast}_b25;
  HULL = the 21 deg drift plans of the same tracks (validation plans next to the trained 25 deg);
  OUT  = the grip plans b0 of the same tracks + the held-out track's 25 deg drift plan (mocap_squareH_b25, never trained).
Task error: RMS lateral error (m) over the 3 x 2-lap validation runs of pi_i on the car, a crashed run's remaining periods
counted at the room's envelope (0.3 m); plus the stage's success test (no crash, RMS e_y <= 0.10 m, pace >= 0.9).

  flows_car.py compute --stage DIR --name NAME [--procs 5]       -> DIR_flows/NAME.npz + NAME.json (per car / plan / it)
  flows_car.py portrait --stage DIR --name NAME --car C --plan P  -> figs/portrait_NAME_C.png (+ .npz)
  flows_car.py report                                            -> summary.json, figs/field_err.png, figs/task_err.png
"""
import argparse
import json
import math
import os
import sys

REPO = '/home/henry/ilc_ws/src/ilc_f1tenth/ilc2real'
OUT = os.path.dirname(os.path.abspath(__file__))
# the tight-track setup (env_tight.sh), set here so every worker sees it
for k, v in dict(F1T_MU='0.2', F1T_PLANS=f'{REPO}/plans_mu02_tight', F1T_TRACKS=f'{REPO}/../lifted_linear_tire_20261009/tracks_tight',
                 F1T_SAFETY_EY='0.3', JAX_PLATFORMS='cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false').items():
    if os.environ.get(k, v) != v:
        raise SystemExit(f'{k}={os.environ[k]} differs from the tight-track setup ({v})')
    os.environ[k] = v
sys.path.insert(0, REPO)

import numpy as np                      # noqa: E402

import bank                             # noqa: E402
import car_model as cm                  # noqa: E402
import mb_car as mc                     # noqa: E402
import real_car as rc                   # noqa: E402
import sim_jax_np as sn                 # noqa: E402
from consts import DEV_SCALE, TASK_SCALE   # noqa: E402

H = int(os.environ.get('FLOWS_H', '20'))   # control periods per map (20: 0.5 s; FLOWS_H overrides, --H sets it)
SEED = 301                # the VALIDATION reeval seed (test seeds 1701 / 2701 / 3701 are spent: never used here)
RUNS, LAPS = 3, 2.0
STRIDE = int(os.environ.get('FLOWS_STRIDE', '40'))   # tube subsampling (periods); --stride sets it
PERT, PERT_OFF = 0.25, 1.0
S5 = np.asarray(DEV_SCALE[:5])
TRACKS = ['mocap_square2fast', 'mocap_figfast']
SETS = {'IN': [f'{t}_b25' for t in TRACKS], 'HULL': [f'{t}_b21' for t in TRACKS],
        'OUT': [f'{t}_b0' for t in TRACKS] + ['mocap_squareH_b25']}
BASES = [n for s in SETS.values() for n in s]
ENV = float(os.environ['F1T_SAFETY_EY'])

_PL, _PNP, _POL, _CAR = {}, {}, {}, {}


def plan(n):
    if n not in _PL:
        _PL[n] = bank.load_plan(n)
        _PNP[n] = sn.PlanNP(_PL[n])
    return _PL[n]


def pnp(n):
    plan(n)
    return _PNP[n]


def car(c):
    if c not in _CAR:
        _CAR[c] = mc.car_config(c)
    return _CAR[c]


def have(n):
    return os.path.exists(os.path.join(bank.PLAN_DIR, n + '.npz'))


def on_rung(base, rungs_car):
    n = base + (rungs_car or {}).get(base.split('_b')[0], '')
    return n if have(n) else base


def load_pol(path):
    if path not in _POL:
        d = np.load(path)
        n = len([k for k in d.files if k.startswith('W')])
        pol = [(np.asarray(d[f'W{i}']), np.asarray(d[f'b{i}'])) for i in range(n)]
        ad = os.path.join(os.path.dirname(path), 'rma_adapt.npz')
        if 'mode' in d.files and str(d['mode']) == 'rma':
            e = np.load(ad)
            m = len([k for k in e.files if k.startswith('W')])
            pol = dict(actor=pol, adapt=[(np.asarray(e[f'W{i}']), np.asarray(e[f'b{i}'])) for i in range(m)], hist=int(e['hist']))
        _POL[path] = pol
    return _POL[path]


def act(pol, obs, z=None):
    if isinstance(pol, dict):
        return rc.mlp_np(pol['actor'], np.concatenate([obs, z], -1))
    return rc.mlp_np(pol, obs)


def rma_zhat(rma, obs, a):
    """hw_stage.rma_zhat: z_hat_t = phi(last H of [obs_k[:8], a_{k-1}]) along a flown trajectory."""
    Hh = rma['hist']
    f = np.concatenate([obs[:, :8], np.concatenate([np.zeros((1, 2)), a[:-1]])], 1)
    f = np.concatenate([np.zeros((Hh - 1, 10)), f])
    hv = np.lib.stride_tricks.sliding_window_view(f, (Hh, 10))[:, 0].reshape(len(obs), Hh * 10)
    return rc.mlp_np(rma['adapt'], hv)


def starts(base, name):
    """The validation starts (reeval.py's pattern on seed 301): s0 spread over the lap, a +-0.5 relative perturbation."""
    rs = np.random.default_rng(SEED + BASES.index(base))
    pl = plan(name)
    return [dict(s0=(k / RUNS) * float(pl['length']), pert=rs.uniform(-0.5, 0.5, 5), seed=SEED * 3 + k) for k in range(RUNS)]


def x_start(pl, s0, pert):
    """real_car.run's start state."""
    i0 = int(np.round(np.mod(s0, float(pl['length'])) / float(pl['ds']))) % len(pl['s'])
    x = pl['x'][i0].copy()
    r = np.asarray(pert)
    x[0] += -math.sin(x[2]) * 0.1 * r[0]
    x[1] += math.cos(x[2]) * 0.1 * r[0]
    x[2] += 0.1 * r[1]
    x[3] += 0.3 * r[2]
    x[4] += 0.3 * r[3]
    x[5] += 0.5 * r[4]
    x[6] = max(x[6] + 0.3 * r[2] / cm.NOMINAL['rw'], 0.0)
    return x, i0


def perturb(X, scale, rng):
    """Relative perturbation (sim_jax.start_states' pattern x scale)."""
    r = rng.uniform(-1, 1, (len(X), 5)) * scale
    X = X.copy()
    X[:, 0] += -np.sin(X[:, 2]) * 0.1 * r[:, 0]
    X[:, 1] += np.cos(X[:, 2]) * 0.1 * r[:, 0]
    X[:, 2] += 0.1 * r[:, 1]
    X[:, 3] += 0.3 * r[:, 2]
    X[:, 4] += 0.3 * r[:, 3]
    X[:, 5] += 0.5 * r[:, 4]
    X[:, 6] = np.maximum(X[:, 6] + 0.3 * r[:, 2] / cm.NOMINAL['rw'], 0.0)
    return X


def design_mu(name):
    """The friction factor a plan was designed for (the ladder's rungs: _mu80 / _mu60): the reference sim of that plan."""
    return 0.8 if name.endswith('_mu80') else 0.6 if name.endswith('_mu60') else 1.0


def mb_add(xm, dx):
    """A planar perturbation dx (9) applied to a multi-body state (mb_car.planar's layout)."""
    xm = np.asarray(xm, float).copy()
    xm[0] += dx[0]; xm[1] += dx[1]; xm[4] += dx[2]; xm[3] += dx[3]; xm[10] += dx[4]; xm[5] += dx[5]
    xm[23:27] = np.maximum(xm[23:27] + dx[6], 0.0); xm[29] += dx[6]
    return xm


def errs(name, X, I):
    p = pnp(name)
    return np.array([p.features(int(i), x)[1] for x, i in zip(X, I)])


def nearest_z(Itraj, Ztraj, Iq, M):
    if Ztraj is None or len(Itraj) == 0:
        return None
    d = np.abs(((np.asarray(Iq)[:, None] - np.asarray(Itraj)[None, :]) + M // 2) % M - M // 2)
    return Ztraj[np.argmin(d, 1)]


# ------------------------------------------------------------------------------------------------- worker tasks (numpy)
def w_fly(job):
    """One validation run of policy (path) on a car: 2 chained laps, sensing noise on, as the evaluation."""
    pl = plan(job['plan'])
    pol = load_pol(job['pol'])
    rec, step0 = [], mc.step

    def step_rec(xm, *args, **kw):                   # the true multi-body state before each period (aligned with x_true)
        rec.append(np.asarray(xm, float).copy())
        return step0(xm, *args, **kw)
    mc.step = step_rec
    try:
        r = rc.run(job['car'], pl, pol, laps=LAPS, s0=job['s0'], pert=job['pert'], seed=job['seed'])
    finally:
        mc.step = step0
    XM = np.full((len(r['x_true']), mc.NXM + 2), np.nan)
    if rec:
        XM[:len(rec)] = np.array(rec)[:len(XM)]
    m = rc.lap_metrics(r, pl)
    T = int(math.ceil(LAPS * float(pl['lap_time']) / rc.DT))
    e = np.abs(r['err_true'][:, 0]) if len(r['err_true']) else np.zeros(0)
    e_pad = np.concatenate([e, np.full(max(T - len(e), 0), ENV)]) if r['failed'] else e
    z = rma_zhat(pol, r['obs'], r['a']) if isinstance(pol, dict) and len(r['obs']) else None
    return dict(x=r['x_true'], xm=XM, idx=r['idx'], z=z, failed=bool(r['failed']), success=bool(m.get('success', False)),
                rms_ey_pad=float(np.sqrt(np.mean(e_pad ** 2))) if len(e_pad) else ENV, rms_ey=m.get('rms_ey'),
                pace=m.get('pace'), laps=float(r['laps']))


def real_map_one(c, p, pol, x0, i0, z, xm0=None):
    xm = mc.mb_from_planar(np.asarray(x0, float), c['p']) if xm0 is None or not np.all(np.isfinite(xm0)) else np.asarray(xm0, float).copy()
    idx = p.project(int(i0), x0[:2])
    obs, _ = p.features(idx, x0)
    hist = [act(pol, obs, z) * cm.A_SCALE + cm.A_OFF] * 4      # warm start: the command history at pi's action at x
    for _ in range(H):
        xt = mc.planar(xm)
        idx = p.project(idx, xt[:2])
        obs, _ = p.features(idx, xt)
        hist.append(act(pol, obs, z) * cm.A_SCALE + cm.A_OFF)
        hist.pop(0)
        d_cmd = hist[-1 - int(c['d_delay'])][0] + c['d_off']
        i_cmd = hist[-1 - int(c['i_delay'])][1]
        try:
            xm = mc.step(xm, d_cmd, i_cmd, c, rc.DT)
        except (ZeroDivisionError, FloatingPointError):
            return np.full(9, np.nan), idx
    xt = mc.planar(xm)
    return xt, p.project(idx, xt[:2])


def w_map(job):
    c, p, pol = car(job['car']), pnp(job['plan']), load_pol(job['pol'])
    Z, XM = job.get('Z'), job.get('XM')
    out = [real_map_one(c, p, pol, x, i, None if Z is None else Z[k], None if XM is None else XM[k])
           for k, (x, i) in enumerate(zip(job['X'], job['I']))]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


_POOL = None


def pmap(fn, jobs, procs):
    global _POOL
    if procs <= 1:
        return [fn(j) for j in jobs]
    if _POOL is None:
        import multiprocessing as mp
        _POOL = mp.get_context('spawn').Pool(procs)
    return _POOL.map(fn, jobs, chunksize=1)


# ------------------------------------------------------------------------------------------------- the nominal sim (JAX)
class Sim:
    def __init__(self, names):
        import jax
        import jax.numpy as jnp
        import sim_jax as sj
        self.jax, self.jnp, self.sj = jax, jnp, sj
        self.names = list(names)
        self.P = sj.Plans([plan(n) for n in self.names], self.names)
        p0 = sj.nominal_params(1)
        self.p1 = {k: v[0] for k, v in p0.items()}
        self._f = {}

    def _mapper(self, kind, nz):
        if (kind, nz) in self._f:
            return self._f[(kind, nz)]
        jax, jnp, sj, P, p1 = self.jax, self.jnp, self.sj, self.P, self.p1

        def one(ps, x, i, pid, z, pp):
            def body(c, _):
                x, i = c
                i = sj.project(P, pid, i, x[:2])
                obs, _ = sj.features(P, pid, i, x)
                if kind == 'lqr':
                    a = sj.expert_action(P, pid, i, x)
                else:
                    mu = sj.mlp(ps, jnp.concatenate([obs, z]) if nz else obs)
                    a = mu * sj.A_SCALE + sj.A_OFF
                return (sj.step(x, a, pp), i), None
            (x, i), _ = jax.lax.scan(body, (x, i), None, length=H)
            return x, sj.project(P, pid, i, x[:2])
        f = jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0, 0, 0)))
        self._f[(kind, nz)] = f
        return f

    def params(self, mu):
        """Nominal model parameters per lane, the friction at each plan's design value (1 for the plans themselves)."""
        p = self.sj.nominal_params(len(mu))
        p['muf'], p['mur'] = p['muf'] * self.jnp.asarray(mu), p['mur'] * self.jnp.asarray(mu)
        return p

    def map(self, pol, X, I, names, Z=None, kind='net'):
        """H-period map of many states (each on its plan) under pol (MLP / RMA with per-state Z) or the plans' LQR."""
        jnp = self.jnp
        ps = None
        nz = 0
        if kind == 'net':
            mlp = pol['actor'] if isinstance(pol, dict) else pol
            ps = [(jnp.array(W), jnp.array(b)) for W, b in mlp]
            nz = Z.shape[1] if Z is not None else 0
        f = self._mapper(kind, nz)
        n = len(X)
        CH = 2048
        XH, IH = np.zeros((n, 9)), np.zeros(n, np.int64)
        pid = np.array([self.names.index(m) for m in names])
        Zf = Z if nz else np.zeros((n, 1))
        mu = np.array([design_mu(m) for m in names])
        for a in range(0, n, CH):
            b = min(a + CH, n)
            pad = CH - (b - a)
            sl = lambda A: np.concatenate([A[a:b], np.repeat(A[b - 1:b], pad, 0)])
            pp = self.params(sl(mu))
            xh, ih = f(ps, jnp.array(sl(X)), jnp.array(sl(I).astype(np.int32)), jnp.array(sl(pid).astype(np.int32)),
                       jnp.array(sl(Zf)), pp)
            XH[a:b], IH[a:b] = np.asarray(xh)[:b - a], np.asarray(ih)[:b - a]
        return XH, IH

    def fly(self, pol, name, base):
        """sim pi_0 from the validation starts: 3 runs x 2 laps (sim_jax.rollout; RMA with its deployed history loop)."""
        jax, jnp, sj = self.jax, self.jnp, self.sj
        pl = plan(name)
        st = starts(base, name)
        X0, I0 = zip(*[x_start(pl, s['s0'], s['pert']) for s in st])
        T = int(math.ceil(LAPS * float(pl['lap_time']) / sj.DT))
        pid = jnp.full((len(st),), self.names.index(name), jnp.int32)
        if isinstance(pol, dict):
            ja = [(jnp.array(W), jnp.array(b)) for W, b in pol['adapt']]
            jb = [(jnp.array(W), jnp.array(b)) for W, b in pol['actor']]

            def pr(o, s):
                buf, ap_ = s
                buf = jnp.concatenate([buf[1:], jnp.concatenate([o[:8], ap_])[None]], 0)
                mu = sj.mlp(jb, jnp.concatenate([o, sj.mlp(ja, buf.reshape(-1))]))
                return mu, (buf, mu)
            out = sj.rollout(self.P, pr, T, jnp.array(np.array(X0)), jnp.array(np.array(I0), jnp.int32), pid,
                             self.params(np.full(len(st), design_mu(name))), pstate=(jnp.zeros((len(st), pol['hist'], 10)), jnp.zeros((len(st), 2))))
        else:
            jb = [(jnp.array(W), jnp.array(b)) for W, b in pol]
            out = sj.rollout(self.P, lambda o: sj.mlp(jb, o), T, jnp.array(np.array(X0)), jnp.array(np.array(I0), jnp.int32),
                             pid, self.params(np.full(len(st), design_mu(name))))
        res = []
        for k in range(len(st)):
            fl = np.asarray(out['failed'][k])
            m = int(np.argmax(fl)) if fl.any() else T
            x, idx = np.asarray(out['x'][k][:m]), np.asarray(out['idx'][k][:m])
            obs, a = np.asarray(out['obs'][k][:m]), np.asarray(out['a'][k][:m])
            e = np.abs(np.asarray(out['err'][k][:m, 0]))
            e_pad = np.concatenate([e, np.full(T - m, ENV)]) if fl.any() else e
            z = rma_zhat(pol, obs, a) if isinstance(pol, dict) and m else None
            res.append(dict(x=x, idx=idx, z=z, failed=bool(fl.any()), rms_ey_pad=float(np.sqrt(np.mean(e_pad ** 2))) if m else ENV))
        return res


def tube_base(trajs):
    """Every STRIDE-th period of each trajectory (not the last 8 before a crash), with its plan index and, for the cars'
    trajectories, the full multi-body state there (NaN for the sim's: the car then starts at mb_from_planar)."""
    X, I, XM = [], [], []
    for r in trajs:
        n = len(r['x']) - (8 if r['failed'] else 0)
        ks = np.arange(STRIDE // 2, max(n, 0), STRIDE)
        X += [np.asarray(r['x'])[k] for k in ks]
        I += [int(r['idx'][k]) for k in ks]
        XM += [np.asarray(r['xm'])[k] if 'xm' in r else np.full(mc.NXM + 2, np.nan) for k in ks]
    return np.array(X).reshape(-1, 9), np.array(I, np.int64), np.array(XM).reshape(-1, mc.NXM + 2)


def ctx(trajs, I, M):
    """RMA: per state, the z_hat of the policy's own trajectory (these trajs) at the nearest plan index."""
    tr = [r for r in trajs if r.get('z') is not None and len(r['idx'])]
    if not tr:
        return None
    return nearest_z(np.concatenate([r['idx'] for r in tr]), np.concatenate([r['z'] for r in tr]), I, M)


# ------------------------------------------------------------------------------------------------- compute
def compute(a):
    D = a.stage
    its = json.load(open(os.path.join(D, 'its', 'its.json')))
    K = sorted(int(k) for k in its)
    summ = json.load(open(os.path.join(D, 'summary.json')))
    cars = a.cars or json.load(open(os.path.join(D, 'args.json')))['cars']
    rungs = summ.get('rungs', {})
    pol_path = lambda c, k: os.path.join(D, 'its', f'it{k}_{c}_policy.npz')
    pi0 = load_pol(pol_path(cars[0], 0))
    is_rma = isinstance(pi0, dict)
    plans_c = {c: {s: [on_rung(b, rungs.get(c)) for b in SETS[s]] for s in SETS} for c in cars}
    base_of = {}
    for c in cars:
        for s in SETS:
            for b, n in zip(SETS[s], plans_c[c][s]):
                base_of[(c, n)] = b
    names = sorted({n for c in cars for s in SETS for n in plans_c[c][s]})
    sim = Sim(names)
    # 1. sim pi_0 trajectories (car independent)
    sim_tr = {n: sim.fly(pi0, n, next(b for (c, m), b in base_of.items() if m == n)) for n in names}
    print(json.dumps(dict(stage='sim flown', plans=len(names))), flush=True)
    # 2. the cars under every pi_i: the validation runs (task error + the tube)
    jobs = []
    for c in cars:
        for k in K:
            for s in SETS:
                for n in plans_c[c][s]:
                    for st in starts(base_of[(c, n)], n):
                        jobs.append(dict(car=c, plan=n, pol=pol_path(c, k), it=k, **st))
    res = pmap(w_fly, jobs, a.procs)
    real_tr = {}
    for j, r in zip(jobs, res):
        real_tr.setdefault((j['car'], j['plan'], j['it']), []).append(r)
    print(json.dumps(dict(stage='cars flown', runs=len(jobs))), flush=True)
    # 3. the tube per (car, plan): sim pi_0 + every pi_i on the car; perturbed copies; off-tube for IN
    tube = {}
    for c in cars:
        for s in SETS:
            for n in plans_c[c][s]:
                Xb, Ib, XMb = tube_base(sim_tr[n] + sum((real_tr[(c, n, k)] for k in K), []))
                rng = np.random.default_rng(SEED + 1000 * cars.index(c) + names.index(n))
                Xp = perturb(Xb, PERT, rng)
                X, I = np.concatenate([Xb, Xp]), np.concatenate([Ib, Ib])
                XM = np.concatenate([XMb, np.array([mb_add(m, d) for m, d in zip(XMb, Xp - Xb)]).reshape(XMb.shape)])
                kind = np.r_[np.zeros(len(Xb), int), np.ones(len(Xb), int)]          # 0 visited, 1 perturbed
                if s == 'IN':
                    Xo = perturb(Xb, PERT_OFF, rng)
                    X, I = np.concatenate([X, Xo]), np.concatenate([I, Ib])
                    XM = np.concatenate([XM, np.array([mb_add(m, d) for m, d in zip(XMb, Xo - Xb)]).reshape(XMb.shape)])
                    kind = np.r_[kind, np.full(len(Xb), 2)]                          # 2 off-tube
                tube[(c, n)] = (X, I, kind, XM)
    # 4. the sim fields at every tube state: pi_0 (the reference) and the plans' LQR (the plan's flow)
    keys = list(tube)
    Xall = np.concatenate([tube[k][0] for k in keys])
    Iall = np.concatenate([tube[k][1] for k in keys])
    Nall = sum([[k[1]] * len(tube[k][0]) for k in keys], [])
    Zs = np.concatenate([ctx(sim_tr[k[1]], tube[k][1], pnp(k[1]).M) for k in keys]) if is_rma else None
    XH0, IH0 = sim.map(pi0, Xall, Iall, Nall, Zs)
    XHl, IHl = sim.map(None, Xall, Iall, Nall, kind='lqr')
    out, o = {}, 0
    for k in keys:
        X, I, kind, _ = tube[k]
        n = len(X)
        E0 = errs(k[1], X, I)
        out[k] = dict(X=X, I=I, kind=kind, E0=E0, Vsim=errs(k[1], XH0[o:o + n], IH0[o:o + n]) - E0,
                      Vlqr=errs(k[1], XHl[o:o + n], IHl[o:o + n]) - E0, Vcar={})
        o += n
    print(json.dumps(dict(stage='sim fields', states=len(Xall))), flush=True)
    # 5. the car fields under every pi_i at the same states
    jobs, jk = [], []
    for c in cars:
        for k in K:
            for s in SETS:
                for n in plans_c[c][s]:
                    X, I, _, XM = tube[(c, n)]
                    Z = ctx(real_tr[(c, n, k)], I, pnp(n).M) if is_rma else None
                    for a0 in range(0, len(X), 64):
                        jobs.append(dict(car=c, plan=n, pol=pol_path(c, k), X=X[a0:a0 + 64], I=I[a0:a0 + 64], XM=XM[a0:a0 + 64],
                                         Z=None if Z is None else Z[a0:a0 + 64]))
                        jk.append((c, n, k))
    res = pmap(w_map, jobs, a.procs)
    parts = {}
    for key, (XH, IH) in zip(jk, res):
        parts.setdefault(key, []).append((XH, IH))
    for (c, n, k), pp in parts.items():
        XH, IH = np.concatenate([p[0] for p in pp]), np.concatenate([p[1] for p in pp])
        out[(c, n)]['Vcar'][k] = errs(n, XH, IH) - out[(c, n)]['E0']
    print(json.dumps(dict(stage='car fields', jobs=len(jobs))), flush=True)
    # 6. save: raw fields (npz) + per car / set / it metrics (json)
    os.makedirs(a.dest, exist_ok=True)
    arr = {}
    for (c, n), d in out.items():
        p = f'{c}|{n}|'
        for f in ('X', 'I', 'kind', 'E0', 'Vsim', 'Vlqr'):
            arr[p + f] = d[f]
        for k, V in d['Vcar'].items():
            arr[p + f'Vcar{k}'] = V
    np.savez_compressed(os.path.join(a.dest, f'{a.name}.npz'), **arr)
    met = dict(name=a.name, stage=D, iters=K, cars=cars, plans=plans_c, H=H, stride=STRIDE, seed=SEED, rma=is_rma,
               sim_task={n: float(np.mean([r['rms_ey_pad'] for r in sim_tr[n]])) for n in names}, per={})
    for c in cars:
        met['per'][c] = {}
        for s in SETS:
            row = dict(field={}, field_task={}, field_plan={}, field_vis={}, field_off={}, field_ey={}, change={}, task={}, success={},
                       ref_plan={})
            for k in K:
                F, FT, FP, FV, FO, FE, CH = [], [], [], [], [], [], []
                for n in plans_c[c][s]:
                    d = out[(c, n)]
                    dV = d['Vcar'][k] - d['Vsim']
                    on = d['kind'] < 2
                    nrm = np.sqrt(np.nanmean((dV[:, :5] / S5) ** 2, 1))
                    F.append(np.nanmean(nrm[on]))
                    FE.append(np.nanmean(np.abs(dV[on][:, 0])))                                   # |dV| in e_y alone (m)
                    CH.append(np.nanmean(np.sqrt(np.nanmean(((d['Vcar'][k] - d['Vcar'][K[0]])[on][:, :5] / S5) ** 2, 1))))
                    FV.append(np.nanmean(nrm[d['kind'] == 0]))
                    if (d['kind'] == 2).any():
                        FO.append(np.nanmean(nrm[d['kind'] == 2]))
                    FT.append(np.nanmean(np.sqrt(np.nanmean((dV[on][:, [0, 5]] / TASK_SCALE[[0, 5]]) ** 2, 1))))
                    dP = d['Vcar'][k] - d['Vlqr']
                    FP.append(np.nanmean(np.sqrt(np.nanmean((dP[on][:, :5] / S5) ** 2, 1))))
                row['field'][k], row['field_task'][k], row['field_plan'][k] = map(float, (np.mean(F), np.mean(FT), np.mean(FP)))
                row['field_vis'][k] = float(np.mean(FV))
                row['field_ey'][k], row['change'][k] = float(np.mean(FE)), float(np.mean(CH))
                row['field_off'][k] = float(np.mean(FO)) if FO else None
                rr = sum((real_tr[(c, n, k)] for n in plans_c[c][s]), [])
                row['task'][k] = float(np.mean([r['rms_ey_pad'] for r in rr]))
                row['success'][k] = float(np.mean([r['success'] for r in rr]))
            row['ref_plan'] = float(np.mean([np.mean(np.sqrt(np.mean(((out[(c, n)]['Vsim'] - out[(c, n)]['Vlqr'])[out[(c, n)]['kind'] < 2][:, :5] / S5) ** 2, 1)))
                                             for n in plans_c[c][s]]))
            row['sim_task'] = float(np.mean([met['sim_task'][n] for n in plans_c[c][s]]))
            met['per'][c][s] = row
        print(json.dumps({c: {s: [round(met['per'][c][s]['field'][k], 3) for k in K] for s in SETS}}), flush=True)
    json.dump(met, open(os.path.join(a.dest, f'{a.name}.json'), 'w'), indent=1)


# ------------------------------------------------------------------------------------------------- phase portraits
def portrait(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    D = a.stage
    its = json.load(open(os.path.join(D, 'its', 'its.json')))
    K = sorted(int(k) for k in its)
    summ = json.load(open(os.path.join(D, 'summary.json')))
    name = on_rung(a.plan, summ.get('rungs', {}).get(a.car))
    pl, p = plan(name), pnp(name)
    pol_path = lambda k: os.path.join(D, 'its', f'it{k}_{a.car}_policy.npz')
    pi0 = load_pol(pol_path(0))
    is_rma = isinstance(pi0, dict)
    z = np.asarray(pl['z'])
    i_s = int(np.argmax(np.abs(z[:, 4]) * (np.abs(pl['kappa_s']) > 0.5 * np.abs(pl['kappa_s']).max())))   # mid-corner
    if a.s is not None:
        i_s = int(np.round(a.s / float(pl['ds']))) % len(z)
    xp = np.asarray(pl['x'][i_s]).copy()
    th = float(pl['theta'][i_s])
    show = list(dict.fromkeys(k for k in (0, 1, 2, 3, K[-1]) if k in K))
    sim = Sim([name])
    G = a.grid
    # the trajectories (for z_hat context and the lap-to-lap crossings of the section)
    sim_tr = sim.fly(pi0, name, a.plan)
    real_tr = {k: pmap(w_fly, [dict(car=a.car, plan=name, pol=pol_path(k), **st) for st in starts(a.plan, name)], a.procs) for k in show}
    def proj2(kind, X, I):
        E = errs(name, X, I)
        if kind == 'path':     # plan-relative lateral (m) and heading (rad) errors
            return np.c_[E[:, 0], E[:, 1]]
        bp = np.degrees(np.arctan2(z[np.asarray(I) % len(z), 3], z[np.asarray(I) % len(z), 2]))
        return np.c_[np.degrees(np.arctan2(X[:, 4], X[:, 3])) - bp, E[:, 4]]   # sideslip (deg) and yaw rate vs the plan's
    HW = {'path': (0.15, 0.3), 'drift': (15.0, 1.5)}
    LAB = {'path': ('lateral error vs plan (m)', 'heading error vs plan (rad)'),
           'drift': ('sideslip vs plan (deg)', 'yaw rate vs plan (rad/s)')}
    u = np.linspace(-1, 1, G)
    U1, U2 = np.meshgrid(u, u)
    panels = []
    for proj in ('path', 'drift'):
        X = np.repeat(xp[None], G * G, 0)
        if proj == 'path':      # lateral offset (m) and heading offset (rad) from the plan point
            dy, dpsi = HW[proj][0] * U1.ravel(), HW[proj][1] * U2.ravel()
            X[:, 0] += -np.sin(th) * dy
            X[:, 1] += np.cos(th) * dy
            X[:, 2] += dpsi
        else:                   # sideslip and yaw rate about the plan's drift equilibrium (speed and heading at the plan)
            V = math.hypot(xp[3], xp[4])
            bb = math.atan2(xp[4], xp[3]) + np.radians(HW[proj][0]) * U1.ravel()
            X[:, 3], X[:, 4] = V * np.cos(bb), V * np.sin(bb)
            X[:, 5] = xp[5] + HW[proj][1] * U2.ravel()
        I = np.full(len(X), i_s)
        P0 = proj2(proj, X, I)
        Zs = ctx(sim_tr, I, p.M) if is_rma else None
        XH, IH = sim.map(pi0, X, I, [name] * len(X), Zs)
        fields = {'sim pi_0': proj2(proj, XH, IH) - P0}
        cross = {'sim pi_0': sim_tr}
        for k in show:
            Zc = ctx(real_tr[k], I, p.M) if is_rma else None
            XH, IH = w_map(dict(car=a.car, plan=name, pol=pol_path(k), X=X, I=I, Z=Zc))
            fields[f'{a.car} pi_{k}'] = proj2(proj, XH, IH) - P0
            cross[f'{a.car} pi_{k}'] = real_tr[k]
        cr = {}
        for lab, tr in cross.items():
            pts = [(np.asarray(t['x'])[m], np.asarray(t['idx'])[m]) for t in tr for m in
                   [np.abs(((np.asarray(t['idx']) - i_s) + p.M // 2) % p.M - p.M // 2) <= 3] if m.any()]
            cr[lab] = proj2(proj, np.concatenate([q[0] for q in pts]), np.concatenate([q[1] for q in pts])) if pts else np.zeros((0, 2))
        panels.append((proj, P0, fields, cr))
    os.makedirs(os.path.join(OUT, 'figs'), exist_ok=True)
    np.savez_compressed(os.path.join(OUT, 'figs', f'portrait_{a.name}_{a.car}.npz'),
                        **{f'{proj}|{lab}': V for proj, P0, f, cr in panels for lab, V in f.items()},
                        **{f'{proj}|P0': P0 for proj, P0, f, cr in panels})
    labs = list(panels[0][2])
    fig, ax = plt.subplots(2, len(labs), figsize=(2.9 * len(labs), 6.0), squeeze=False)
    for r, (proj, P0, fields, cr) in enumerate(panels):
        hx, hy = HW[proj]
        mags = {lab: np.hypot(V[:, 0] / hx, V[:, 1] / hy) for lab, V in fields.items()}
        vmax = max(np.quantile(m, 0.95) for m in mags.values())
        for j, lab in enumerate(labs):
            A = ax[r, j]
            V = fields[lab]
            gx, gy = (V[:, 0] / hx).reshape(G, G), (V[:, 1] / hy).reshape(G, G)
            A.streamplot(u, u, gx, gy, color=mags[lab].reshape(G, G), cmap='viridis', norm=matplotlib.colors.Normalize(0, vmax),
                         density=0.7, linewidth=0.9, arrowsize=0.8)
            A.plot(0, 0, marker='*', ms=13, color='crimson', mec='k', zorder=5)
            c_ = cr[lab]
            if len(c_):
                A.plot(c_[:, 0] / hx, c_[:, 1] / hy, 'o', ms=4, mfc='orange', mec='k', zorder=6)
            A.set_xlim(-1.05, 1.05); A.set_ylim(-1.05, 1.05)
            A.set_xticks([-1, 0, 1]); A.set_xticklabels([f'{-hx:g}', '0', f'{hx:g}'])
            A.set_yticks([-1, 0, 1]); A.set_yticklabels([f'{-hy:g}', '0', f'{hy:g}'])
            A.set_title(f'{lab}  (mean |V| {np.mean(mags[lab]):.2f})', fontsize=8)
            A.set_xlabel(LAB[proj][0], fontsize=8)
            if j == 0:
                A.set_ylabel(LAB[proj][1], fontsize=8)
            A.tick_params(labelsize=7)
    fig.suptitle(f'{a.name}: closed-loop field (0.1 s map) at mid-corner of {name} (s = {i_s * float(pl["ds"]):.2f} m); '
                 'star = plan, orange = where the 3 x 2-lap runs cross this section', fontsize=9)
    fig.tight_layout()
    f = os.path.join(OUT, 'figs', f'portrait_{a.name}_{a.car}.png')
    fig.savefig(f, dpi=130)
    print(f)


# ------------------------------------------------------------------------------------------------- report
STARTS = {'ours': 'v3drBtL', 'PPO+DR': 'ppoL_task', 'RMA': 'rmaL_task', 'ours (nominal sim, extra)': 'v3nom'}


def boot_ci(M, B=10000):
    """M (cars, iters): mean over cars and a 95 % percentile bootstrap over cars."""
    M = np.asarray(M, float)
    rng = np.random.default_rng(0)
    bs = M[rng.integers(0, len(M), (B, len(M)))].mean(1)
    return M.mean(0), np.quantile(bs, 0.025, 0), np.quantile(bs, 0.975, 0)


def report(a):
    import glob
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    res = {}
    for f in sorted(glob.glob(os.path.join(OUT, 'fields', '*.json'))):
        m = json.load(open(f))
        st = next((s for s, p in STARTS.items() if m['name'].startswith(p + '_s')), None)
        if st:
            res.setdefault(st, []).append(m)
    H_ = json.load(open(sorted(glob.glob(os.path.join(OUT, 'fields', '*.json')))[0])).get('H', H)
    summary = dict(domain='car', map=f'{H_}-period ({H_ * 0.025:g} s) closed-loop map in plan-relative error coordinates (e_y, e_psi, vx, vy, r '
                   'minus the plan at the car\'s arc length)', proj=['lateral error e_y (m)', 'heading error e_psi (rad)'],
                   starts={}, notes='')
    extra = {}
    for st, ms in res.items():
        K = ms[0]['iters']
        robots, F, T = [], {s: [] for s in SETS}, {s: [] for s in SETS}
        ex = {q: {s: [] for s in SETS} for q in ('field_task', 'field_plan', 'field_vis', 'field_off', 'field_ey', 'change', 'success', 'ref_plan', 'sim_task')}
        for m in ms:
            seed = m['name'].split('_s')[-1].split('_')[0]
            for c in m['cars']:
                robots.append(f'{c}@s{seed}')
                for s in SETS:
                    r = m['per'][c][s]
                    F[s].append([r['field'][str(k)] for k in K])
                    T[s].append([r['task'][str(k)] for k in K])
                    for q in ex:
                        v = r[q]
                        ex[q][s].append([v.get(str(k)) for k in K] if isinstance(v, dict) else v)
        summary['starts'][st] = dict(iters=K, robots=robots, field_err=F, task_err=T)
        extra[st] = dict(iters=K, robots=robots, **ex, seeds=sorted({m['name'] for m in ms}))
    summary['notes'] = ('F1TENTH drift car. Robots = car@sim-seed (8 multi-body cars x 3 sim seeds; v3nom: 1 seed). field_err = mean '
                        'over the tube of the RMS over (e_y, e_psi, vx, vy, r) of (V^car_{pi_i} - V^sim_{pi_0}) / (0.1 m, 0.2 rad, 0.5, '
                        '0.5 m/s, 1 rad/s), V over a ' + f'{H_ * 0.025:g} s map; reference sim pi_0 at the plan\'s design friction. task_err = RMS lateral error (m) of 3 x 2-lap validation runs '
                        '(seed 301), crashed periods at the 0.3 m envelope. IN = b25 (trained goals), HULL = b21 of the same '
                        'tracks, OUT = b0 of the same tracks + held-out track squareH_b25; each on the car\'s final ladder rung. '
                        'CIs: bootstrap over cars (seeds averaged per car). See FLOWS.md.')
    json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
    json.dump(extra, open(os.path.join(OUT, 'summary_extra.json'), 'w'), indent=1)
    # per-car cluster means (seeds averaged) for CIs
    def per_car(st, rows):
        robots = summary['starts'][st]['robots']
        cars = sorted({r.split('@')[0] for r in robots})
        return np.array([np.mean([rows[k] for k, r in enumerate(robots) if r.split('@')[0] == c], 0) for c in cars])
    cols = {'ours': '#1f77b4', 'PPO+DR': '#d62728', 'RMA': '#2ca02c', 'ours (nominal sim, extra)': '#7f7f7f'}
    for key, fname, yl in (('field_err', 'field_err.png', 'field error |dV| (scaled RMS, per 0.1 s)'),
                           ('task_err', 'task_err.png', 'RMS lateral error (m), crashes at 0.3 m')):
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
        for j, s in enumerate(SETS):
            for st in summary['starts']:
                K = summary['starts'][st]['iters']
                m, lo, hi = boot_ci(per_car(st, summary['starts'][st][key][s]))
                ax[j].plot(K, m, 'o-', color=cols.get(st), label=st)
                ax[j].fill_between(K, lo, hi, color=cols.get(st), alpha=0.15)
            ax[j].set_title(s)
            ax[j].set_xlabel('hardware-stage iteration (2 real laps each)')
            ax[j].grid(alpha=0.3)
        ax[0].set_ylabel(yl)
        ax[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, 'figs', fname), dpi=130)
    # the table for FLOWS.md
    lines = []
    for st in summary['starts']:
        K = summary['starts'][st]['iters']
        for s in SETS:
            for key in ('field_err', 'task_err'):
                m, lo, hi = boot_ci(per_car(st, summary['starts'][st][key][s]))
                lines.append(f'| {st} | {s} | {key} | {m[0]:.3f} [{lo[0]:.3f}, {hi[0]:.3f}] | {m[-1]:.3f} [{lo[-1]:.3f}, {hi[-1]:.3f}] | '
                             f'{100 * (m[-1] / m[0] - 1):+.0f} % |')
        for q, s in [('field_off', 'IN')] + [(q, s) for q in ('field_task', 'field_ey', 'change', 'field_plan', 'success') for s in SETS]:
            m, lo, hi = boot_ci(per_car(st, extra[st][q][s]))
            lines.append(f'| {st} | {s} | {q} | {m[0]:.3f} [{lo[0]:.3f}, {hi[0]:.3f}] | {m[-1]:.3f} [{lo[-1]:.3f}, {hi[-1]:.3f}] | '
                         f'{100 * (m[-1] / m[0] - 1):+.0f} % |')
        for q in ('ref_plan', 'sim_task'):
            v = [np.mean(extra[st][q][s]) for s in SETS]
            lines.append(f'| {st} | IN/HULL/OUT | {q} (sim pi_0) | ' + ' / '.join(f'{x:.3f}' for x in v) + ' | | |')
    open(os.path.join(OUT, 'table.md'), 'w').write('| start | set | metric | it 0 [95 % CI] | final [95 % CI] | change |\n'
                                                   '|---|---|---|---|---|---|\n' + '\n'.join(lines) + '\n')
    print(open(os.path.join(OUT, 'table.md')).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('compute', 'portrait', 'report'))
    ap.add_argument('--stage', default='')
    ap.add_argument('--name', default='')
    ap.add_argument('--cars', nargs='+', default=None)
    ap.add_argument('--car', default='real_mu')
    ap.add_argument('--plan', default='mocap_square2fast_b25')
    ap.add_argument('--s', type=float, default=None)
    ap.add_argument('--grid', type=int, default=11)
    ap.add_argument('--procs', type=int, default=5)
    ap.add_argument('--dest', default=os.path.join(OUT, 'fields'))
    ap.add_argument('--H', type=int, default=None, help='control periods per map (default 20 = 0.5 s; or $FLOWS_H)')
    ap.add_argument('--stride', type=int, default=None, help='tube subsampling in periods (default 40; or $FLOWS_STRIDE)')
    a = ap.parse_args()
    if a.stride:
        global STRIDE
        STRIDE = a.stride
        os.environ['FLOWS_STRIDE'] = str(a.stride)
    if a.H:
        global H
        H = a.H
        os.environ['FLOWS_H'] = str(a.H)                 # the spawned workers re-import with it
    if a.stage and not a.name:
        a.name = os.path.basename(a.stage.rstrip('/'))
    {'compute': compute, 'portrait': portrait, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()
