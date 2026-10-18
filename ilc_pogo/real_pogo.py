"""
real_pogo.py: the "real" pogo sticks -- an independent numpy simulator on the CPU (vectorized over runs) of the same
2-D SLIP pogo stick, with each robot's own dynamics, 0.1 ms steps, and IDEALIZED sensing: the policy reads the exact
state (perfect mocap), no latency, no actuator delay. The structural gap to the nominal model: leg damping (the
nominal has none) and the finer integration (touchdown / liftoff timing); then each robot's perturbation.

Robots (ROBOTS; 'ax:<key>=<value>' builds a single-axis robot on top of real_nom, e.g. ax:k=0.6 multiplies k):
  real_nom     damping 15 N s/m only
  real_mass    + mass x1.2
  real_spring  + spring stiffness x0.8 (a worn spring)
  real_act     + thrust gain x0.75 (a weak actuator)
  real_offset  + leg-angle calibration offset +0.05 rad (outside the DR family's +-0.03)
  real_prog    + progressive spring, k (1 + 1.5 c / l0) (outside the DR family: a different spring law)

Unmodelled-dynamics suite (UNMODELLED; 2026-10-18): effects the nominal model has no parameter for (not in the DR
family either). Each on top of real_nom; the nominal sim (common.substep, pogo_jax) is unchanged, and with these keys at
their defaults (0) the real substep below reduces exactly to common.substep.
  real_dead    thrust dead zone 6 mm (a command delta < dz gives no thrust; above, delta - dz) after a backlash
               (mechanical play) of 6 mm on the thrust actuator; thrust range 0-80 mm
  real_bias    constant sensor bias on the policy's observed state: height +2 cm, vx +0.1 m/s, vz +0.1 m/s (the
               policy sees the biased state; the trial logs (x, err) and the scoring use the true state, as mocap)
  real_lag     first-order thrust actuator lag, tau 20 ms (2 control periods; stance lasts ~160 ms)
  real_ground  compliant ground: a spring-damper under the foot (kg 15 kN/m, bg 150 N s/m; massless foot), the foot
               sinks in stance and the leg force acts from the sunk foot
  real_legm    leg mass 1 kg at the foot (unsprung; the body stays 10 kg): pinned in stance, at liftoff the body must
               drag it up (a plastic collision: the body's velocity x m / (m + mf)); its swing in flight is neglected
"""
import numpy as np

import common as C
from common import NOMINAL, NX, DT, features, errors, failed, substep, hop_metrics

SUB_REAL = 100
REAL_ONLY = dict(dz=0.0, bl=0.0, tau=0.0, bz=0.0, bvx=0.0, bvz=0.0, kg=0.0, bg=0.0, mf=0.0)
#   dz / bl: thrust dead zone / backlash (m), tau: thrust lag (s), bz / bvx / bvz: sensor bias on z / vx / vz,
#   kg / bg: ground stiffness / damping under the foot (kg 0 = rigid), mf: leg (foot) mass (kg)
BASE = dict(NOMINAL, b=15.0, **REAL_ONLY)
ROBOTS = {
    'real_nom': {},
    'real_mass': dict(m=1.2),
    'real_spring': dict(k=0.8),
    'real_act': dict(gth=0.75),
    'real_offset': dict(phi_off=('add', 0.05)),
    'real_prog': dict(kp=('add', 1.5)),
}
DEFAULT_ROBOTS = list(ROBOTS)
UNMODELLED = {
    'real_dead': dict(dz=('set', 0.006), bl=('set', 0.006)),
    'real_bias': dict(bz=('set', 0.02), bvx=('set', 0.1), bvz=('set', 0.1)),
    'real_lag': dict(tau=('set', 0.02)),
    'real_ground': dict(kg=('set', 15000.0), bg=('set', 150.0)),
    'real_legm': dict(mf=('set', 1.0)),
}
UNMODELLED_ROBOTS = list(UNMODELLED)
ROBOTS.update(UNMODELLED)                         # robot_params knows them; DEFAULT_ROBOTS stays the original six
MULT = ('m', 'k', 'gth', 'l0')


def robot_params(name):
    """Parameter dict of a robot: ROBOTS entries multiply (m, k, gth, l0) or add ('add', v) / set ('set', v);
    'ax:key=v' multiplies (or, for b / phi_off / kp, sets) one key of real_nom."""
    p = dict(BASE)
    if name.startswith('ax:'):
        key, v = name[3:].split('=')
        v = float(v)
        p[key] = p[key] * v if key in MULT else v
        return p
    for key, v in ROBOTS[name].items():
        if isinstance(v, tuple):
            p[key] = p[key] + v[1] if v[0] == 'add' else v[1]
        else:
            p[key] = p[key] * v
    return p


from policies import as_policy  # noqa: E402


def simulate(policy, goals, x0, params, T, noise=None, ff=None):
    """Runs n pogo sticks (rows) for T control periods. params: a dict of per-run arrays (or scalars); ff (n, T, 2):
    an additive feedforward (the classical per-goal ILC). Returns dict of x (n, T+1, 6), obs, mu, a, err (n, T, .),
    failed (n, T) (sticky; a run freezes at its fall)."""
    pol = as_policy(policy)
    goals = np.asarray(goals, np.float64)
    x = np.asarray(x0, np.float64).copy()
    n = len(x)
    pol.reset(n)
    p = {k: np.broadcast_to(np.asarray(v, np.float64), (n,)) for k, v in dict(REAL_ONLY, **params).items()}
    xs = np.zeros((n, T + 1, NX))
    O, MU, A, E = (np.zeros((n, T, d)) for d in (len(features(x[:1], goals[:1], np)[0]), 2, 2, 2))
    F = np.zeros((n, T), bool)
    f = np.zeros(n, bool)
    EV = [[] for _ in range(n)]                       # per run: (period, fraction of the period, 1 touchdown / 0 liftoff)
    h = DT / SUB_REAL
    real = any(np.any(p[k] != 0) for k in REAL_ONLY)  # any unmodelled effect in the batch -> the real substep
    bias = np.stack([np.zeros(n), p['bz'], p['bvx'], p['bvz'], np.zeros(n), np.zeros(n)], -1)
    aux = dict(yb=None, dl=None, zf=np.zeros(n), zfd=np.zeros(n))
    for t in range(T):
        xs[:, t] = x
        xo = x + bias                                  # what the robot's sensors report to the policy
        o = features(xo, goals, np)
        mu = pol(o, xo, goals)
        a = mu + (0 if noise is None else noise[:, t]) + (0 if ff is None else ff[:, t])
        O[:, t], MU[:, t], A[:, t], E[:, t], F[:, t] = o, mu, a, errors(x, goals, np), f
        if real:
            dbl = thrust_chain(a, p, aux)
        xn = x
        for j in range(SUB_REAL):
            st0 = xn[:, 5]
            xn = substep_real(xn, a, dbl, p, h, aux) if real else substep(xn, a, p, h, np)
            ch = np.where((xn[:, 5] != st0) & ~f)[0]
            for i in ch:                              # the mode signature: event, its control period and sub-step
                EV[i].append((t, (j + 1) / SUB_REAL, int(xn[i, 5] > 0.5)))
        fn = f | failed(xn, np)
        x = np.where(fn[:, None], x, xn)
        f = fn
    xs[:, T] = x
    return dict(x=xs, obs=O, mu=MU, a=A, err=E, failed=F, failed_end=f, events=EV)


def thrust_chain(a, p, aux):
    """Once per control period: the commanded thrust delta through the actuator's backlash (play bl: the output moves
    only when the command pushes against it) and dead zone (dz) -> the thrust the lag (substep_real) then follows."""
    delta, _ = C.unnorm(a, np)
    if aux['yb'] is None:
        aux['yb'] = delta.copy()
    aux['yb'] = np.clip(aux['yb'], delta - 0.5 * p['bl'], delta + 0.5 * p['bl'])
    return np.maximum(aux['yb'] - p['dz'], 0.0)


def substep_real(x, a, dbl, p, h, aux):
    """common.substep plus the unmodelled effects (UNMODELLED): the thrust lag (tau), the compliant ground under the
    foot (kg, bg: the foot at height zf <= 0, massless) and the leg mass (mf) at liftoff. With those keys 0 it computes
    exactly common.substep (numpy: the straight-through saltation terms are 0 there)."""
    _, phi = C.unnorm(a, np)
    phi = phi + p['phi_off']
    if aux['dl'] is None:
        aux['dl'] = dbl.copy()
    lag = p['tau'] > 0
    aux['dl'] = np.where(lag, aux['dl'] + h / np.where(lag, p['tau'], 1.0) * (dbl - aux['dl']), dbl)
    delta = aux['dl']
    zf, zfd = aux['zf'], aux['zfd']
    X, Z, VX, VZ, XF, ST = (x[..., i] for i in range(6))
    dx, dzl = X - XF, Z - zf
    L = np.sqrt(dx ** 2 + dzl ** 2)
    Ls = np.maximum(L, 1e-6)
    Ld = (dx * VX + dzl * (VZ - zfd)) / Ls
    comp = np.maximum(p['l0'] - L, 0.0)
    F = p['k'] * (1.0 + p['kp'] * comp / p['l0']) * (p['l0'] + p['gth'] * delta - L) - p['b'] * Ld
    F = np.maximum(F, 0.0) * ST
    ax = F * dx / Ls / p['m']
    az = F * dzl / Ls / p['m'] - C.GRAV
    VX, VZ = VX + h * ax, VZ + h * az
    X, Z = X + h * VX, Z + h * VZ
    soft = (p['kg'] > 0) & (ST > 0.5)                 # the massless foot: ground force -kg zf - bg zfd = leg force (z)
    zfd = np.where(soft, (-p['kg'] * zf - F * dzl / Ls) / np.where(p['bg'] > 0, p['bg'], 1.0), 0.0)
    zf = np.where(soft, np.minimum(zf + h * zfd, 0.0), 0.0)
    td = (ST < 0.5) & (Z - p['l0'] * np.cos(phi) <= 0.0) & (VZ < 0)
    Ln = np.sqrt((X - XF) ** 2 + (Z - zf) ** 2)
    lo = (ST > 0.5) & (Ln >= p['l0'] + p['gth'] * delta) & (Ld > 0)
    XF = np.where(td, X + p['l0'] * np.sin(phi), XF)
    drag = p['m'] / (p['m'] + p['mf'])                # liftoff: the body picks up the leg mass
    VX, VZ = np.where(lo, VX * drag, VX), np.where(lo, VZ * drag, VZ)
    zf, zfd = np.where(td | lo, 0.0, zf), np.where(td | lo, 0.0, zfd)
    aux['zf'], aux['zfd'] = zf, zfd
    ST = np.where(td, 1.0, np.where(lo, 0.0, ST))
    return np.stack([X, Z, VX, VZ, XF, ST], -1)


def signature(res, i, a_lim=1.0):
    """Mode signature of run i: the event sequence (type, control period) -- the trial map's piece -- plus the
    clip pattern (periods with a saturated action component); the margins: the smallest distance of an event from a
    control-period boundary (fraction of a period) and the smallest |1 - |a|| over unsaturated actions."""
    ev = res['events'][i]
    a = np.asarray(res['a'][i])
    sat = np.abs(a) >= a_lim
    t_margin = min([min(fr, 1.0 - fr) for _, fr, _ in ev] + [0.5])
    free = ~sat & ~res['failed'][i][:, None]
    a_margin = float(np.min(np.abs(a_lim - np.abs(a[free])))) if free.any() else 0.0
    return dict(events=[(int(t), int(k)) for t, _, k in ev], n_td=int(sum(k for _, _, k in ev)),
                fell=bool(res['failed_end'][i]), sat=sat, t_margin=float(t_margin), a_margin=a_margin)


def same_cell(s0, s1):
    """(topological: same event types in the same order and no fall; piece: also every event in the same control
    period and the same clip pattern)."""
    # events are compared on their common prefix: a hop that moves past the trial's end (the hop period changed) is
    # the horizon truncating the trial, not a change of the dynamics' mode sequence
    n = min(len(s0['events']), len(s1['events']))
    t0, t1 = [k for _, k in s0['events'][:n]], [k for _, k in s1['events'][:n]]
    topo = t0 == t1 and s0['fell'] == s1['fell'] and abs(len(s0['events']) - len(s1['events'])) <= 2
    piece = topo and s0['events'][:n] == s1['events'][:n] and bool(np.array_equal(s0['sat'], s1['sat']))
    return topo, piece


def start_states(goals, pert, rs):
    """numpy twin of pogo_jax.start_states (drops at the goal's apex, perturbed)."""
    goals = np.asarray(goals)
    n = len(goals)
    d = rs.uniform(-1, 1, (n, 3)) * np.broadcast_to(pert, (n,))[:, None]
    x0 = np.zeros((n, NX))
    x0[:, 1] = goals[:, 1] + 0.05 * d[:, 0]
    x0[:, 2] = goals[:, 0] + 0.4 * d[:, 1]
    x0[:, 3] = 0.3 * d[:, 2]
    return x0


def evaluate(policies, goals, robots, runs=4, seconds=10.0, seed=None, pert=1.0):
    """policies: {robot: policy}; every robot flies every goal from `runs` perturbed drops (reserved seed), for
    `seconds` without a reset -> {robot: [per (goal, run) hop_metrics + goal]}."""
    T = int(round(seconds / DT))
    seed = C.EVAL_SEED if seed is None else seed
    out = {}
    for r in robots:
        rs = np.random.default_rng(seed)
        G = np.repeat(np.asarray(goals), runs, 0)
        x0 = start_states(G, pert, rs)
        pp = robot_params(r)
        res = simulate(policies[r], G, x0, pp, T)
        ms = []
        for i in range(len(G)):
            m = hop_metrics(res['x'][i], G[i], res['failed_end'][i])
            m['goal'] = G[i].tolist()
            ms.append(m)
        out[r] = ms
    return out


def summarize(ms):
    ok = [m for m in ms if not m['failed']]
    mean = lambda k: float(np.mean([abs(m[k]) for m in ok if m[k] is not None])) if ok else None
    return dict(success=float(np.mean([m['success'] for m in ms])), fail=float(np.mean([m['failed'] for m in ms])),
                rms_apex=mean('rms_apex'), v_err=mean('v_err'), n=len(ms))
