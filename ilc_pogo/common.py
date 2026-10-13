"""
Shared definitions of the pogo-stick study: the simplest 2-D pogo stick, a spring-loaded inverted pendulum (SLIP).
Kept jax-free so the CPU "real" pogo sticks never touch a GPU; functions take an array module `xp` (numpy / jax.numpy).

The pogo stick: a point mass m (body + rider) on a massless spring leg of rest length l0 in the x-z plane.
  flight  ballistic; the massless leg points along the commanded angle phi (CCW from straight down: phi > 0 puts the
          foot ahead); touchdown when the foot l0 (sin phi, -cos phi) below the body reaches the ground
  stance  the foot is pinned (perfect friction); the leg force k (l0 + delta - L) (+ the real sticks' damping) acts
          along the leg, L = |body - foot|; delta (the thrust: the spring's rest length pushed out by the actuator,
          Raibert's pneumatic thrust) is commanded; liftoff when the spring is back at rest while extending
State x (6): [x, z, vx, vz, xf, stance]  (xf the pinned foot, stance 0 / 1)
Action a (2), normalized to [-1, 1]: delta = D_OFF + D_SCALE a0 (m), phi = PHI_SCALE a1 (rad, used in flight).
Goal g (2): [v*, h*], the forward speed (m/s) and the apex height of the body (m).
"""
import numpy as np

DT = 0.01                     # control period (100 Hz)
GRAV = 9.81
NX, NU, NG = 6, 2, 2
NOMINAL = dict(m=10.0, k=4000.0, l0=0.5, b=0.0, gth=1.0, phi_off=0.0, kp=0.0)
#   b: leg damping (N s/m, the nominal has none), gth: thrust gain, phi_off: leg-angle calibration offset (rad),
#   kp: progressive-spring coefficient (k (1 + kp c / l0) with c the compression; 0 = linear)
PKEYS = list(NOMINAL)
D_OFF, D_SCALE, PHI_SCALE = 0.04, 0.04, 0.6

GOAL_LO = np.array([0.0, 0.55])
GOAL_HI = np.array([1.5, 0.75])
TRAIN_GOALS = np.array([[0.0, 0.58], [0.4, 0.70], [0.8, 0.62], [1.1, 0.72], [1.4, 0.65], [0.6, 0.74]])
VAL_SEED, TEST_SEED = 301, 701


def goals_from_seed(n, seed):
    rs = np.random.default_rng(seed)
    return GOAL_LO + (GOAL_HI - GOAL_LO) * rs.uniform(size=(n, 2))


def test_goals(n=8):
    return goals_from_seed(n, TEST_SEED)


OBS_SCALE = np.array([0.1, 0.5, 1.0, 1.0, 0.1, 0.05, 0.5, 0.06])
NOBS = len(OBS_SCALE)
ERR_SCALE = np.array([0.02, 0.10])    # the ILC error rows: predicted apex - h*, vx - v* (in flight)
NE = len(ERR_SCALE)
FAIL_Z = 0.25                         # a fall: the body below this (the leg folded), or a non-finite state


def leg_length(x, xp):
    return xp.sqrt((x[..., 0] - x[..., 4]) ** 2 + x[..., 1] ** 2)


def features(x, g, xp, l0=0.5):
    """Observation: [z - h*, vx - v*, vz, stance, (x - xf) stance, (L - l0) stance, v*, h*] scaled."""
    st = x[..., 5]
    L = leg_length(x, xp)
    o = xp.stack([x[..., 1] - g[..., 1], x[..., 2] - g[..., 0], x[..., 3], st, (x[..., 0] - x[..., 4]) * st,
                  (L - l0) * st, g[..., 0] - 0.75, g[..., 1] - 0.65], -1)
    return o / OBS_SCALE


def errors(x, g, xp):
    """Task error rows (unscaled), in flight only: the ballistic apex z + vz^2 / 2g - h* (constant along a flight) and
    vx - v*. They need no time alignment with any reference."""
    w = 1.0 - x[..., 5]
    apex = x[..., 1] + 0.5 * x[..., 3] ** 2 / GRAV
    return xp.stack([(apex - g[..., 1]) * w, (x[..., 2] - g[..., 0]) * w], -1)


def failed(x, xp):
    return (x[..., 1] < FAIL_Z) | ~xp.all(xp.isfinite(x), -1)


def unnorm(a, xp):
    a = xp.clip(a, -1.0, 1.0)
    return D_OFF + D_SCALE * a[..., 0], PHI_SCALE * a[..., 1]


def sg(v):
    """stop_gradient for jax arrays; identity for numpy."""
    if type(v).__module__.startswith('jax'):
        import jax
        return jax.lax.stop_gradient(v)
    return v


def substep(x, a, p, h, xp):
    """One explicit (semi-implicit Euler) step of length h of the SLIP pogo stick with its touchdown / liftoff
    switching, under the normalized action a and parameters p (a dict; values scalars or per-lane arrays)."""
    delta, phi = unnorm(a, xp)
    phi = phi + p['phi_off']
    X, Z, VX, VZ, XF, ST = (x[..., i] for i in range(6))
    dx, L = X - XF, xp.sqrt((X - XF) ** 2 + Z ** 2)
    Ls = xp.maximum(L, 1e-6)
    Ld = (dx * VX + Z * VZ) / Ls
    comp = xp.maximum(p['l0'] - L, 0.0)
    F = p['k'] * (1.0 + p['kp'] * comp / p['l0']) * (p['l0'] + p['gth'] * delta - L) - p['b'] * Ld
    F = xp.maximum(F, 0.0) * ST                       # the leg only pushes, and only in stance
    ax = F * dx / Ls / p['m']
    az = F * Z / Ls / p['m'] - GRAV
    VX, VZ = VX + h * ax, VZ + h * az
    X, Z = X + h * VX, Z + h * VZ
    # events: touchdown (flight, the foot at the leg's rest length reaches the ground) / liftoff (spring at rest)
    td = (ST < 0.5) & (Z - p['l0'] * xp.cos(phi) <= 0.0) & (VZ < 0)
    Ln = xp.sqrt((X - XF) ** 2 + Z ** 2)
    lo = (ST > 0.5) & (Ln >= p['l0'] + p['gth'] * delta) & (Ld > 0)
    # touchdown: the foot at l0 (sin phi, -cos phi) from the body. The event time depends on phi and the state; its
    # first-order effect (the saltation terms: the body reaches the touchdown height l0 cos phi, the time shift moves
    # x and vz ballistically) enters through straight-through terms dz whose VALUE is 0 (sg = stop_gradient)
    dz = (p['l0'] * xp.cos(phi) - Z) - sg(p['l0'] * xp.cos(phi) - Z)
    vzs = xp.minimum(sg(VZ), -0.05)
    Xt, Zt, VZt = X + VX * dz / vzs, Z + dz, VZ - GRAV * dz / vzs
    X, Z, VZ = xp.where(td, Xt, X), xp.where(td, Zt, Z), xp.where(td, VZt, VZ)
    XF = xp.where(td, X + p['l0'] * xp.sin(phi), XF)
    ST = xp.where(td, 1.0, xp.where(lo, 0.0, ST))
    return xp.stack([X, Z, VX, VZ, XF, ST], -1)


# ---- the classical expert: Raibert's hopping controller (on nominal constants) -----------------------------------
RAIBERT = dict(Ts=0.157, kv=0.08, d0=0.0, kE=0.03)


def expert_action(x, g, xp, gains=None):
    """Raibert: in flight the leg goes to the foot-placement angle asin((vx Ts / 2 + kv (vx - v*)) / l0); in stance
    the thrust is off while compressing and, while extending, delta = d0 + kE (g h* - e), e the vertical energy per
    unit mass including the spring's."""
    r = dict(RAIBERT, **(gains or {}))
    n = NOMINAL
    X, Z, VX, VZ, XF, ST = (x[..., i] for i in range(6))
    L = leg_length(x, xp)
    Ld = ((X - XF) * VX + Z * VZ) / xp.maximum(L, 1e-6)
    xfp = VX * r['Ts'] / 2 + r['kv'] * (VX - g[..., 0])
    phi = xp.arcsin(xp.clip(xfp / n['l0'], -0.55, 0.55))
    e = GRAV * Z + 0.5 * VZ ** 2 + 0.5 * n['k'] / n['m'] * xp.maximum(n['l0'] - L, 0.0) ** 2 * ST
    delta = xp.where((ST > 0.5) & (Ld > 0), xp.clip(r['d0'] + r['kE'] * (GRAV * g[..., 1] - e), 0.0, 0.08), 0.0)
    a = xp.stack([(delta - D_OFF) / D_SCALE, phi / PHI_SCALE], -1)
    return xp.clip(a, -1.0, 1.0)


def mlp(params, o, xp):
    h = o
    for i, (W, b) in enumerate(params):
        h = h @ W + b
        if i < len(params) - 1:
            h = xp.tanh(h)
    return h


def load_policy(path):
    d = np.load(path, allow_pickle=True)
    n = len([k for k in d.files if k.startswith('W')])
    return [(np.asarray(d[f'W{i}']), np.asarray(d[f'b{i}'])) for i in range(n)]


def save_policy(path, params, **extra):
    np.savez(path, **{f'W{i}': np.asarray(W) for i, (W, b) in enumerate(params)},
             **{f'b{i}': np.asarray(b) for i, (W, b) in enumerate(params)}, **extra)


def hop_metrics(xs, g, fail, dt=DT, skip=2):
    """One run: the apexes (vz sign change in flight); over the hops after the first `skip`: RMS apex error, mean
    speed between the first and last used apex. success = no fall, RMS apex error <= 2 cm, |mean vx - v*| <= 0.1."""
    xs = np.asarray(xs)
    vz, st = xs[:, 3], xs[:, 5]
    idx = np.where((vz[:-1] > 0) & (vz[1:] <= 0) & (st[1:] < 0.5))[0] + 1
    out = dict(failed=bool(fail), hops=int(len(idx)))
    use = idx[skip:]
    if len(use) >= 2 and not fail:
        t0, t1 = use[0], use[-1]
        vmean = (xs[t1, 0] - xs[t0, 0]) / ((t1 - t0) * dt)
        ea = xs[use, 1] + 0.5 * xs[use, 3] ** 2 / GRAV - g[1]
        out.update(rms_apex=float(np.sqrt(np.mean(ea ** 2))), mean_apex_err=float(np.mean(ea)), v_err=float(vmean - g[0]))
        out['success'] = bool(out['rms_apex'] <= 0.02 and abs(out['v_err']) <= 0.10)
    else:
        out.update(rms_apex=None, mean_apex_err=None, v_err=None, success=False)
    return out
