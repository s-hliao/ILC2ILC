"""The drift-parking task in the nominal car, batched in JAX.

Dynamics: car_model.dynamics (the nominal Fiala single-track, vendored), RK4 at 0.25 ms inside each 25 ms control
period -- exactly ilc2real/sim_jax.py's integration, so the two packages fly the same car.

An episode (one park): the car is RELEASED from a steady drift on a circle (radius R0, sideslip beta0; the operator's
"go" while it circles) and must stop in a slot: position and heading, at rest. Slots are sampled relative to the release
state. Left-turning circles only; right-turning ones are the mirror image (`mirror_*`), used for augmentation.

Observation (11, normalized): the car's position relative to the slot in the SLOT frame / 1 m, (cos, sin) of the heading
error, body velocities vx / 3, vy / 3, r / 3, wheel speed rw * omega_w / 3, current I / 25, steering delta / 0.34, the time
since release / T_MAX. Action (2, in [-1, 1]): steering set point * s_max, current set point A_OFF + a * I_HALF.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ilc_park import car_model as cm

O = cm.jax_ops()
DT = 0.025
SUB = 100
T_MAX = 4.0                                       # s from release to the stop (the plans take 1-2 s)
NOBS, NACT = 11, 2
PKEYS = ('m', 'Iz', 'muf', 'mur', 'Cf', 'Cr', 'kt', 'tau_s')


def nominal_params(n, **over):
    """Per-lane model parameters (the nominal car, optionally overridden: a 'real' car for checks)."""
    d = {k: jnp.full((n,), float(v)) for k, v in cm.NOMINAL.items()}
    for k, v in over.items():
        d[k] = jnp.broadcast_to(jnp.asarray(v, float), (n,))
    return d


def _rk4_period(x, a, p):
    h = DT / SUB

    def body(_, x):
        k1 = cm.dynamics(x, a, p, O)
        k2 = cm.dynamics(x + h / 2 * k1, a, p, O)
        k3 = cm.dynamics(x + h / 2 * k2, a, p, O)
        k4 = cm.dynamics(x + h * k3, a, p, O)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x.at[6].set(jnp.maximum(x[6], 0.0))
    return jax.lax.fori_loop(0, SUB, body, x)


def action_to_setpoints(a):
    """Normalized action in [-1, 1]^2 -> [delta_cmd, I_cmd] (physical)."""
    a = jnp.clip(a, -1.0, 1.0)
    return jnp.stack([a[..., 0] * cm.NOMINAL['s_max'], cm.A_OFF[1] + a[..., 1] * cm.A_SCALE[1]], -1)


def step(x, a, p):
    """One 25 ms control period for one lane (normalized action)."""
    return _rk4_period(x, action_to_setpoints(a), p)


def slot_frame(x, slot):
    """(position of the car in the slot frame (2,), heading error) for one lane; slot = [X, Y, psi]."""
    c, s = jnp.cos(slot[2]), jnp.sin(slot[2])
    d = x[:2] - slot[:2]
    px = c * d[0] + s * d[1]
    py = -s * d[0] + c * d[1]
    dpsi = jnp.arctan2(jnp.sin(x[2] - slot[2]), jnp.cos(x[2] - slot[2]))
    return jnp.stack([px, py]), dpsi


def observe(x, slot, t):
    """The policy's observation for one lane (see the module docstring)."""
    pos, dpsi = slot_frame(x, slot)
    return jnp.concatenate([pos, jnp.stack([jnp.cos(dpsi), jnp.sin(dpsi), x[3] / 3.0, x[4] / 3.0, x[5] / 3.0,
                                            cm.NOMINAL['rw'] * x[6] / 3.0, x[7] / 25.0, x[8] / cm.NOMINAL['s_max'],
                                            t / T_MAX])])


def observe_batch(xs, slot):
    """Observations (T, 11) of one lane's states (T, 9), the t-th at time t * DT."""
    return jax.vmap(observe, (0, None, 0))(jnp.asarray(xs), jnp.asarray(slot), jnp.arange(len(xs)) * DT)


def speed(x):
    return jnp.hypot(x[..., 3], x[..., 4])


def sideslip(x):
    return jnp.arctan2(x[..., 4], jnp.sqrt(x[..., 3] ** 2 + 1e-3))


# ---------------------------------------------------------------------------------------------------------------
# Release states: steady drifts of the nominal car (solved once on the CPU, CasADi), tabulated over (R0, beta0)
# ---------------------------------------------------------------------------------------------------------------

def drift_table(R_values, beta_deg_values, p=None):
    """(n, 9) release states at the origin heading the circle's tangent (+x), turning left, and their (R0, beta0)."""
    from ilc_park.trajopt import start_state
    p = dict(cm.NOMINAL) if p is None else p
    states, meta = [], []
    for R in R_values:
        for b in beta_deg_values:
            x0, eq = start_state(R, b, p)
            if np.all(np.isfinite(x0)) and 0.5 < eq[0] < 5.0:
                states.append(x0)
                meta.append((R, b))
    return np.asarray(states), np.asarray(meta)


def place(x0, origin):
    """Move a release state (at the origin, heading +x velocity) to a room pose origin = [X, Y, theta]."""
    c, s = np.cos(origin[2]), np.sin(origin[2])
    x = np.array(x0, float).copy()
    x[0], x[1] = origin[0] + c * x0[0] - s * x0[1], origin[1] + s * x0[0] + c * x0[1]
    x[2] = x0[2] + origin[2]
    return x


# ---------------------------------------------------------------------------------------------------------------
# Mirror symmetry (left <-> right): reflect y. Used to train / refit on both turn directions.
# ---------------------------------------------------------------------------------------------------------------

MIRROR_STATE = np.array([1, -1, -1, 1, -1, -1, 1, 1, -1], float)    # X, Y, psi, vx, vy, r, omega_w, I, delta
MIRROR_OBS = np.array([1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1], float)  # px, py, cos, sin, vx, vy, r, w, I, delta, t
MIRROR_ACT = np.array([-1, 1], float)


def mirror_slot(slot):
    return slot * np.array([1, -1, -1])


@partial(jax.jit, static_argnums=(3,))
def rollout(policy_params, x0, slot, policy_fn, p, noise):
    """Fly a batch of lanes for T_MAX: returns states (T+1, n, 9), observations (T, n, 11), actions (T, n, 2).

    ``noise`` (T, n, 2): added to the policy mean (exploration / perturbation); zeros for a deterministic flight.
    The flight continues after the stop (the policy is asked to stay put); callers cut at the stop.
    """
    T = noise.shape[0]

    def one(x, inp):
        k, e = inp
        o = jax.vmap(observe, (0, 0, None))(x, slot, k * DT)
        a = policy_fn(policy_params, o) + e
        x1 = jax.vmap(step)(x, a, p)
        return x1, (x1, o, a)
    _, (xs, os_, as_) = jax.lax.scan(one, x0, (jnp.arange(T), noise))
    return jnp.concatenate([x0[None], xs]), os_, as_
