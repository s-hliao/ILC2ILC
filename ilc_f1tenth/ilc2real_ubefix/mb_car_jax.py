"""JAX twin of mb_car (the multi-body car: mb_jax dynamics + steering servo + current loop + locked centre shaft to
open axle diffs with a viscous limited-slip coupling): the NOMINAL sim of the sim stage, batched on the GPU with
autodiff Jacobians. State: [MB 29, w_m (motor shaft, wheel units), I] (31). check_mb_jax.py compares it with the
numba version."""
import jax
import jax.numpy as jnp
import numpy as np

import car_model as cm
import mb_car as mc
from mb_jax import vehicle_dynamics_mb_jax

NXM = mc.NXM
NXF = NXM + 2
N = cm.NOMINAL


def _sat(x, lo, hi, k):
    return x - jnp.logaddexp(0.0, k * (x - hi)) / k + jnp.logaddexp(0.0, k * (lo - x)) / k


def deriv(x, dc, Ic, car):
    """car: dict of jnp arrays p (MB vector), fiala (4), kt, tau (from mb_car.car_config)."""
    sv = N['sv_max'] * jnp.tanh((dc - x[2]) / (car['tau'] * N['sv_max']))
    f = vehicle_dynamics_mb_jax(x[:NXM], jnp.stack([sv, 0.0]), car['p'], car['fiala'])
    wm, I = x[NXM], x[NXM + 1]
    I_y_w = car['I_y_w']
    T_front = mc.C_SHAFT * (wm - 0.5 * (x[23] + x[24]))
    T_rear = mc.C_SHAFT * (wm - 0.5 * (x[25] + x[26]))
    cv = car['fiala'][3]
    T_lf = cv * (x[24] - x[23])
    T_lr = cv * (x[26] - x[25])
    add = jnp.zeros(NXM).at[23].set((0.5 * T_front + T_lf) / I_y_w).at[24].set((0.5 * T_front - T_lf) / I_y_w) \
        .at[25].set((0.5 * T_rear + T_lr) / I_y_w).at[26].set((0.5 * T_rear - T_lr) / I_y_w)
    dwm = (car['kt'] * I - T_front - T_rear) / N['Im']
    dI = N['slew'] * jnp.tanh((Ic - I) / (N['tau_I'] * N['slew']))
    return jnp.concatenate([f + add, jnp.stack([dwm, dI])])


def step(x, a_phys, car, dt=0.025, n=100):
    """One control period with the set points a_phys = [delta_cmd, I_cmd] held; RK4 at dt / n (mb_car.SUBSTEP)."""
    dc = _sat(a_phys[0], -N['s_max'], N['s_max'], cm.SAT_K[0])
    Ic = _sat(a_phys[1], N['I_min'], N['I_max'], cm.SAT_K[1])
    h = dt / n

    def body(_, x):
        k1 = deriv(x, dc, Ic, car)
        k2 = deriv(x + 0.5 * h * k1, dc, Ic, car)
        k3 = deriv(x + 0.5 * h * k2, dc, Ic, car)
        k4 = deriv(x + h * k3, dc, Ic, car)
        x = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x.at[23:27].set(jnp.maximum(x[23:27], 0.0))
    return jax.lax.fori_loop(0, n, body, x)


def car_arrays(name='real_nom'):
    c = mc.car_config(name)
    return dict(p=jnp.array(c['p']), fiala=jnp.array(c['fiala']), kt=jnp.array(c['kt']), tau=jnp.array(c['tau']),
                I_y_w=jnp.array(c['P'].I_y_w))


def planar(xm):
    """[X, Y, psi, vx, vy, r, omega_w (shaft), I, delta] (mb_car.planar)."""
    return jnp.stack([xm[0], xm[1], xm[4], xm[3], xm[10], xm[5], xm[NXM], xm[NXM + 1], xm[2]])
