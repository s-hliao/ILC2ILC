#!/usr/bin/env python3
"""check_mb_jax.py: the JAX multi-body car (mb_car_jax) against the numba one (mb_car) -- derivatives at random
drifting states and one control period from a plan state; prints max relative differences."""
import os
import sys
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax                                       # noqa: E402
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp                          # noqa: E402
import numpy as np                               # noqa: E402

import mb_car as mc                              # noqa: E402
import mb_car_jax as mj                          # noqa: E402

c = mc.car_config('real_nom')
cj = mj.car_arrays('real_nom')
rng = np.random.default_rng(0)
worst_d = 0.0
for k in range(50):
    V, b = rng.uniform(1.5, 3.5), rng.uniform(-0.5, 0.5)
    xs = np.array([0, 0, rng.uniform(-3, 3), V * np.cos(b), V * np.sin(b), rng.uniform(-3, 3),
                   V / 0.051 * rng.uniform(0.9, 1.4), rng.uniform(-10, 45), rng.uniform(-0.3, 0.3)])
    xm = mc.mb_from_planar(xs, c['p'])
    xm[23:27] *= rng.uniform(0.9, 1.1, 4)
    xm[[6, 7, 13, 18]] += rng.normal(size=4) * 0.02
    dc, Ic = rng.uniform(-0.3, 0.3), rng.uniform(-20, 45)
    fn = mc._deriv(xm, dc, Ic, c['p'], c['fiala'], c['kt'], c['tau'], c['P'].I_y_w, mc.N['Im'], mc.N['sv_max'],
                   mc.N['slew'], mc.N['tau_I'])
    fj = np.asarray(mj.deriv(jnp.array(xm), dc, Ic, cj))
    worst_d = max(worst_d, float(np.max(np.abs(fn - fj) / (1 + np.abs(fn)))))
xm1 = mc.step(xm.copy(), 0.1, 20.0, c)
xj1 = np.asarray(jax.jit(lambda x: mj.step(x, jnp.array([0.1, 20.0]), cj))(jnp.array(xm)))
print('derivative max rel diff', worst_d, '| one 25 ms step max abs diff', float(np.max(np.abs(xm1 - xj1))))
