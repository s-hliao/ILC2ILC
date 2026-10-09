#!/usr/bin/env python3
"""check_real_vs_sim.py: the numpy twin (sim_jax_np, used on the real cars) against the JAX sim's projection,
observation, ILC error and expert, on the states of a nominal-sim rollout (and perturbed copies of them), for every
plan of the bank. Prints the max absolute differences; they should be at float32 round-off."""
import os
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax                                       # noqa: E402
import jax.numpy as jnp                          # noqa: E402
import numpy as np                               # noqa: E402

import bank                                      # noqa: E402
import sim_jax as sj                             # noqa: E402
import sim_jax_np as sn                          # noqa: E402

names = [f'mocap_{t}_b{b}' for t in ('square2fast', 'figfast') for b in (0, 25)]
plans = bank.load_bank(names)
P = sj.Plans(plans, names)
worst = dict(obs=0.0, err=0.0, expert=0.0, idx=0)
for q, pl in enumerate(plans):
    x0, i0 = sj.start_states(P, jax.random.PRNGKey(q), [q], [0.0], pert=1.0)
    out = sj.rollout(P, None, 200, x0, i0, jnp.array([q]), sj.nominal_params(1))
    X = np.asarray(out['x'][0, :-1])
    I = np.asarray(out['idx'][0])
    pn = sn.PlanNP(pl)
    rng = np.random.default_rng(q)
    for t in range(0, len(X), 5):
        x = X[t] + np.r_[rng.normal(size=6) * [0.02, 0.02, 0.02, 0.1, 0.1, 0.1], 0, 0, 0]
        i_prev = int(I[max(t - 1, 0)])
        i_np = pn.project(i_prev, x[:2])
        i_jx = int(sj.project(P, q, jnp.int32(i_prev), jnp.array(x[:2])))
        worst['idx'] = max(worst['idx'], abs(i_np - i_jx))
        o_np, e_np = pn.features(i_np, x)
        o_jx, e_jx = sj.features(P, q, jnp.int32(i_np), jnp.array(x))
        worst['obs'] = max(worst['obs'], float(np.abs(o_np - np.asarray(o_jx)).max()))
        worst['err'] = max(worst['err'], float(np.abs(e_np - np.asarray(e_jx)).max()))
        u_np = pn.expert(i_np, x)
        u_jx = np.asarray(sj.expert_action(P, q, jnp.int32(i_np), jnp.array(x)))
        worst['expert'] = max(worst['expert'], float(np.abs(u_np - u_jx).max() / np.abs(u_jx).max()))
print('max |numpy - jax|:', worst)
