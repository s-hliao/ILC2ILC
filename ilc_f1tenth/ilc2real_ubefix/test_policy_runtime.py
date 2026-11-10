#!/usr/bin/env python3
"""test_policy_runtime.py: policy_runtime.Controller (the real car's controller, net_runner.py) against real_car.run (the
simulated cars every result comes from): real_car.run trials replayed state by state through the Controller must give the
identical actions, observations and plan indices -- networks, a slow launch, a FADA policy with its oracle planner.
    source env_tight.sh && python test_policy_runtime.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bank                                      # noqa: E402
import policy_runtime as prt                     # noqa: E402
import real_car as rc                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def replay(r, plan, policy, launch=None):
    ctl = prt.Controller(plan, policy, launch=launch, idx0=int(r['idx'][0]))
    worst = dict(mu=0.0, obs=0.0, idx=0)
    for t in range(len(r['x'])):
        out = ctl.step(r['x'][t], v_launch=float(np.hypot(r['x_true'][t][3], r['x_true'][t][4])))
        worst['mu'] = max(worst['mu'], float(np.abs(out['mu'] - r['mu'][t]).max()))
        worst['obs'] = max(worst['obs'], float(np.abs(out['obs'] - r['obs'][t]).max()))
        worst['idx'] = max(worst['idx'], abs(int(out['idx']) - int(r['idx'][t])))
    rec = ctl.record(r['failed'], r['crash'])
    worst['handover'] = rec['handover'] - r['handover']
    return worst


def main():
    fails = 0
    net = prt.load_policy(os.path.join(HERE, 'runs/hw/v3drBtL_s0_lad1_lap10/real_mu_policy.npz'))
    cases = [('net, real_nom, square b25', 'real_nom', 'mocap_square2fast_b25', net, None),
             ('net, real_mu, figfast b25 _mu80', 'real_mu', 'mocap_figfast_b25_mu80', net, None),
             ('net, real_lag, slow launch', 'real_lag', 'mocap_square2fast_b25', net, dict(v0=0.6, v_hand=1.0, current=8.0))]
    fada_p = os.path.join(HERE, 'runs/hw/fada_exact2_ep24/real_nom_policy.npz')
    if os.path.exists(fada_p):
        cases.append(('FADA + oracle planner, real_nom', 'real_nom', 'mocap_figfast_b25', prt.load_policy(fada_p), None))
    for name, car, pname, pol, launch in cases:
        plan = bank.load_plan(pname)
        plan['name'] = pname
        r = rc.run(car, plan, pol, laps=1.5, s0=0.3, pert=np.zeros(5), seed=3, launch=launch)
        w = replay(r, plan, pol, launch)
        ok = w['mu'] < 1e-9 and w['obs'] < 1e-9 and w['idx'] == 0 and w['handover'] == 0
        fails += not ok
        print(f"{name:34s} {len(r['x']):4d} steps  |d mu| {w['mu']:.1e}  |d obs| {w['obs']:.1e}  d idx {w['idx']}  "
              f"d handover {w['handover']}  {'ok' if ok else 'FAIL'}")
    print('all tests passed' if not fails else f'{fails} FAILED')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
