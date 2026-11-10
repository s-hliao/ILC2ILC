#!/usr/bin/env python3
"""rma_oracle_car.py OUTDIR RMA_DIR CAR [CAR ...] (2026-11-07, paper/FIGURE_PLAN_1106.md "RMA with the ORACLE latent"): writes
OUTDIR/its/it0_<car>_policy.npz = the RMA teacher's actor (RMA_DIR/policy_final.npz; the phase-1 actor, frozen in phase 2) with mode
'rma_oracle' and z = the car's TRUE latent: its parameters (mb_car.raw_car) mapped onto ppo_dr.py's DR keys and normalized by its ZNOM /
ZSC: m = m_nom x mass, muf = muf_nom x mu, Cf = Cf_nom x ky, kt = kt_nom x kt, tau_s = tau, d_off, d_delay, i_delay. Sensing presets,
tire wear (degrade) and the rear-axle split (kyr) are not DR keys: invisible to z (declared)."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mb_car as mb                              # noqa: E402
import car_model as cm                           # noqa: E402

ZNOM = np.array([cm.NOMINAL['m'], cm.NOMINAL['muf'], 250.0, 0.0258, 0.055, 0.0, 0.5, 0.5])   # ppo_dr.py
ZSC = np.array([0.6, 0.12, 50.0, 0.005, 0.025, 0.03, 0.5, 0.5])
KT_NOM, CF_NOM = 0.025743959471583366, 250.0                                                  # sim_jax.nominal_params


def car_z(name):
    c, _ = mb.raw_car(name)
    m = c.get('mb', {})
    raw = np.array([cm.NOMINAL['m'] * m.get('mass', 1.0), cm.NOMINAL['muf'] * m.get('mu', 1.0), CF_NOM * m.get('ky', 1.0),
                    KT_NOM * c['kt'], c['tau'], c['d_off'], float(c['d_delay']), float(c['i_delay'])])
    return (raw - ZNOM) / ZSC, raw


if __name__ == '__main__':
    out, rd, cars = sys.argv[1], sys.argv[2], sys.argv[3:]
    d = np.load(os.path.join(rd, 'policy_final.npz'))
    W = {k: d[k] for k in d.files if k[0] in 'Wb' and k[1:].isdigit()}
    assert W['W0'].shape[0] == 55 + 8, W['W0'].shape
    os.makedirs(os.path.join(out, 'its'), exist_ok=True)
    for c in cars:
        z, raw = car_z(c)
        np.savez(os.path.join(out, 'its', f'it0_{c}_policy.npz'), **W, mode='rma_oracle', z=z, z_raw=raw)
    print(out, len(cars), 'cars')
