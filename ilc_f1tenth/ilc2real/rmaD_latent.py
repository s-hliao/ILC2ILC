"""rmaD_latent.py (2026-11-07, user: the RMA reference "has to cover all deployment dynamics"): the deployment-covering RMA teacher's
privileged latent and its randomization, over EVERY axis the deployment cars (mb_car: tiers A-C, plastic, rubber) vary, in the deployment
simulator itself (mb_car via real_car.run: the multi-body car, its sensing and its tire wear).

z (14) = ppo_dr.py's 8 (same keys, ZNOM / ZSC: mass, front friction, front cornering stiffness, motor constant, servo tau, steering offset,
steering / current delay) + rear-stiffness split kyr, sensing noise scale, measurement delay, measurement hold, velocity-filter lag, tire wear
(the friction / stiffness factor reached after 5 laps; 1 = none).
sample_car(rng): one random car as an mb_car "ax:" spec, uniform on a box containing all 50 deployment cars (+ margin; checked by
coverage_check()), friction and kyr log-uniform, the sensing block and the wear each nominal with probability 0.5 (most deployment cars
have nominal sensing and no wear) else random; delays 0 / 1 with probability 0.3 each (ppo_dr's).
"""
import numpy as np

import car_model as cm
import mb_car as mb

ZNOM8 = np.array([cm.NOMINAL['m'], cm.NOMINAL['muf'], 250.0, 0.0258, 0.055, 0.0, 0.5, 0.5])   # ppo_dr.py
ZSC8 = np.array([0.6, 0.12, 50.0, 0.005, 0.025, 0.03, 0.5, 0.5])
ZNOM = np.r_[ZNOM8, 1.0, 1.0, 1.0, 2.0, 0.05, 0.8]
ZSC = np.r_[ZSC8, 0.3, 1.0, 1.0, 1.0, 0.05, 0.2]
NZ = len(ZNOM)
KT_NOM, CF_NOM = 0.025743959471583366, 250.0                                                    # sim_jax.nominal_params
BOX = dict(mass=(0.6, 1.4), mu=(0.38, 1.93), ky=(0.68, 1.32), kyr=(0.65, 1.5), kt=(0.68, 1.32), tau=(0.015, 0.155),
           d_off=(-0.105, 0.105), noise=(1.0, 3.2), sdelay=(0, 2), hold=(1, 3), vtau=(0.0, 0.105), wear=(0.55, 1.0))


def raw(name):
    """The car's 14 raw parameters (mb_car.raw_car + its sensing preset / continuous sensing + wear)."""
    c, sense = mb.raw_car(name)
    m = c.get('mb', {})
    sense = sense or c.get('sense', '')
    S = mb.SENSE[sense] if sense else (c.get('sense_d') or {})
    w = (c.get('degrade') or {}).get('to', 1.0)
    return np.array([cm.NOMINAL['m'] * m.get('mass', 1.0), cm.NOMINAL['muf'] * m.get('mu', 1.0), CF_NOM * m.get('ky', 1.0),
                     KT_NOM * c['kt'], c['tau'], c['d_off'], float(c['d_delay']), float(c['i_delay']), m.get('kyr', 1.0),
                     S.get('noise', 1.0), float(S.get('delay', 0)), float(S.get('hold', 1)), S.get('vel_tau', 0.0), w])


def z_of(name):
    return ((raw(name) - ZNOM) / ZSC).astype(np.float32)


def sample_car(rng):
    u = lambda k: rng.uniform(*BOX[k])
    lu = lambda k: float(np.exp(rng.uniform(np.log(BOX[k][0]), np.log(BOX[k][1]))))
    kv = dict(mass=u('mass'), mu=lu('mu'), ky=u('ky'), kyr=(lu('kyr') if rng.uniform() < 0.5 else 1.0), kt=u('kt'), tau=u('tau'),
              d_off=u('d_off'), d_delay=int(rng.uniform() < 0.3), i_delay=int(rng.uniform() < 0.3))
    if rng.uniform() < 0.5:
        kv.update(noise=u('noise'), sdelay=int(rng.integers(0, 3)), hold=int(rng.integers(1, 4)), vtau=u('vtau'))
    if rng.uniform() < 0.5:
        kv.update(wear=u('wear'))
    return 'ax:' + ','.join(f'{k}={v:.6g}' for k, v in kv.items())


def deployment_cars():
    tier = sorted(k for k in mb.REAL_CARS if k[:5] in ('tiera', 'tierb', 'tierc'))
    return tier + list(mb.PLASTIC_N04) + list(mb.RUBBER_N06)


def coverage_check(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    Zs = np.array([z_of(sample_car(rng)) for _ in range(n)])
    lo, hi = Zs.min(0), Zs.max(0)
    bad = [(c, [j for j in range(NZ) if not lo[j] - 1e-5 <= z[j] <= hi[j] + 1e-5]) for c in deployment_cars() for z in [z_of(c)]]
    return lo, hi, [b for b in bad if b[1]]
