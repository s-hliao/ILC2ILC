"""
The "real" cars: the f1tenth_gym fork's multi-body model (s-hliao/f1tenth_gym dev-humble, vehicle_dynamics_mb:
sprung body with roll and pitch, four unsprung wheels with their own spin, suspension, compliant joints, Pacejka MF
tires per corner including the full-scale tire's offsets) calibrated to LLA-MPC's car (car_model.NOMINAL):
  mass car_model.NOMINAL['m'] = 3.74 kg (was 4.6 until 2026-10-30; sprung / unsprung masses, roll / pitch inertias and suspension rates scaled from the gym set by the
  mass ratio, so static deflections stay), yaw inertia 0.04712, steering +-0.34 rad at 3.2 rad/s;
  tire: LLA-MPC's Fiala / brush combined-slip tire on each corner (mb_fiala.py, generated from the fork's
  multi_body.py with only the tire block replaced): per-tire stiffness Cf / 2, Cr / 2, peak mu Fz with each wheel's
  own load from the suspension (so roll and pitch load transfer act on the tires, unlike the single-track model);
  driveline: AWD as LLA-MPC's model lumps it (one omega_w for both axles = the motor), resolved per wheel as on the
  Traxxas 4x4 chassis: a locked centre shaft (reflected motor inertia Im, wheel units) drives each axle's open
  differential through a stiff driveshaft (torque c_d (w_m - axle mean wheel speed)), each diff splitting its torque
  equally left / right; motor torque kt I, the current I a state behind the same current loop as the nominal model
  (slew 300 A/s, -25..50 A). The gym's accel input is
  zero; the wheel torques are added to the MB wheel equations here.
Integrated by RK4 at 0.25 ms in numba, behind the same steering servo; command delays and sensing noise per variant.
"""
import math
import os
import sys

import numba
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lifted_linear_tire_20261009', 'mb_sim'))
from fork_import import (vehicle_dynamics_mb, init_mb, mb_vector, mb_index,           # noqa: E402,F401
                         FULLSCALE_VEHICLE_PARAMETERS as FS)                          # noqa: F401
from mb_params import F1TENTH_MB_PARAMETERS as F1MB                                   # noqa: E402
import car_model as cm                                                                # noqa: E402
from mb_fiala import vehicle_dynamics_mb_fiala                                        # noqa: E402

SUBSTEP = 2.5e-4
NXM = 29
C_SHAFT = 0.5                      # N m s / rad: driveshaft coupling to each axle's diff (~0.7 ms)
LSD = 0.0                          # N m s / rad: viscous limited-slip coupling across each axle diff (stock: open)

N = cm.NOMINAL
_L = N['lf'] + N['lr']
K_PER_LOAD = 0.5 * (N['Cf'] / (N['m'] * cm.G * N['lr'] / _L) + N['Cr'] / (N['m'] * cm.G * N['lf'] / _L))


def mb_params(mass=1.0, mu=1.0, ky=1.0):
    """The MB parameter vector for the LLA car, times the perturbations (mass, friction, slip stiffness)."""
    k = N['m'] / F1MB.m * mass                        # mass ratio to the gym's F1TENTH MB set
    P = F1MB.with_updates(
        m=F1MB.m * k, m_s=F1MB.m_s * k, m_uf=F1MB.m_uf * k, m_ur=F1MB.m_ur * k,
        I_z_body=N['Iz'] * mass, I_Phi_s=F1MB.I_Phi_s * k, I_y_s=F1MB.I_y_s * k,
        I_uf=F1MB.I_uf * k, I_ur=F1MB.I_ur * k,
        K_sf=F1MB.K_sf * k, K_sr=F1MB.K_sr * k, K_sdf=F1MB.K_sdf * k, K_sdr=F1MB.K_sdr * k,
        K_ras=F1MB.K_ras * k, K_rad=F1MB.K_rad * k, K_tsf=F1MB.K_tsf * k, K_tsr=F1MB.K_tsr * k,
        K_zt=F1MB.K_zt * k, K_lt=F1MB.K_lt / k,
        lf=N['lf'], lr=N['lr'], h=N['h'],
        s_min=-N['s_max'], s_max=N['s_max'], sv_min=-N['sv_max'], sv_max=N['sv_max'],
        R_w=N['rw'], I_y_w=N['Iw'] / 4,
        tire_p_dy1=N['muf'] * mu, tire_p_dx1=N['muf'] * mu,
        tire_p_ky1=-K_PER_LOAD * ky, tire_p_kx1=K_PER_LOAD,
        T_se=0.5, T_sb=0.5, a_max=200.0, v_switch=200.0, v_max=50.0, v_min=-5.0)
    return mb_vector(P), P


# variants: MB perturbations, actuation (kt scale, steering offset rad, servo tau s, steering / current delay in
# control periods); sensing noise is common (NOISE)
REAL_CARS = {
    'real_nom': dict(mb={}, kt=1.0, d_off=0.0, tau=0.04, d_delay=0, i_delay=0),
    'real_mass': dict(mb=dict(mass=1.25), kt=1.0, d_off=0.0, tau=0.04, d_delay=0, i_delay=0),
    'real_mu': dict(mb=dict(mu=0.8), kt=1.0, d_off=0.0, tau=0.04, d_delay=0, i_delay=0),
    'real_act': dict(mb={}, kt=0.8, d_off=0.03, tau=0.07, d_delay=0, i_delay=1),
    'real_lag': dict(mb=dict(ky=0.8), kt=1.0, d_off=0.0, tau=0.05, d_delay=1, i_delay=0),
}
# the HARSH deployment suite (2026-10-26, defined before any method was run on it): larger or combined gaps, so that
# zero-shot breaks and real-car data has to carry the result. kyr: rear cornering stiffness scale (front unchanged);
# sense: a default sensing preset; degrade: the tires (stiffness and friction) wear linearly over the run's own laps,
# x1 -> x`to` after `laps` laps (non-stationary; per run -- every trial starts on fresh tires)
HARSH_CARS = {
    'harsh_grip': dict(mb=dict(mu=0.6), kt=1.0, d_off=0.0, tau=0.04, d_delay=0, i_delay=0),
    'harsh_combo': dict(mb=dict(mass=1.25, mu=0.8), kt=1.0, d_off=0.0, tau=0.04, d_delay=1, i_delay=1),
    # calibrated on the plans' LQR only (not a compared method; log/dilc/car_free/calib1.txt): d_off 0.05 / tau 0.10 left
    # LQR at 100 % -> 0.1 rad / 0.15 s / 1 period (LQR 50 %); the rear cornering stiffness is irrelevant at mu 0.2 (the
    # tires are friction-saturated: LQR 100 % even at x0.3) -> harsh_tire dropped
    'harsh_steer': dict(mb={}, kt=1.0, d_off=0.10, tau=0.15, d_delay=1, i_delay=0),
    'harsh_sense': dict(mb={}, kt=1.0, d_off=0.0, tau=0.04, d_delay=0, i_delay=0, sense='mocapbad'),
    'harsh_drift': dict(mb={}, kt=1.0, d_off=0.0, tau=0.04, d_delay=0, i_delay=0, degrade=dict(to=0.75, laps=5.0)),
}
REAL_CARS.update(HARSH_CARS)
# Systematic coverage tiers (2026-10-29, defined before any method was run on them; as the pogo's real_pogo.TIER_A / B):
# A / B: each continuous DR axis of sim_jax.dr_params moved alone from real_nom, both directions, by 0.75 (A, inside the
# DR range) / 1.5 (B, extrapolated) of its DR half-range: mass 25 %, friction 20 %, tire stiffness 20 %, motor constant
# 20 %, steering offset 0.03 rad; the servo time constant's DR range is 0.03-0.08 s (centre 0.055, half-range 0.025) ->
# absolute A 0.036 / 0.074, B 0.0175 / 0.0925. The command delays (DR: 0 / 1 period) are discrete: not an axis here.
# C: 12 effects outside the DR class: the sensing presets (DR randomizes none), tire wear during the run (DR is
# stationary), front / rear cornering-stiffness imbalance (DR scales both axles together).
_NOM = REAL_CARS['real_nom']
TIER_CARS = {}
for _t, _f in (('a', 0.75), ('b', 1.5)):
    for _key, _h in (('mass', 0.25), ('mu', 0.20), ('ky', 0.20)):
        for _s, _tag in ((1, 'hi'), (-1, 'lo')):
            TIER_CARS[f'tier{_t}_{_key}_{_tag}'] = dict(_NOM, mb={_key: 1.0 + _s * _f * _h})
    for _s, _tag in ((1, 'hi'), (-1, 'lo')):
        TIER_CARS[f'tier{_t}_kt_{_tag}'] = dict(_NOM, kt=1.0 + _s * _f * 0.20)
        TIER_CARS[f'tier{_t}_doff_{_tag}'] = dict(_NOM, d_off=_s * _f * 0.03)
        TIER_CARS[f'tier{_t}_tau_{_tag}'] = dict(_NOM, tau=0.055 + _s * _f * 0.025)
for _p in ('mocapbad', 'latency2', 'lowrate', 'ekflag', 'noisy3'):
    TIER_CARS[f'tierc_sense_{_p}'] = dict(_NOM, sense=_p)
for _to in (0.85, 0.75, 0.6):
    TIER_CARS[f'tierc_wear_{int(_to * 100)}'] = dict(_NOM, degrade=dict(to=_to, laps=5.0))
TIER_CARS.update(tierc_rear_lo=dict(_NOM, mb=dict(kyr=0.7)), tierc_rear_hi=dict(_NOM, mb=dict(kyr=1.3)),
                 tierc_front_lo=dict(_NOM, mb=dict(ky=0.7, kyr=1 / 0.7)), tierc_front_hi=dict(_NOM, mb=dict(ky=1.3, kyr=1 / 1.3)))
REAL_CARS.update(TIER_CARS)
TIER_A_CARS = [c for c in TIER_CARS if c.startswith('tiera_')]
TIER_B_CARS = [c for c in TIER_CARS if c.startswith('tierb_')]
TIER_C_CARS = [c for c in TIER_CARS if c.startswith('tierc_')]
NOISE = np.array([0.003, 0.003, 0.005, 0.03, 0.03, 0.03, 0.5, 0.2])   # X Y psi vx vy r omega_w I (sensing)


# sensing / estimation perturbations, applied to the MEASURED state only (real_car.run): name suffix "+<preset>"
#   noise: x NOISE; delay: control periods of latency; hold: the measurement refreshes every hold periods; vel_tau: a
#   first-order lag (s) on vx, vy, r (the EKF's velocity lag)
SENSE = {'mocapbad': dict(noise=3.0, delay=1, hold=2, vel_tau=0.05), 'latency2': dict(delay=2),
         'lowrate': dict(hold=3), 'ekflag': dict(vel_tau=0.1), 'noisy3': dict(noise=3.0)}


def car_config(name):
    """A "real" car: REAL_CARS[name]; or "ax:key=value,..." -- the nominal car with single parameters changed (mass, mu,
    ky: multi-body scales; kt: torque-constant scale; tau: servo time constant; d_off: steering offset; d_delay /
    i_delay: steering / current command delay in periods); either with "+<SENSE preset>" for a sensing perturbation."""
    c, sense = raw_car(name)
    sense = sense or c.get('sense', '')
    mb = c['mb']
    p, P = mb_params(**{k: v for k, v in mb.items() if k != 'kyr'})
    fiala = np.array([N['Cf'] / 2 * mb.get('ky', 1.0), N['Cr'] / 2 * mb.get('ky', 1.0) * mb.get('kyr', 1.0),
                      N['muf'] * mb.get('mu', 1.0), c.get('lsd', LSD)])
    return dict(name=name, p=p, P=P, fiala=fiala, kt=c['kt'] * N['kt'], d_off=c['d_off'], tau=c['tau'],
                d_delay=c['d_delay'], i_delay=c['i_delay'], sense=SENSE[sense] if sense else None,
                degrade=c.get('degrade'))


def raw_car(name):
    """car_config's parameter dict before the multi-body build (REAL_CARS entry, or the "ax:" overrides), and the
    "+<SENSE preset>" suffix."""
    base, _, sense = name.partition('+')
    if base.startswith('ax:'):
        c = dict(REAL_CARS['real_nom'], mb={})
        for kv in base[3:].split(','):
            k, v = kv.split('=')
            if k in ('mass', 'mu', 'ky', 'kyr'):
                c['mb'] = dict(c['mb'], **{k: float(v)})
            elif k in ('d_delay', 'i_delay'):
                c[k] = int(float(v))
            else:
                c[k] = float(v)
    else:
        c = REAL_CARS[base]
    return c, sense


@numba.njit(cache=True)
def _sat(x, lo, hi, k):
    """car_model.sat (numerically stable softplus)."""
    a = k * (x - hi)
    b = k * (lo - x)
    sa = max(a, 0.0) + math.log1p(math.exp(-abs(a)))
    sb = max(b, 0.0) + math.log1p(math.exp(-abs(b)))
    return x - sa / k + sb / k


@numba.njit(cache=True)
def _deriv(x, dc, Ic, p, fi, kt, tau, I_y_w, Im, sv_max, slew, tau_I):
    """x = [MB 29 states, w_m (motor shaft, wheel units), I]."""
    u = np.empty(2)
    u[0] = sv_max * math.tanh((dc - x[2]) / (tau * sv_max))
    u[1] = 0.0
    f = np.zeros(NXM + 2)
    f[:NXM] = vehicle_dynamics_mb_fiala(x[:NXM].copy(), u, p, fi)
    wm = x[NXM]
    I = x[NXM + 1]
    # locked centre shaft to each axle's open differential: the carrier turns at the axle's mean wheel speed and the
    # diff splits its torque equally left / right (wheels free to differ, as on the Traxxas 4x4 chassis)
    T_front = C_SHAFT * (wm - 0.5 * (x[23] + x[24]))
    T_rear = C_SHAFT * (wm - 0.5 * (x[25] + x[26]))
    # limited-slip (viscous) coupling across each axle's diff: torque c_v (w_other - w_this)
    cv = fi[3]
    T_lsd_f = cv * (x[24] - x[23])
    T_lsd_r = cv * (x[26] - x[25])
    f[23] += (0.5 * T_front + T_lsd_f) / I_y_w
    f[24] += (0.5 * T_front - T_lsd_f) / I_y_w
    f[25] += (0.5 * T_rear + T_lsd_r) / I_y_w
    f[26] += (0.5 * T_rear - T_lsd_r) / I_y_w
    f[NXM] = (kt * I - T_front - T_rear) / Im
    f[NXM + 1] = slew * math.tanh((Ic - I) / (tau_I * slew))
    return f


@numba.njit(cache=True)
def mb_step(x, d_cmd, i_cmd, p, fi, kt, tau, dt, I_y_w, Im, sv_max, s_max, slew, tau_I, I_lo, I_hi, k_d, k_i):
    """One control period with the set points held, RK4 at SUBSTEP."""
    n = int(round(dt / SUBSTEP))
    h = dt / n
    dc = _sat(d_cmd, -s_max, s_max, k_d)
    Ic = _sat(i_cmd, I_lo, I_hi, k_i)
    for _ in range(n):
        k1 = _deriv(x, dc, Ic, p, fi, kt, tau, I_y_w, Im, sv_max, slew, tau_I)
        k2 = _deriv(x + 0.5 * h * k1, dc, Ic, p, fi, kt, tau, I_y_w, Im, sv_max, slew, tau_I)
        k3 = _deriv(x + 0.5 * h * k2, dc, Ic, p, fi, kt, tau, I_y_w, Im, sv_max, slew, tau_I)
        k4 = _deriv(x + h * k3, dc, Ic, p, fi, kt, tau, I_y_w, Im, sv_max, slew, tau_I)
        x = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        for i in range(23, 27):
            if x[i] < 0.0:
                x[i] = 0.0
    return x


def step(xm, d_cmd, i_cmd, c, dt=0.025):
    return mb_step(xm, d_cmd, i_cmd, c['p'], c['fiala'], c['kt'], c['tau'], dt, c['P'].I_y_w, N['Im'], N['sv_max'],
                   N['s_max'], N['slew'], N['tau_I'], N['I_min'], N['I_max'], cm.SAT_K[0], cm.SAT_K[1])


def planar(xm):
    """MB state -> the nominal model's [X, Y, psi, vx, vy, r, omega_w, I, delta] (omega_w: the motor shaft, which
    the VESC measures)."""
    return np.array([xm[0], xm[1], xm[4], xm[3], xm[10], xm[5], xm[NXM], xm[NXM + 1], xm[2]])


def mb_from_planar(xs, p):
    """An MB state at the planar state xs (suspension at its static equilibrium from init_mb, all wheels and the
    shaft at xs's wheel speed, the current at xs's)."""
    x0 = np.array([xs[0], xs[1], xs[8], math.hypot(xs[3], xs[4]), xs[2], xs[5], math.atan2(xs[4], xs[3])])
    xm = np.zeros(NXM + 2)
    xm[:NXM] = init_mb(x0, p)
    xm[23:27] = xs[6]
    xm[NXM] = xs[6]
    xm[NXM + 1] = xs[7]
    return xm
# Plastic at mu 0.4 (2026-10-31, user: the plastic tire's friction "is really underestimated at 0.2, should be closer to like 0.4"):
# the 12 plastic validation cars with every friction doubled (the model, the plans and every network stay at mu 0.2, so all arms
# face the same grippier car). p04_<car> for the named cars; the single-axis ones are "ax:mu=1.8" / "ax:mu=1.4" / "ax:mu=1.2".
P04_CARS = {f'p04_{_n}': dict(REAL_CARS[_n], mb=dict(REAL_CARS[_n]['mb'], mu=2.0 * REAL_CARS[_n]['mb'].get('mu', 1.0)))
            for _n in ('real_nom', 'real_mass', 'real_mu', 'real_act', 'real_lag', 'harsh_combo', 'harsh_steer', 'harsh_sense',
                       'harsh_drift')}
REAL_CARS.update(P04_CARS)
PLASTIC04 = list(P04_CARS) + ['ax:mu=1.8', 'ax:mu=1.4', 'ax:mu=1.2']
