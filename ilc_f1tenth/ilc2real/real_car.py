"""
Flying the policy (or the plan's LQR, or a per-track ILC feedforward) on a "real" car (mb_car.py): numpy ports of
sim_jax's projection / observation / error / expert (checked equal to the JAX ones by check_real_vs_sim), sensing
noise on the measured planar state, the car's command delays and steering offset, and the MB plant.

run(car, plan, policy, laps, x0=None, s0=0, ...) -> dict with the measured planar states, true planar states,
observations, actions (normalized, mean and applied), errors, indices, progress, failure. Runs in a worker pool via
run_many (one process per run, numba compiled once per worker).
"""
import math
import os

import numpy as np

import car_model as cm
import mb_car as mc
import sim_jax_np as sn

DT = 0.025


def mlp_np(ps, o):
    h = o
    for W, b in ps[:-1]:
        h = np.tanh(h @ W + b)
    W, b = ps[-1]
    return h @ W + b


def run(car_name, plan, policy=None, laps=1.0, s0=0.0, pert=None, seed=0, noise=True, T_max=None, ff=None,
        x_start=None, idx_start=None, action_noise=None, launch=None):
    """policy: list of (W, b) numpy (the network) or None (the plan's LQR). ff: (M, 2) normalized feedforward
    offsets on the plan's s-grid added to the action (per-track ILC baseline). x_start: a planar start state
    (continuing a run); else the plan's state at s0 plus pert (5 numbers, as sim_jax.start_states).
    launch: a SLOW ROLLING start -- dict(v0, v_hand, current): the car starts at v0 (0.6 m/s: the multi-body car is
    kinematic below 0.5 m/s, without the driveline, so it cannot launch from standstill) with zero current and steering;
    a launch controller (the plan's steering plus path feedback, a fixed current, as LLA-MPC's low-speed mode) drives
    until |V| >= v_hand (v_hand <= v0: the policy from the start), then the policy (the stall check is off until then). The car's sensing perturbation (car_config 'sense') applies
    to the measured state."""
    rng = np.random.default_rng(seed)
    # the room's safety envelope (F1T_SAFETY_EY, m): the run is stopped -- a crash -- as soon as the TRUE lateral
    # offset from the path exceeds it (the walls / the operator's kill switch); unset: the old departure limit
    max_ey = float(os.environ.get('F1T_SAFETY_EY') or sn.FAIL_EY)
    c = mc.car_config(car_name)
    N0 = cm.NOMINAL
    pl = sn.PlanNP(plan)
    if x_start is None:
        i0 = int(np.round(np.mod(s0, pl.L) / pl.ds)) % pl.M
        xs0 = plan['x'][i0].copy()
        if pert is not None:
            r = np.asarray(pert)
            xs0[0] += -math.sin(xs0[2]) * 0.1 * r[0]
            xs0[1] += math.cos(xs0[2]) * 0.1 * r[0]
            xs0[2] += 0.1 * r[1]
            xs0[3] += 0.3 * r[2]
            xs0[4] += 0.3 * r[3]
            xs0[5] += 0.5 * r[4]
            xs0[6] = max(xs0[6] + 0.3 * r[2] / N0['rw'], 0.0)
        idx = i0
    else:
        xs0 = np.asarray(x_start, float)
        idx = int(idx_start)
    if launch:                                        # a slow rolling start: the multi-body car is kinematic below
        xs0 = np.asarray(xs0, float).copy()           # 0.5 m/s (mb_fiala KIN_THRESH: no driveline force), so a true
        xs0[3:9] = 0.0                                # standstill cannot launch; v0 0.6 m/s, wheels rolling without
        xs0[3] = launch.get('v0', 0.6)                # slip, zero current and steering
        xs0[6] = xs0[3] / N0['rw']
    xm = mc.mb_from_planar(xs0, c['p'])
    T = T_max or int(math.ceil(laps * plan['lap_time'] / DT))
    cmd_hist = [np.zeros(2)] * 4                     # commands, newest last (the steering / current delays)
    S = c.get('sense') or {}
    meas_hist, m_last, vel_f = [], None, None
    launched, handover = not launch, (0 if not launch else -1)
    X, Xm, OBS, MU, A, ERR, ERRT, IDX, APP, OBST = [], [], [], [], [], [], [], [], [], []
    pname = plan.get('name', '')
    prog, failed, crash, peak_ey = 0.0, False, '', 0.0
    hist, last_a = [], np.zeros(2)
    for t in range(T):
        xt = mc.planar(xm)
        meas_hist.append(xt)                          # sensing: latency, refresh rate, noise, velocity lag
        if m_last is None or t % int(S.get('hold', 1)) == 0:
            m_last = meas_hist[max(0, len(meas_hist) - 1 - int(S.get('delay', 0)))].copy()
            if noise:
                m_last[:8] += rng.normal(size=8) * mc.NOISE * S.get('noise', 1.0)
        xmeas = m_last.copy()
        if S.get('vel_tau', 0) > 0:
            vel_f = xmeas[3:6].copy() if vel_f is None else vel_f + DT / (S['vel_tau'] + DT) * (xmeas[3:6] - vel_f)
            xmeas[3:6] = vel_f
        if len(meas_hist) > 8:
            meas_hist.pop(0)
        i_new = pl.project(idx, xmeas[:2])
        dprog = ((i_new - idx) + pl.M // 2) % pl.M - pl.M // 2
        prog += dprog * pl.ds
        idx = i_new
        obs, err = pl.features(idx, xmeas)
        obs_true, err_true = pl.features(idx, xt)
        if policy is None:
            mu = (pl.expert(idx, xmeas) - cm.A_OFF) / cm.A_SCALE
        elif isinstance(policy, dict) and 'fada' in policy:    # FADA: IDM(o, lam d) with d the plan deviation
            orc = policy.get('oracle') or {}
            dims = policy.get('dims')
            d_next = orc[pname][(idx + 1) % len(orc[pname])] if pname in orc else \
                policy['lam'] * (obs[:8] if dims is None else obs[np.asarray(dims)])
            mu = mlp_np(policy['fada'], np.r_[obs, d_next])          # oracle planner: the deviation a policy
            #                                                          that works on this car reaches there
        elif isinstance(policy, dict):                 # RMA: pi(o, phi(history of obs deviations, actions))
            hist.append(np.r_[obs[:8], last_a])
            Hn = policy['hist']
            hv = np.concatenate(([np.zeros(10)] * max(0, Hn - len(hist))) + hist[-Hn:])
            zhat = mlp_np(policy['adapt'], hv)
            mu = mlp_np(policy['actor'], np.r_[obs, zhat])
        else:
            mu = mlp_np(policy, obs)
        if ff is not None:
            mu = mu + pl.ref_at(idx, pl.path_coords(idx, xmeas)[0], ff)
        if not launched:                              # the launch controller until v_hand
            if math.hypot(xt[3], xt[4]) >= launch['v_hand']:
                launched, handover = True, t
            else:
                s_off, e_y, e_psi = pl.path_coords(idx, xmeas)
                d_ref = pl.ref_at(idx, s_off, pl.u)[0]
                d_l = float(np.clip(d_ref - 1.0 * e_y - 0.8 * e_psi, -N0['s_max'], N0['s_max']))
                mu = (np.array([d_l, launch['current']]) - cm.A_OFF) / cm.A_SCALE
        a = mu + (action_noise[t] if action_noise is not None else 0.0)
        last_a = a
        ap = a * cm.A_SCALE + cm.A_OFF
        cmd_hist.append(ap)
        if len(cmd_hist) > 8:
            cmd_hist.pop(0)
        d_cmd = cmd_hist[-1 - int(c['d_delay'])][0] + c['d_off']
        i_cmd = cmd_hist[-1 - int(c['i_delay'])][1]
        X.append(xmeas); Xm.append(xt); OBS.append(obs); OBST.append(obs_true); MU.append(mu); A.append(a); ERR.append(err); ERRT.append(err_true)
        IDX.append(idx); APP.append([d_cmd, i_cmd])
        peak_ey = max(peak_ey, abs(float(err_true[0])))
        if abs(err_true[0]) > max_ey or abs(err_true[1]) > sn.FAIL_EPSI or (launched and xt[3] < 0.3):
            failed = True
            crash = 'wall' if abs(err_true[0]) > max_ey else ('spin' if abs(err_true[1]) > sn.FAIL_EPSI else 'stall')
            break
        try:
            xm = mc.step(xm, d_cmd, i_cmd, c, DT)
        except (ZeroDivisionError, FloatingPointError):
            failed, crash = True, 'numerics'
            break
        if not np.all(np.isfinite(xm)):
            failed, crash = True, 'numerics'
            break
    xt = mc.planar(xm)
    return dict(x=np.array(X), x_true=np.array(Xm), x_end=xt, idx_end=idx, obs=np.array(OBS), obs_true=np.array(OBST),
                mu=np.array(MU),
                a=np.array(A), err=np.array(ERR), err_true=np.array(ERRT), idx=np.array(IDX), applied=np.array(APP), progress=prog,
                laps=prog / pl.L, failed=failed, crash=crash, peak_ey=peak_ey, max_ey=max_ey, steps=len(X), car=car_name,
                handover=handover,
                plan=plan.get('name', ''))


def _worker(args):
    return run(**args)


_POOL = None


def run_many(jobs, procs=None):
    """Runs in a persistent pool of SPAWNED workers (the parent has JAX threads; forking it can deadlock)."""
    global _POOL
    import multiprocessing as mp
    procs = procs or int(os.environ.get('F1T_PROCS', '16'))
    if procs <= 1:
        return [run(**j) for j in jobs]
    if _POOL is None:
        _POOL = mp.get_context('spawn').Pool(procs)
    return _POOL.map(_worker, jobs, chunksize=1)


def lap_metrics(r, plan, ey_ok=0.10, pace_ok=0.9):
    """Metrics of a run against the plan at the car's own arc length: completed laps, RMS lateral error (m), RMS
    sideslip error (deg), mean |beta| (deg), RMS speed error, failure, and per-lap RMS lateral error (multi-lap
    stability). success: no failure (spin-out / departure), rms e_y <= ey_ok and the pace (mean speed over the plan's
    at the same arc lengths) >= pace_ok -- grip or drift alike (the sideslip is descriptive: mean |beta|)."""
    e = r['err_true']
    if len(e) < 2:
        return dict(laps=0.0, failed=True, success=False)
    x = r['x_true']
    z = np.asarray(plan['z'])[r['idx']]
    beta = np.degrees(np.arctan2(x[:, 4], x[:, 3]))
    beta_ref = np.degrees(np.arctan2(z[:, 3], z[:, 2]))
    v, v_ref = np.hypot(x[:, 3], x[:, 4]), np.hypot(z[:, 2], z[:, 3])
    rms = lambda q: float(np.sqrt(np.mean(np.square(q))))
    steps_per_lap = max(1, int(round(float(plan['lap_time']) / DT)))
    per_lap = [rms(e[k:k + steps_per_lap, 0]) for k in range(0, len(e) - steps_per_lap // 2, steps_per_lap)]
    m = dict(laps=float(r['laps']), failed=bool(r['failed']), rms_ey=rms(e[:, 0]), rms_dbeta=rms(beta - beta_ref),
             mean_abs_beta=float(np.mean(np.abs(beta))), mean_abs_beta_ref=float(np.mean(np.abs(beta_ref))),
             rms_dv=rms(v - v_ref), pace=float(np.mean(v) / np.mean(v_ref)), per_lap_ey=per_lap,
             crash=r.get('crash', ''), peak_ey=float(r.get('peak_ey', np.max(np.abs(e[:, 0])))))
    m['success'] = bool(not m['failed'] and m['rms_ey'] <= ey_ok and m['pace'] >= pace_ok)
    return m
