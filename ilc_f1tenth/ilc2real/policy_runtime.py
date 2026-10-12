"""
policy_runtime.py: the controller side of one control period, exactly as real_car.run flies a network on the simulated cars,
for the REAL car (ilc_f1tenth_hw/scripts/net_runner.py) -- the plan projection, the observation / error features, the
policy (MLP network, FADA planner-IDM, RMA with its adaptation module), the slow-start launch controller, and the trial
record in real_car.run's format (so hw_stage.py's hardware stage takes real trials unchanged). test_policy_runtime.py
replays real_car.run trials through it and checks every action, observation and index is identical.

    ctl = Controller(plan, policy, launch=dict(v_hand=1.0, current=8.0))
    for each period: out = ctl.step(xmeas)  ->  out['cmd'] = [steering rad, motor current A]
    rec = ctl.record(failed, crash)          ->  the real_car.run dict (true = measured: the car has no ground truth)

xmeas: the measured planar state [X, Y, psi, vx, vy, r, omega_w, I, delta] (psi unwrapped; omega_w the wheel speed rad/s;
I the motor current A; delta the steering angle, the last command on the real car).
"""
import math

import numpy as np

import car_model as cm
import sim_jax_np as sn

DT = 0.025
N0 = cm.NOMINAL


def mlp_np(ps, o):
    h = o
    for W, b in ps[:-1]:
        h = np.tanh(h @ W + b)
    W, b = ps[-1]
    return h @ W + b


def load_policy(path):
    """A policy file as hw_stage / reeval save them: an MLP (W0, b0, ...), FADA (mode fada) or RMA (+ rma_adapt.npz)."""
    import os
    d = np.load(path)
    n = len([k for k in d.files if k.startswith('W')])
    pol = [(np.asarray(d[f'W{i}']), np.asarray(d[f'b{i}'])) for i in range(n)]
    if 'mode' in d.files and str(d['mode']) == 'fada' or 'lam' in d.files:
        orc = {k[len('oracle__'):]: np.asarray(d[k]) for k in d.files if k.startswith('oracle__')}
        return dict(fada=pol, lam=float(d['lam']), oracle=orc or None,
                    dims=np.asarray(d['dims']) if 'dims' in d.files else None,
                    horizon=int(d['horizon']) if 'horizon' in d.files else 1)
    ad = os.path.join(os.path.dirname(path), 'rma_adapt.npz')
    if 'mode' in d.files and str(d['mode']) == 'rma' and os.path.exists(ad):
        e = np.load(ad)
        m = len([k for k in e.files if k.startswith('W')])
        return dict(actor=pol, adapt=[(e[f'W{i}'], e[f'b{i}']) for i in range(m)], hist=int(e['hist']))
    return pol


class Controller:
    def __init__(self, plan, policy, launch=None, idx0=None, x0=None):
        """plan: a bank plan dict; policy: load_policy's result; launch: dict(v_hand, current) for a start below v_hand
        (the plan's steering + path feedback at a fixed current until |V| >= v_hand, as real_car.run's launch); idx0: the
        start index on the plan, or x0 (a measured state) for the nearest plan point."""
        self.plan, self.policy, self.launch = plan, policy, launch
        self.pl = sn.PlanNP(plan)
        self.pname = plan.get('name', '')
        if idx0 is None:
            idx0 = int(np.argmin(np.sum((self.pl.xy - np.asarray(x0)[:2]) ** 2, -1))) if x0 is not None else 0
        self.idx = int(idx0)
        self.prog = 0.0
        self.launched = not launch
        self.handover = 0 if not launch else -1
        self.hist, self.last_a = [], np.zeros(2)
        self.t = 0
        self.rec = {k: [] for k in ('x', 'obs', 'mu', 'a', 'err', 'idx', 'applied')}
        self.peak_ey = 0.0

    def step(self, xmeas, v_launch=None, action_noise=None):
        """One control period from the measured state. v_launch: the speed the launch hand-over checks (real_car.run uses
        the true state's; on the car, the measured). Returns dict(mu, a, cmd, obs, err, idx, progress, launched)."""
        pl, policy = self.pl, self.policy
        xmeas = np.asarray(xmeas, float)
        i_new = pl.project(self.idx, xmeas[:2])
        dprog = ((i_new - self.idx) + pl.M // 2) % pl.M - pl.M // 2
        self.prog += dprog * pl.ds
        self.idx = i_new
        idx = self.idx
        obs, err = pl.features(idx, xmeas)
        if policy is None:
            mu = (pl.expert(idx, xmeas) - cm.A_OFF) / cm.A_SCALE
        elif isinstance(policy, dict) and 'fada' in policy:
            orc = policy.get('oracle') or {}
            dims = policy.get('dims')
            ahead = max(1, int(round(int(policy.get('horizon', 1)) * math.hypot(xmeas[3], xmeas[4]) * DT / pl.ds)))
            d_next = orc[self.pname][(idx + ahead) % len(orc[self.pname])] if self.pname in orc else \
                policy['lam'] * (obs[:8] if dims is None else obs[np.asarray(dims)])
            mu = mlp_np(policy['fada'], np.r_[obs, d_next])
        elif isinstance(policy, dict):
            self.hist.append(np.r_[obs[:8], self.last_a])
            Hn = policy['hist']
            hv = np.concatenate(([np.zeros(10)] * max(0, Hn - len(self.hist))) + self.hist[-Hn:])
            zhat = mlp_np(policy['adapt'], hv)
            mu = mlp_np(policy['actor'], np.r_[obs, zhat])
        else:
            mu = mlp_np(policy, obs)
        if not self.launched:
            v = math.hypot(xmeas[3], xmeas[4]) if v_launch is None else v_launch
            if v >= self.launch['v_hand']:
                self.launched, self.handover = True, self.t
            else:
                s_off, e_y, e_psi = pl.path_coords(idx, xmeas)
                d_ref = pl.ref_at(idx, s_off, pl.u)[0]
                d_l = float(np.clip(d_ref - 1.0 * e_y - 0.8 * e_psi, -N0['s_max'], N0['s_max']))
                mu = (np.array([d_l, self.launch['current']]) - cm.A_OFF) / cm.A_SCALE
        a = mu + (action_noise if action_noise is not None else 0.0)
        self.last_a = a
        cmd = a * cm.A_SCALE + cm.A_OFF
        for k, v in (('x', xmeas), ('obs', obs), ('mu', mu), ('a', a), ('err', err), ('idx', idx), ('applied', cmd)):
            self.rec[k].append(v)
        self.peak_ey = max(self.peak_ey, abs(float(err[0])))
        self.t += 1
        return dict(mu=mu, a=a, cmd=cmd, obs=obs, err=err, idx=idx, progress=self.prog, launched=self.launched)

    def check_stop(self, err, xmeas, safety_ey):
        """real_car.run's failure checks on the measured state: '' (fly on), 'wall' (|e_y| > safety_ey: the room's safety
        envelope), 'spin' (|e_psi| > FAIL_EPSI) or 'stall' (launched and vx < 0.3 m/s)."""
        if abs(err[0]) > safety_ey:
            return 'wall'
        if abs(err[1]) > sn.FAIL_EPSI:
            return 'spin'
        if self.launched and xmeas[3] < 0.3:
            return 'stall'
        return ''

    def record(self, failed=False, crash='', safety_ey=None):
        """The trial in real_car.run's format; the real car has no ground truth, so the 'true' entries are the measured."""
        R = {k: np.array(v) for k, v in self.rec.items()}
        x = R['x']
        return dict(x=x, x_true=x, x_end=x[-1] if len(x) else None, idx_end=self.idx, obs=R['obs'], obs_true=R['obs'],
                    mu=R['mu'], a=R['a'], err=R['err'], err_true=R['err'], idx=R['idx'], applied=R['applied'],
                    progress=self.prog, laps=self.prog / self.pl.L, failed=bool(failed), crash=crash,
                    peak_ey=self.peak_ey, max_ey=safety_ey, steps=len(x), car='real', handover=self.handover,
                    plan=self.pname)
