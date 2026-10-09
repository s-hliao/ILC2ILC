"""
Trial-1 trajectory optimization as the quadruped's (ilc_gen.QuadILCStageSolver.init_trajopt):
CasADi Opti + IPOPT, direct multiple shooting with the dynamics as hard constraints, solved
once, started from a simpler model's plan (the quadruped's "srb" guess):

  1. linear-tire TO (lifted_ilc.linear_tire_model: no tire saturation, no wheel slip) from
     the reference and its finite-difference controls;
  2. Fiala TO (ilc_f1tenth/dynamics.py: brush tires with combined slip and wheel-spin
     dynamics, Cf = Cr = 225 N/rad -- the model the earlier iLQR ran with) from step 1.

Cost: the task cost the ILC is scored on (weights.json: Q on samples 0..N-1, Q_f on N, R
on u). Constraints: x_0 = reference start; |current slew| <= 300, |steer rate| <= 3.2 rad/s,
|steering| <= max_steer, vx >= -0.5 (the NMPC's bound: both tire models make a lateral
force from steering at standstill, which a car starting on a curve at full lock cannot
avoid braking slightly backward against). RK4 with `substeps` per control interval: the wheel-spin
mode of the Fiala model reaches ~1e3 1/s near standstill.

    python3 trajopt.py mocap_square --rate 40
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import casadi as ca
import numpy as np

try:
    import tqdm  # noqa: F401
except ImportError:
    sys.modules['tqdm'] = types.SimpleNamespace(tqdm=lambda it, **kw: it)

HERE = Path(__file__).resolve().parent
from lifted_ilc import SNAPSHOT, F110, linear_tire_model, _get_tire_params   # noqa: E402

sys.path.insert(0, str(HERE.parent))                     # ilc_f1tenth/dynamics.py
import dynamics as fiala_dynamics                         # noqa: E402

NX, NU = 9, 2


def step_function(model, car, dt, substeps):
    p = _get_tire_params(car)
    x, u = ca.SX.sym('x', NX), ca.SX.sym('u', NU)
    if model == 'fiala':
        f = lambda x_, u_: fiala_dynamics.continous_dynamics_model(x_, u_, ca.DM(p), car, False)
    elif model == 'linear':
        f = lambda x_, u_: linear_tire_model(x_, u_, car, p[0], p[1])
    else:
        raise ValueError(model)
    h, xn = dt / substeps, x
    for _ in range(substeps):
        k1 = f(xn, u); k2 = f(xn + h / 2 * k1, u); k3 = f(xn + h / 2 * k2, u); k4 = f(xn + h * k3, u)
        xn = xn + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ca.Function(f'{model}_step', [x, u], [xn])


def weights():
    w = json.loads((SNAPSHOT / 'weights.json').read_text())
    return (np.array(w[k]) for k in ('Q', 'Q_f', 'R'))


def trajopt(ref, dt, model='fiala', guess=None, substeps=16, max_iter=3000, car=None):
    """Returns X (N+1, 9), U (N, 2), info (success, IPOPT status, iterations, cost, max
    dynamics residual)."""
    car = car or F110()
    N = len(ref) - 1
    Q, Qf, R = weights()
    F = step_function(model, car, dt, substeps).map(N)
    opti = ca.Opti()
    X = opti.variable(NX, N + 1)
    U = opti.variable(NU, N)
    E = X - ref.T
    cost = 0.5 * ca.sum2(ca.sum1(E[:, :N] * ca.mtimes(ca.DM(Q), E[:, :N])))
    cost += 0.5 * ca.mtimes([E[:, N].T, ca.DM(Qf), E[:, N]])
    cost += 0.5 * ca.sum2(ca.sum1(U * ca.mtimes(ca.DM(R), U)))
    opti.minimize(cost)
    opti.subject_to(X[:, 0] == ref[0])
    opti.subject_to(X[:, 1:] == F(X[:, :N], U))
    opti.subject_to(opti.bounded(-300.0, U[0, :], 300.0))
    opti.subject_to(opti.bounded(-car['max_steer_vel'], U[1, :], car['max_steer_vel']))
    opti.subject_to(opti.bounded(-car['max_steer'], X[8, :], car['max_steer']))
    opti.subject_to(X[3, :] >= -0.5)
    if guess is None:                                     # reference + finite-difference rates
        Xg = ref.copy()
        Ug = np.column_stack([np.diff(ref[:, 7]), np.diff(ref[:, 8])]) / dt
        Ug = np.clip(Ug, [-300, -car['max_steer_vel']], [300, car['max_steer_vel']])
    else:
        Xg, Ug = guess
    opti.set_initial(X, Xg.T)
    opti.set_initial(U, Ug.T)
    opti.solver('ipopt', {'print_time': False, 'expand': True},
                {'print_level': 0, 'sb': 'yes', 'max_iter': int(max_iter)})
    t0 = time.time()
    try:
        sol = opti.solve()
        ok = True
    except RuntimeError:
        sol, ok = opti.debug, False
    stats = opti.stats()
    Xs = np.array(sol.value(X)).T
    Us = np.array(sol.value(U)).T.reshape(N, NU)
    step = step_function(model, car, dt, substeps)
    dyn = max(float(np.abs(np.asarray(step(Xs[k], Us[k])).ravel() - Xs[k + 1]).max()) for k in range(N))
    info = dict(success=ok, status=stats.get('return_status', ''), iterations=int(stats.get('iter_count', -1)),
                cost=float(sol.value(cost)), dynamics_residual=dyn, seconds=time.time() - t0,
                model=model, substeps=substeps, N=N, dt=dt, rate_hz=1.0 / dt, solver='IPOPT (CasADi Opti)')
    return Xs, Us, info


def trajopt_two_stage(ref, dt, substeps=16, max_iter=3000):
    """Linear-tire plan, then the Fiala plan from it (the quadruped's SRB-guess pattern)."""
    Xl, Ul, il = trajopt(ref, dt, 'linear', None, substeps, max_iter)
    Xl[:, 6] = Xl[:, 3] / F110()['rw']                    # wheel spin at rolling speed
    Xf, Uf, inf = trajopt(ref, dt, 'fiala', (Xl, Ul), substeps, max_iter)
    inf['converged'] = bool(inf['success'])
    inf['guess'] = dict(il, converged=bool(il['success']))
    return Xf, Uf, inf


def main():
    from initial_to import make_reference
    ap = argparse.ArgumentParser()
    ap.add_argument('reference')
    ap.add_argument('--rate', type=float, default=40.0)
    ap.add_argument('--substeps', type=int, default=16)
    a = ap.parse_args()
    ref, dt = make_reference(a.reference, a.rate)
    X, U, info = trajopt_two_stage(ref, dt, a.substeps)
    out = HERE / 'references' / f'{a.reference}_{a.rate:g}hz_ipopt_to.npz'
    np.savez(out, reference=ref, controls=U[None], to_states=X, dt=dt, to_info=json.dumps(info))
    print(json.dumps(info))
    print('saved', out)


if __name__ == '__main__':
    main()
