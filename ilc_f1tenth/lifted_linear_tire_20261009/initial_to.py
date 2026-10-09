"""
Trial-1 controls from a nonlinear trajectory optimization, as the quadruped's full-body TO
(a richer model than the one the ILC learns with): an acados NMPC (SQP, NONLINEAR_LS) solved
once over the whole reference, with the parameters, initial-state constraint and warm start
of ilc_f1tenth_ilqr_defect_aware.solve_nmpc_initial_solution, plus its status and iteration
count, which that function only prints.

  model "fiala": llampc's Fiala/brush-tire NMPC (nmpc_gen_fiala_fixed: export_model and
                 create_ocp unchanged), with ilc_f1tenth's params.py car (Cf = Cr = 225 N/rad,
                 mu 0.65 -- the Fiala model the earlier iLQR ran with; llampc's own params.py
                 has no Cf/Cr and falls back to 5 N/rad), built in ./solvers
  model "blend": the snapshot's blend NMPC (blend_solver.setup_mpc)

References are rebuilt at any control rate from the snapshot's generators: the S-curve
from trajectory_generator.generate_s_curve's defaults (dt = 1/35 reproduces
references/s_curve.npz exactly), the figure-eight from generate_figure_eight_reference.py's
construction with dt as a parameter.
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
from scipy.interpolate import splprep, splev

try:
    import tqdm  # noqa: F401
except ImportError:
    sys.modules['tqdm'] = types.SimpleNamespace(tqdm=lambda it, **kw: it)

from lifted_ilc import SNAPSHOT, F110                      # noqa: E402
import blend_solver                                       # noqa: E402
from blend_solver import setup_mpc, _get_tire_params      # noqa: E402
from trajectory_generator import generate_s_curve         # noqa: E402


def figure_eight(dt, car=None):
    """generate_figure_eight_reference.py with dt a parameter (1/35 there)."""
    car = car or F110()
    L, lr = car['lf'] + car['lr'], car['lr']
    wx = np.array([.5, 1.5, 2.5, 3.5, 4.5, 3.5, 2.5, 1.5, .5])
    wy = np.array([.5, -.5, .5, 1.5, .5, -.5, .5, 1.5, .5])
    tck, knots = splprep([wx, wy], s=0, per=True)
    u = np.linspace(knots[2], knots[2] + 1, 10001); up = u % 1
    xy = np.array(splev(up, tck)).T
    der = np.array(splev(up, tck, der=1)).T
    der2 = np.array(splev(up, tck, der=2)).T
    curv = (der[:, 0] * der2[:, 1] - der[:, 1] * der2[:, 0]) / np.linalg.norm(der, axis=1)**3
    scale = max(3., np.max(np.abs(curv)) * L / np.tan(.28)); xy *= scale; curv /= scale
    phi = np.unwrap(np.arctan2(der[:, 1], der[:, 0])); rot = phi[0]
    matrix = np.array([[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]])
    xy = (xy - xy[0]) @ matrix.T; phi -= rot
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]; length = arc[-1]
    accel_time, nominal_v = 1.5, 2.
    N = int(np.ceil((length / nominal_v + accel_time) / dt)); t = np.arange(N + 1) * dt; T = t[-1]
    v = np.ones_like(t) * nominal_v; a = np.zeros_like(t)
    for mask, z, sign in [(t < accel_time, t / accel_time, 1), (t > T - accel_time, (T - t) / accel_time, -1)]:
        v[mask] = nominal_v * (3 * z[mask]**2 - 2 * z[mask]**3)
        a[mask] = sign * nominal_v * 6 * z[mask] * (1 - z[mask]) / accel_time
    s = np.r_[0, np.cumsum((v[:-1] + v[1:]) * .5 * dt)]; ratio = length / s[-1]
    v *= ratio; a *= ratio; s *= ratio
    rear = np.column_stack([np.interp(s, arc, xy[:, i]) for i in range(2)])
    heading = np.interp(s, arc, phi); kappa = np.interp(s, arc, curv)
    yaw = v * kappa; steer = np.arctan(L * kappa)
    cog = rear + lr * np.column_stack([np.cos(heading), np.sin(heading)]); cog -= cog[0]
    factor = car['gear_ratio'] * 1.5 * car['pole_pairs'] * car['lambda'] / (car['mass'] * car['rw'])
    ref = np.column_stack([cog, heading, v, lr * yaw, yaw, v / car['rw'], a / factor, steer])
    assert np.isfinite(ref).all() and np.max(np.abs(steer)) < car['max_steer']
    return ref


def make_reference(name, rate_hz):
    dt = 1.0 / rate_hz
    if name == 's_curve':
        return generate_s_curve(F110(), dt=dt), dt
    if name == 'figure_eight':
        return figure_eight(dt), dt
    raise ValueError(name)


def _fiala_solver(N, dt, tag, max_iter):
    import os
    from acados_template import AcadosOcpSolver
    from llampc import nmpc_gen_fiala_fixed as fiala
    car = F110()                                  # ilc_f1tenth params (Cf = Cr = 225)
    ocp = fiala.create_ocp(fiala.export_model(car, exact=False), car, N, N * dt)
    if max_iter is not None:
        ocp.solver_options.nlp_solver_max_iter = int(max_iter)
    d = Path(__file__).resolve().parent / 'solvers' / f'fiala_{tag}'
    d.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    try:
        os.chdir(d)
        return AcadosOcpSolver(ocp, json_file='f1tenth_acados_ocp.json', build=True)
    finally:
        os.chdir(cwd)


def solve_initial_to(ref, dt, tag, max_iter=None, model='fiala'):
    """The NMPC over the whole reference. Returns X (N+1, 9), U (N, 2), info (status 0 =
    converged)."""
    N = ref.shape[0] - 1
    if max_iter is not None:
        tag = f'{tag}_it{int(max_iter)}'
    if model == 'fiala':
        solver = _fiala_solver(N, dt, tag, max_iter)
    elif model == 'blend':
        create = blend_solver.create_ocp
        if max_iter is not None:      # acados fixes the SQP cap at build time (snapshot: 20)
            def create_more(*args, **kw):
                ocp = create(*args, **kw)
                ocp.solver_options.nlp_solver_max_iter = int(max_iter)
                return ocp
            blend_solver.create_ocp = create_more
        try:
            solver = setup_mpc(steps=N, horizon=N * dt, solver_config=f'lifted_{tag}', build=True)
        finally:
            blend_solver.create_ocp = create
    else:
        raise ValueError(model)
    tire = _get_tire_params(F110())
    for k in range(N + 1):
        solver.set(k, 'p', np.concatenate([tire, ref[k, :6]]))
    solver.set(0, 'lbx', ref[0]); solver.set(0, 'ubx', ref[0])
    for k in range(N + 1):
        solver.set(k, 'x', ref[k])
    for k in range(N):
        solver.set(k, 'u', np.zeros(2))
    status = int(solver.solve())
    X = solver.get_flat('x').reshape(N + 1, 9)
    U = solver.get_flat('u').reshape(N, 2)
    info = dict(status=status, converged=status == 0, cost=float(solver.get_cost()),
                sqp_iter=int(solver.get_stats('sqp_iter')),
                residuals=np.asarray(solver.get_residuals()).tolist(),
                N=N, dt=dt, rate_hz=1.0 / dt,
                model={'fiala': 'llampc Fiala NMPC (brush tire), Cf=Cr=225',
                       'blend': 'snapshot blend NMPC (linear/brush blend)'}[model]
                      + ', acados SQP, NONLINEAR_LS')
    return X, U, info


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('reference', choices=['s_curve', 'figure_eight'])
    ap.add_argument('--rate', type=float, default=40.0)
    ap.add_argument('--max-iter', type=int, default=200)
    ap.add_argument('--model', choices=['fiala', 'blend'], default='fiala')
    a = ap.parse_args()
    ref, dt = make_reference(a.reference, a.rate)
    tag = f"{a.reference}_{a.rate:g}hz"
    X, U, info = solve_initial_to(ref, dt, tag, a.max_iter, a.model)
    out = Path(__file__).resolve().parent / 'references' / f'{tag}_{a.model}_to.npz'
    out.parent.mkdir(exist_ok=True)
    np.savez(out, reference=ref, controls=U[None], to_states=X, dt=dt, to_info=json.dumps(info))
    print(json.dumps(info))
    print('saved', out)


if __name__ == '__main__':
    main()
