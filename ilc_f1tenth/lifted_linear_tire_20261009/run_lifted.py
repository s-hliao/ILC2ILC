"""
ILC trials on the F1TENTH gym bridge (public odometry, 35 Hz fixed deadlines), from the
controls of a nonlinear trajectory optimization (initial_to.py: the snapshot's blend NMPC,
solved once over the whole reference at --rate), or from a stored history (--history).

  --method lifted   : LiftedILC (lifted_ilc.py), G from --model
  --method ilqr     : the snapshot's defect-aware iLQR (prepare_update), --alpha
  --method repeat   : fly the initial controls every trial (run-to-run noise floor)
"""
import argparse
import json
import threading
import time
from pathlib import Path

import sys
import types

import numpy as np

try:
    import tqdm                                            # noqa: F401  (snapshot imports it)
except ImportError:                                        # only its old rollout uses it
    sys.modules['tqdm'] = types.SimpleNamespace(tqdm=lambda it, **kw: it)

from lifted_ilc import LiftedILC, SNAPSHOT, trajectory_cost, position_rmse
import ilc_f1tenth_ilqr_defect_aware as exp               # snapshot (on sys.path)
from fixed_deadline_rollout import rollout


class Node(exp.F1tenth_ILC):
    def __init__(self):
        self.audit_lock = threading.Lock()
        super().__init__()

    def pose_callback(self, msg):
        with self.audit_lock:
            super().pose_callback(msg)
            self.audit_odom_stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            self.audit_receive_time = time.monotonic()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['lifted', 'ilqr', 'repeat'], default='lifted')
    ap.add_argument('--model', choices=['linear', 'blend_linear', 'blend'], default='linear')
    ap.add_argument('--reference', choices=['s_curve', 'figure_eight'], default='s_curve')
    ap.add_argument('--rate', type=float, default=40.0, help='control / execution rate, Hz')
    ap.add_argument('--to-max-iter', type=int, default=1000,
                    help="SQP iteration cap for the initial TO (the snapshot's NMPC: 20, which "
                         "stops short of acados' 1e-6 stationarity tolerance)")
    ap.add_argument('--allow-unconverged-to', action='store_true')
    ap.add_argument('--to-model', choices=['fiala', 'blend'], default='fiala',
                    help='NMPC model of the initial TO (the lifted ILC learns with the linear tire)')
    ap.add_argument('--history', default=None,
                    help='reuse (reference, controls[0], dt) from a stored history instead')
    ap.add_argument('--trials', type=int, default=15, help='trials flown, the first with U0')
    ap.add_argument('--qu', type=float, nargs=2, default=[1e-4, 1.0])
    ap.add_argument('--gain', type=float, default=1.0)
    ap.add_argument('--no-safeguard', action='store_true')
    ap.add_argument('--accept-tol', type=float, default=0.02)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        ap.error('output exists; use a new directory')
    a.output.mkdir(parents=True)

    if a.history:
        h = np.load(a.history, allow_pickle=True)
        ref, U, dt = h['reference'], h['controls'][0].copy(), float(h['dt'])
        to_info = json.loads(str(h['to_info'])) if 'to_info' in h.files else None
    else:                                        # nonlinear TO for trial 1
        from initial_to import make_reference, solve_initial_to
        ref, dt = make_reference(a.reference, a.rate)
        X_to, U, to_info = solve_initial_to(ref, dt, f'{a.reference}_{a.rate:g}hz', a.to_max_iter,
                                         a.to_model)
        np.savez(a.output / 'initial_to.npz', reference=ref, controls=U[None], to_states=X_to, dt=dt,
                 to_info=json.dumps(to_info))
        print('INITIAL_TO', json.dumps(to_info), flush=True)
        if not to_info['converged'] and not a.allow_unconverged_to:
            raise RuntimeError(f"initial TO did not converge (acados status {to_info['status']})")
    (a.output / 'initial_to.json').write_text(json.dumps(to_info, indent=2))
    w = json.loads((SNAPSHOT / 'weights.json').read_text())
    Q, Qf, R = (np.array(w[k]) for k in ('Q', 'Q_f', 'R'))
    car = exp.F110()
    p = exp._get_tire_params(car)
    ilc = None
    if a.method == 'lifted':
        ilc = LiftedILC(ref, dt, U, Q, Qf, R, Qu_diag=a.qu, gain=a.gain, model=a.model,
                        safeguard=not a.no_safeguard, accept_tol=a.accept_tol, car=car)
    (a.output / 'config.json').write_text(json.dumps(dict(vars(a), output=str(a.output), dt=dt), indent=2))

    states, controls, metrics = [], [], []
    exp.rclpy.init()
    node = Node()

    def fly(u):
        if not exp.reset_and_wait(node, ref[0]):
            raise RuntimeError('reset failed')
        x = rollout(node, u, ref[0], dt, car['gear_ratio'], car['pole_pairs'], car['lambda'],
                    car['mass'], car['rw'], car['max_steer'])
        for _ in range(10):
            node.publish_control(0., 0.)
            exp.rclpy.spin_once(node, timeout_sec=.01)
        if not np.isfinite(x).all():
            raise RuntimeError('nonfinite rollout')
        return x

    try:
        for trial in range(a.trials):
            X = fly(U)
            states.append(X); controls.append(U.copy())
            m = dict(trial=trial, cost=trajectory_cost(X, U, ref, Q, Qf, R),
                     position_rmse=position_rmse(ref, X),
                     lateness_max=float(np.max(node.schedule_diagnostics['lateness'])))
            period = np.diff(node.schedule_diagnostics['actual'])
            m.update(rate_hz=float(1.0 / np.mean(period)), period_ms_min=float(1e3 * period.min()),
                     period_ms_max=float(1e3 * period.max()))
            t0 = time.monotonic()
            if a.method == 'lifted':
                r = ilc.update(X)
                m.update(rejected=bool(r.get('rejected', False)), qu_scale=r.get('qu_scale'),
                         predicted_cost=r['solver']['e_pred'], qp=r['solver']['status'],
                         du_norm=r['du_norm'])
                U = ilc.U.copy()
            elif a.method == 'ilqr':
                rep = exp.prepare_update(X, U, ref, dt, car, p, Q, Qf, R, [a.alpha],
                                         a.output / f'update_{trial + 1}')
                m.update(selection=rep['selection']['status'])
                U = np.load(a.output / f'update_{trial + 1}/prepared_candidate.npz')['controls']
            m['update_seconds'] = time.monotonic() - t0
            metrics.append(m)
            np.savez(a.output / 'history.npz', reference=ref, states=np.asarray(states),
                     controls=np.asarray(controls), dt=dt)
            (a.output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
            print('TRIAL', json.dumps(m), flush=True)
    finally:
        for _ in range(10):
            node.publish_control(0., 0.)
            exp.rclpy.spin_once(node, timeout_sec=.01)
        node.destroy_node()
        exp.rclpy.shutdown()


if __name__ == '__main__':
    main()
