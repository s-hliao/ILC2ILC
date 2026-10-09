"""
ILC trials on the F1TENTH gym bridge (public odometry, 35 Hz fixed deadlines), from the
stored epoch-0 controls the snapshot's validation runs started from.

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
    ap.add_argument('--history', default=str(SNAPSHOT / 'references/s_curve_epoch0.npz'))
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

    h = np.load(a.history, allow_pickle=True)
    ref, U, dt = h['reference'], h['controls'][0].copy(), float(h['dt'])
    w = json.loads((SNAPSHOT / 'weights.json').read_text())
    Q, Qf, R = (np.array(w[k]) for k in ('Q', 'Q_f', 'R'))
    car = exp.F110()
    p = exp._get_tire_params(car)
    ilc = None
    if a.method == 'lifted':
        ilc = LiftedILC(ref, dt, U, Q, Qf, R, Qu_diag=a.qu, gain=a.gain, model=a.model,
                        safeguard=not a.no_safeguard, accept_tol=a.accept_tol, car=car)
    (a.output / 'config.json').write_text(json.dumps(dict(vars(a), output=str(a.output)), indent=2))

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
