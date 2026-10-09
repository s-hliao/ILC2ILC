"""Offline check of one lifted step on a stored trial (no vehicle): the QP's linear
prediction against a nonlinear rollout of the blend model carrying that trial's defects."""
import argparse
import json
from pathlib import Path

import numpy as np

from lifted_ilc import (F110, LiftedILC, SNAPSHOT, make_model, tracking_error, trajectory_cost,
                        position_rmse, command_chain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--history', default=str(SNAPSHOT / 'references/s_curve_epoch0.npz'))
    ap.add_argument('--models', nargs='+', default=['linear', 'blend_linear', 'blend'])
    ap.add_argument('--qu', type=float, nargs=2, default=[1e-4, 1.0])
    ap.add_argument('--qu-scales', type=float, nargs='+', default=[0.01, 0.1, 1, 10, 100])
    a = ap.parse_args()
    h = np.load(a.history, allow_pickle=True)
    ref, X, U, dt = h['reference'], h['states'][0], h['controls'][0], float(h['dt'])
    w = json.loads((SNAPSHOT / 'weights.json').read_text())
    Q, Qf, R = (np.array(w[k]) for k in ('Q', 'Q_f', 'R'))
    blend_step, _ = make_model('blend', F110(), dt)
    # defects of the measured trial in the blend model: X[k+1] = f(X[k], U[k]) + d[k]
    d = np.array([X[k + 1] - np.asarray(blend_step(X[k], U[k])).ravel() for k in range(len(U))])
    print(f"measured: cost {trajectory_cost(X, U, ref, Q, Qf, R):.2f}  rmse {position_rmse(ref, X):.4f}")
    for model in a.models:
        for sc in a.qu_scales:
            ilc = LiftedILC(ref, dt, U, Q, Qf, R, Qu_diag=a.qu, model=model)
            du, info, G = ilc.step(X, U, sc)
            Un = U + du
            X_lin = X.copy()
            X_lin[1:] += (G @ du.reshape(-1)).reshape(-1, 9)
            # nonlinear check: blend + fixed defects, command states from the chain
            Xn = [X[0].copy()]
            for k in range(len(U)):
                Xn.append(np.asarray(blend_step(Xn[-1], Un[k])).ravel() + d[k])
            Xn = np.asarray(Xn)
            chain = command_chain(Un, X[0], ilc.car, dt)
            print(f"{model:13s} qu x{sc:<6g} {info['status']:>8s}  "
                  f"lin: cost {trajectory_cost(X_lin, Un, ref, Q, Qf, R):7.2f} rmse {position_rmse(ref, X_lin):.4f} | "
                  f"blend+defect: cost {trajectory_cost(Xn, Un, ref, Q, Qf, R):7.2f} rmse {position_rmse(ref, Xn):.4f} | "
                  f"|du|max I {np.abs(du[:,0]).max():6.1f} d {np.abs(du[:,1]).max():.3f}  "
                  f"steer max {np.degrees(np.abs(chain[:,1]).max()):.1f} deg")


if __name__ == '__main__':
    main()
