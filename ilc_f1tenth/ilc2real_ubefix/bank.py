"""The plan bank: drift / grip plans per track and target sideslip (trajopt_drift.py outputs in plans/), each with
its periodic LQR gains (plan_lqr.py), cached in plans/<name>_K.npy."""
import os

import numpy as np

import car_model as cm

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN_DIR = os.environ.get('F1T_PLANS') or os.path.join(HERE, 'plans')
TRACKS = ['mocap_square2fast', 'mocap_figfast']


def load_plan(name):
    d = dict(np.load(os.path.join(PLAN_DIR, f'{name}.npz'), allow_pickle=True))
    kf = os.path.join(PLAN_DIR, f'{name}_K.npy')
    if os.path.exists(kf):
        K = np.load(kf)
    else:                                         # casadi only here: the car's computer runs from the cached gains
        import plan_lqr
        K, _, _, info = plan_lqr.lqr_gains(d, dict(cm.NOMINAL))
        np.save(kf, K)
    N = int(d['N'])                               # plan_lqr.dense_gains: K on the fine s-grid (interval-wise constant)
    d['K'] = K[np.minimum((d['s'] / (float(d['length']) / N)).astype(int), N - 1)]
    d['name'] = name
    return d


def load_bank(names):
    return [load_plan(n) for n in names]
