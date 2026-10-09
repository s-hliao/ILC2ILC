"""The plan bank: drift / grip plans per track and target sideslip (trajopt_drift.py outputs in plans/), each with
its periodic LQR gains (plan_lqr.py), cached in plans/<name>_K.npy."""
import os

import numpy as np

import car_model as cm
import plan_lqr

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN_DIR = os.environ.get('F1T_PLANS') or os.path.join(HERE, 'plans')
TRACKS = ['mocap_square2fast', 'mocap_figfast']


def load_plan(name):
    d = dict(np.load(os.path.join(PLAN_DIR, f'{name}.npz'), allow_pickle=True))
    kf = os.path.join(PLAN_DIR, f'{name}_K.npy')
    if os.path.exists(kf):
        K = np.load(kf)
    else:
        K, _, _, info = plan_lqr.lqr_gains(d, dict(cm.NOMINAL))
        np.save(kf, K)
    d['K'] = plan_lqr.dense_gains(d, K)
    d['name'] = name
    return d


def load_bank(names):
    return [load_plan(n) for n in names]
