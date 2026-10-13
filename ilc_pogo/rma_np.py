"""
rma_np.py: the RMA policy on the "real" pogo sticks (numpy; the policies.py interface). The actor pi(o, z) is the
teacher from ppo_dr.py --mode rma; z comes from the adaptation module phi(history of the last H (obs_t, a_{t-1}))
(deployment), or is given (oracle mode: the robot's TRUE normalized parameters, for the data-matching / exact-oracle
studies).
    load(path)              -> RMAPolicy with the adaptation module (policy_*.npz + rma_adapt.npz in the same dir)
    load(path, z=...)       -> oracle: z (NZ,) or (n, NZ) fixed
    true_z(params_dict)     -> the normalized z of a robot (real_pogo.robot_params)
"""
import os

import numpy as np

from common import mlp, load_policy


class RMAPolicy:
    def __init__(self, actor, adapt=None, hist=10, z=None, znom=None, zsc=None, zkeys=None):
        f = lambda ps: [(np.asarray(W, np.float64), np.asarray(b, np.float64)) for W, b in ps]
        self.actor, self.adapt = f(actor), (f(adapt) if adapt is not None else None)
        self.H, self.z = int(hist), (None if z is None else np.asarray(z, np.float64))
        self.znom, self.zsc, self.zkeys = znom, zsc, zkeys
        self.nobs = self.actor[0][0].shape[0] - (len(zkeys) if zkeys is not None else 6)

    def reset(self, n):
        self.buf = np.zeros((n, self.H, self.nobs + 2))
        self.last_a = np.zeros((n, 2))

    def __call__(self, o, x, g):
        n = len(o)
        if self.z is not None:
            zh = np.broadcast_to(self.z, (n, self.z.shape[-1]))
        else:
            self.buf = np.concatenate([self.buf[:, 1:], np.concatenate([o, self.last_a], -1)[:, None]], 1)
            zh = mlp(self.adapt, self.buf.reshape(n, -1), np)
        a = mlp(self.actor, np.concatenate([o, zh], -1), np)
        self.last_a = a
        return a


def load(path, z=None):
    d = np.load(path, allow_pickle=True)
    meta = dict(znom=np.asarray(d['znom']), zsc=np.asarray(d['zsc']), zkeys=[str(k) for k in d['zkeys']])
    adapt, hist = None, 10
    ad = os.path.join(os.path.dirname(path), 'rma_adapt.npz')
    if z is None:
        e = np.load(ad)
        adapt, hist = load_policy(ad), int(e['hist'])
    return RMAPolicy(load_policy(path), adapt, hist, z, **meta)


def true_z(params, path):
    """The normalized z of a robot (a parameter dict, e.g. real_pogo.robot_params(name)) for the actor at `path`."""
    d = np.load(path, allow_pickle=True)
    keys = [str(k) for k in d['zkeys']]
    return (np.array([params[k] for k in keys], np.float64) - np.asarray(d['znom'])) / np.asarray(d['zsc'])
