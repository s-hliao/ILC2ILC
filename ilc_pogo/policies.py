"""
policies.py: numpy policy objects for the "real" pogo sticks (real_pogo.simulate), jax-free.
Interface: obj.reset(n) before a batch of n runs; obj(o (n, NOBS), x (n, 6), g (n, 2)) -> normalized action mean (n, 2).
load_any(path): 'expert' -> the Raibert controller; an npz -> by its 'kind' entry:
  'mlp' (default; ours, PPO+DR)  -> MLPPolicy
  'rma'                          -> rma_np.RMAPolicy  (actor + adaptation module, policy_*.npz + rma_adapt.npz)
  'fada'                         -> fada_np.FADAPolicy
"""
import os

import numpy as np

from common import mlp, expert_action, load_policy


class MLPPolicy:
    def __init__(self, params):
        self.params = [(np.asarray(W, np.float64), np.asarray(b, np.float64)) for W, b in params]

    def reset(self, n):
        pass

    def __call__(self, o, x, g):
        return mlp(self.params, o, np)


class ExpertPolicy:
    def reset(self, n):
        pass

    def __call__(self, o, x, g):
        return expert_action(x, g, np)


def load_any(path):
    if path in (None, 'expert'):
        return ExpertPolicy()
    d = np.load(path, allow_pickle=True)
    kind = str(d['kind']) if 'kind' in d.files else 'mlp'
    if kind == 'rma':
        import rma_np
        return rma_np.load(path)
    if kind == 'fada':
        import fada_np
        return fada_np.load(path)
    return MLPPolicy(load_policy(path))


def as_policy(pol):
    if pol is None or (isinstance(pol, str) and pol == 'expert'):
        return ExpertPolicy()
    if isinstance(pol, str) and os.path.exists(pol):
        return load_any(pol)
    if isinstance(pol, list):
        return MLPPolicy(pol)
    return pol
