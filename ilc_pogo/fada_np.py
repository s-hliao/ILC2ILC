"""
fada_np.py: the FADA-style planner + inverse-dynamics policy on the "real" pogo sticks (numpy, jax-free; the
policies.py interface). a_t = IDM([o_t, P(o_t)]): the planner P(o_t) proposes the state deviation H periods ahead
(the obs entries `dims`), the IDM the action that reaches it. npz keys: kind='fada', W{i}/b{i} (the IDM, LoRA merged),
PW{i}/Pb{i} (the planner), dims, H.
"""
import numpy as np

from common import mlp


class FADAPolicy:
    def __init__(self, idm, planner, dims, H):
        self.idm = [(np.asarray(W, np.float64), np.asarray(b, np.float64)) for W, b in idm]
        self.planner = [(np.asarray(W, np.float64), np.asarray(b, np.float64)) for W, b in planner]
        self.dims, self.H = np.asarray(dims), int(H)

    def reset(self, n):
        pass

    def __call__(self, o, x, g):
        d = mlp(self.planner, o, np)
        return mlp(self.idm, np.concatenate([o, d], -1), np)


def save(path, idm, planner, dims, H, **extra):
    np.savez(path, kind='fada', dims=np.asarray(dims), H=int(H),
             **{f'W{i}': np.asarray(W) for i, (W, b) in enumerate(idm)}, **{f'b{i}': np.asarray(b) for i, (W, b) in enumerate(idm)},
             **{f'PW{i}': np.asarray(W) for i, (W, b) in enumerate(planner)},
             **{f'Pb{i}': np.asarray(b) for i, (W, b) in enumerate(planner)}, **extra)


def load(path):
    d = np.load(path, allow_pickle=True)
    ni = len([k for k in d.files if k.startswith('W')])
    npl = len([k for k in d.files if k.startswith('PW')])
    return FADAPolicy([(d[f'W{i}'], d[f'b{i}']) for i in range(ni)], [(d[f'PW{i}'], d[f'Pb{i}']) for i in range(npl)],
                      d['dims'], int(d['H']))
