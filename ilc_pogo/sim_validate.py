#!/usr/bin/env python3
"""
sim_validate.py RUN [RUN ...]: the sim-only stopping rule (the car's stop_pick.py) -- every snapshot of a sim stage
scored IN SIMULATION ONLY (no robot data) on the validation goals (seed 301, disjoint from the reserved test seed):
  nominal       the nominal model, perturbed drops, 10 s
  dr            each lane on its own DR draw (pogo_jax.dr_params, the family the DR variants train on), 10 s
  long          the nominal model, 20 s (hop-to-hop stationarity)
Rule (the car's finding: the DR-family scores track transfer, the nominal one does not): pick the snapshot with the
best mean success over the non-nominal conditions (dr, long); ties -> the later snapshot.
-> RUN/validate.json, RUN/policy_pick.npz (a copy of the picked snapshot).
"""
import glob
import json
import os
import shutil
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax                                       # noqa: E402
import jax.numpy as jnp                          # noqa: E402
import numpy as np                               # noqa: E402

import common as C                               # noqa: E402
import pogo_jax as pj                            # noqa: E402

G = np.repeat(C.goals_from_seed(8, C.VAL_SEED), 4, 0)
n = len(G)


def score(params, cond):
    T = 2000 if cond == 'long' else 1000
    x0 = pj.start_states(jax.random.PRNGKey(77), G, 1.0)
    p = pj.dr_params(jax.random.PRNGKey(78), n, 1.0) if cond == 'dr' else pj.nominal_params(n)
    jp = [(jnp.asarray(W), jnp.asarray(b)) for W, b in params]
    out = pj.rollout(lambda o: pj.mlp(jp, o), T, x0, jnp.asarray(G), p)
    xs, fe = np.asarray(out['x']), np.asarray(out['failed_end'])
    ms = [C.hop_metrics(xs[i], G[i], fe[i]) for i in range(n)]
    ok = [m for m in ms if not m['failed'] and m['rms_apex'] is not None]
    return dict(success=float(np.mean([m['success'] for m in ms])),
                apex=float(np.mean([m['rms_apex'] for m in ok])) if ok else None,
                dv=float(np.mean([abs(m['v_err']) for m in ok])) if ok else None)


def main():
    for run in sys.argv[1:]:
        snaps = sorted(glob.glob(os.path.join(run, 'policy_it*.npz')))
        res = {}
        for f in snaps:
            pol = C.load_policy(f)
            res[os.path.basename(f)] = {c: score(pol, c) for c in ('nominal', 'dr', 'long')}
            r = res[os.path.basename(f)]
            print(run, os.path.basename(f), {c: round(v['success'], 3) for c, v in r.items()},
                  'dv(dr)', r['dr']['dv'] and round(r['dr']['dv'], 3), flush=True)
        key = lambda k: (np.mean([res[k][c]['success'] for c in ('dr', 'long')]), k)
        pick = max(res, key=key)
        shutil.copy(os.path.join(run, pick), os.path.join(run, 'policy_pick.npz'))
        json.dump(dict(scores=res, pick=pick), open(os.path.join(run, 'validate.json'), 'w'), indent=1)
        print(run, 'PICK', pick)


if __name__ == '__main__':
    main()
