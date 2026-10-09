"""Solve the snapshot's blend NMPC once for a bundled reference and store the controls in
the history format run_lifted.py reads (reference, controls (1, N, 2), dt)."""
import sys
import types
import numpy as np
try:
    import tqdm  # noqa: F401
except ImportError:
    sys.modules['tqdm'] = types.SimpleNamespace(tqdm=lambda it, **kw: it)
from lifted_ilc import SNAPSHOT
import ilc_f1tenth_ilqr_defect_aware as exp

name = sys.argv[1]
h = np.load(SNAPSHOT / f'references/{name}.npz')
ref, dt = h['reference'], float(h['dt'])
X, U = exp.solve_nmpc_initial_solution(ref, dt)
np.savez(f'references/{name}_nmpc0.npz', reference=ref, controls=U[None], nmpc_states=X, dt=dt)
print('saved', U.shape, np.abs(U).max(0))
