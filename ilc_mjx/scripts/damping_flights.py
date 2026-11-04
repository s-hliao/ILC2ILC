#!/usr/bin/env python3
"""damping_flights.py REFERENCE.npz OUTDIR (2026-11-01): the three force files of the Go1 damping test, for the ILC node's initial_forces: the plan's forces, and rear Fz +20 N and rear Fx +15 N
through the rear-contact phase. Fly one trial of each from the same stance, then fit_damping.py u0's trial, the two others' trials, --ref REFERENCE.npz."""
import os, sys
import numpy as np
z = np.load(sys.argv[1], allow_pickle=True); u = np.asarray(z["u_ref"], float); os.makedirs(sys.argv[2], exist_ok=True)
cfg = z["config"].item() if isinstance(z["config"].item(), dict) else __import__("json").loads(str(z["config"]))
Nd, Ns = cfg["phases"][0], cfg["phases"][1]
np.savez(os.path.join(sys.argv[2], "u0.npz"), U=u)
for n, j, d in (("rearFz", 3, 20.0), ("rearFx", 2, 15.0)):
    U = u.copy(); U[Nd:Nd + Ns, j] += d; np.savez(os.path.join(sys.argv[2], n + ".npz"), U=U)
print("wrote", sys.argv[2], "rear phase", Nd, Nd + Ns)
