# smoke test of flows_quad.py: one cell, iteration 0, 1 jump per goal (log/dilc/flows/quad/smoke, not used by the analysis)
import os, sys, glob, time, numpy as np
GPU = sys.argv[1] if len(sys.argv) > 1 else "1"
sys.argv = ["flows_quad.py", "list"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flows_quad as FQ
c = os.path.join(FQ.HERE, "smoke/recipe_sd9001/real_r4")
pass  # (flown already)
F = FQ.Fields(f"{c}/it0/policy/bank.json", GPU)
w = F.weights(f"{c}/it0/policy/policy.npz")
sx = F.sx
for f in sorted(glob.glob(f"{c}/it0/flows_val/*.npz"))[:14]:
    z = np.load(f); t1 = time.time()
    r = F.episode(z, w, w)
    nv = lambda v: np.linalg.norm(v / sx, axis=-1).mean()
    print(os.path.basename(f), "fell", r["fell"], f"field vs plan {nv(r['v_real'] - r['v_plan']):.3f} vs sim0 {nv(r['v_real'] - r['v_sim0']):.3f}",
          f"sim0 vs plan {nv(r['v_sim0'] - r['v_plan']):.3f} tube {nv(r['x_real'] - r['x_ref']):.3f}", f"({time.time() - t1:.0f} s)", flush=True)
