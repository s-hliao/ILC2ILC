# diagnose the sim0-vs-real field gap per SRB component (smoke cell)
import os, sys, glob, numpy as np
GPU = sys.argv[1] if len(sys.argv) > 1 else "1"
sys.argv = ["flows_quad.py", "list"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flows_quad as FQ
c = os.path.join(FQ.HERE, "smoke/recipe_sd9001/real_r4")
F = FQ.Fields(f"{c}/it0/policy/bank.json", GPU)
w = F.weights(f"{c}/it0/policy/policy.npz")
sx = F.sx
print("sx", np.round(sx, 4))
f = sorted(glob.glob(f"{c}/it0/flows_val/*g0.600_0.000*.npz"))[0]
z = np.load(f)
r = F.episode(z, w, w)
names = ["x", "z", "pitch", "vx", "vz", "wy"]
for nm, v in (("real-plan", r["v_real"] - r["v_plan"]), ("sim0-real", r["v_sim0"] - r["v_real"]), ("sim0-plan", r["v_sim0"] - r["v_plan"])):
    print(f"{nm:10s} mean/sx per comp:", dict(zip(names, np.round(np.abs(v / sx).mean(0), 3))))
for k in (0, 1, 10, 30, 50, 59):
    print(k, "real", np.round(r["v_real"][k] / sx, 2), "sim0", np.round(r["v_sim0"][k] / sx, 2), "plan", np.round(r["v_plan"][k] / sx, 2))
# the sim's own consistency: restored start state vs the real state at k (srb), per component
env, jnp = F.env, F.jnp
Nc = env.Nc; ks = np.arange(Nc); g = tuple(np.round(z["goal"], 4))
rec = {k_[4:]: z[k_] for k_ in z.files if k_.startswith("rec_")}
rec["pos"], rec["quat"], rec["q"] = rec["true_pos"], rec["true_quat"], rec["true_q"]
Qr, Vr = env.restore(rec, ks, ks < env.Ndc)
refB = env.references(np.repeat([g], Nc, 0))
xs = np.array(F.srb_of(jnp.asarray(Qr, jnp.float32), jnp.asarray(Vr, jnp.float32), refB))
xt = r["x_real"][:Nc] - (r["x_real"][0] - xs[0]) * np.r_[1, 1, 0, 0, 0, 0]
print("restored-vs-real state |diff|/sx per comp:", dict(zip(names, np.round(np.abs((xs - xt) / sx).mean(0), 3))))
