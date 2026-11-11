"""policy_resid.py VARIANT (2026-11-05; user: "show how the policy has changed as a residual from the initial zero-shot policy"): the
stage's change of the policy itself on flow_car.py's grid of starting states -- pi_k(o) - pi_0(o) for the steering and the motor current
(the network's outputs in physical units), o the observation at each (arc length, lateral offset) state of the variant (the same
starting states as its flow: base "orbit" = the zero-shot car's own lap shifted across the tube). -> <stem>_resid.npz. PPO+DR (plain MLP)
networks only."""
import math
import os
import sys

D = "/home/henry/ilc_ws/src/ilc_f1tenth/ilc2real"
os.environ.update(F1T_MU="0.6", F1T_PLANS=D + "/plans_nom06", F1T_TRACKS=D + "/../lifted_linear_tire_20261009/tracks_tight",
                  F1T_SAFETY_EY="0.5", F1T_NO_CRASH="1", NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1", JAX_PLATFORMS="cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else "ethz_simref"]
import flow_car as fc  # noqa: E402  (the variant table, grid, horizon)
import numpy as np  # noqa: E402

V = fc.V
os.chdir(D)
import bank  # noqa: E402
import car_model as cm  # noqa: E402
import real_car as rc  # noqa: E402
import sim_jax_np as sn  # noqa: E402

pl = bank.load_bank([V["plan"]])[0]
P = sn.PlanNP(pl)
L = float(pl["length"])
pols = [fc.load_policy(f"{fc.RUN}/its/it{k}_{fc.CAR}_policy.npz") for k in range(4)]
ORBIT = V.get("base") == "orbit"
if ORBIT:
    r0 = rc.run(fc.CAR, pl, pols[0], laps=1.05, s0=0.0, pert=np.zeros(5), seed=0, noise=False)
    orbit = {int(i): (np.asarray(x, float), float(e[0])) for i, x, e in zip(r0["idx"], r0["x_true"], r0["err_true"])}
    oi = np.array(sorted(orbit))
ss = np.linspace(0, L, fc.NS, endpoint=False)
es = fc.GRID
U = np.zeros((4, len(ss), len(es), 2))
for i, s in enumerate(ss):
    ig = int(round(s / P.ds)) % P.M
    if ORBIT:
        j = int(oi[np.argmin(np.abs((oi - ig + P.M // 2) % P.M - P.M // 2))])
        xo, eo = orbit[j]
        th = pl["x"][j][2]
    for jj, e in enumerate(es):
        if ORBIT:
            x = xo.copy()
            x[0] += -math.sin(th) * (e - eo)
            x[1] += math.cos(th) * (e - eo)
        else:                       # (2026-11-06) flow_car.py's start(): on the path, the plan's speed + e (vx, vy, r, wheel scaled)
            j = ig
            x = np.asarray(pl["x"][j], float).copy()
            if V["var"] == "lat":
                x[0] += -math.sin(x[2]) * e
                x[1] += math.cos(x[2]) * e
            else:
                f = (math.hypot(x[3], x[4]) + e) / math.hypot(x[3], x[4])
                x[3:7] *= f
        idx = P.project(j, x[:2])
        obs, _ = P.features(idx, x)
        for k in range(4):
            U[k, i, jj] = rc.mlp_np(pols[k], obs) * cm.A_SCALE + cm.A_OFF     # (steering rad, current A)
np.savez(os.path.join(HERE, f"{V['stem']}_resid.npz"), ss=ss, es=es, U=U)
print("steering change vs pi_0 (deg): mean |.|", [round(float(np.degrees(np.abs(U[k, :, :, 0] - U[0, :, :, 0]).mean())), 2) for k in range(4)],
      "max |.|", [round(float(np.degrees(np.abs(U[k, :, :, 0] - U[0, :, :, 0]).max())), 2) for k in range(4)])
print("current change vs pi_0 (A): mean |.|", [round(float(np.abs(U[k, :, :, 1] - U[0, :, :, 1]).mean()), 2) for k in range(4)])
