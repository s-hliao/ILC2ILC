"""Car: the trajectories behind the before/after track figure and the along-the-lap figure (2026-10-31), simulation stand-in
for the hardware logs. Per tire, its practised track and plans (HW_RUNBOOK_N6_1031.md): plastic square2fast _a45 (stage
runs/hw/E5A45_<net>_s0_pl_square2fast_stage), rubber fig8R100 _x120 (runs/hw/F8R100_rb12_<net>_s0_stage). Each network
(PPO + DR, RMA; seed 0) after k = 0..3 stage iterations (k = 0 its own zero-shot network; 2 real laps per iteration) on
every car of that tire's stage, one 5-lap run from the start line with eval_free.py's run-0 start perturbation and seed
(so it is one of the validation runs). -> car_traj_sim.npz (one object array of dicts) + the plans' centrelines.
    cd ilc2real && source env_tight.sh && export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 && \
      JAX_PLATFORMS=cpu NUMBA_NUM_THREADS=1 python ../../paper/final/car_hw/car_traj_data.py
"""
import os
import sys

import numpy as np

CAR = "/home/henry/ilc_ws/src/ilc_f1tenth/ilc2real"
sys.path.insert(0, CAR)
import bank                                                    # noqa: E402
import policy_runtime as pr                                    # noqa: E402
import real_car as rc                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = f"{CAR}/runs"
PLASTIC = ["n04_real_nom", "n04_real_mass", "n04_real_mu", "n04_real_act", "n04_real_lag", "n04_harsh_combo",
           "n04_harsh_steer", "n04_harsh_sense", "n04_harsh_drift", "ax:mu=0.6000", "ax:mu=0.4667", "ax:mu=0.4000"]
RUBBER = ["ax:mu=1.667", "ax:mu=1.833,mass=1.1"]
TIRES = {"plastic": (PLASTIC, "mocap_square2fast_b0_free_a45", "E5A45_{net}_s0_pl_square2fast_stage"),
         "rubber": (RUBBER, "mocap_fig8R100_b0_free_x120", "F8R100_rb12_{net}_s0_stage")}
NETS = ("freePPON", "rmaN")
LAPS, SEED = 5.0, 301


def policy(net, stage, k, car):
    if k == 0:
        return pr.load_policy(f"{RUNS}/{net}_s0/policy_final.npz")
    d = f"{RUNS}/hw/{stage}" + ("" if k == 3 else f"_it{k}")
    return pr.load_policy(f"{d}/{car}_policy.npz")


def main():
    plans = {p["name"]: p for p in bank.load_bank([v[1] for v in TIRES.values()])}
    jobs, meta = [], []
    pert = np.random.default_rng(SEED).normal(size=5) * 0.5
    for tire, (cars, pn, stage_t) in TIRES.items():
        for net in NETS:
            stage = stage_t.format(net=net)
            for car in cars:
                for k in range(4):
                    jobs.append(dict(car_name=car, plan=plans[pn], policy=policy(net, stage, k, car), laps=LAPS, s0=0.0,
                                     pert=pert, seed=SEED))
                    meta.append(dict(tire=tire, net=net, car=car, k=k, plan=pn))
    res = rc.run_many(jobs, procs=int(os.environ.get("F1T_PROCS", "4")))
    out = []
    for m, r in zip(meta, res):
        pl = plans[m["plan"]]
        z = np.asarray(pl["z"])
        idx = np.asarray(r["idx"]) % len(z)
        x = np.asarray(r["x_true"])
        lm = rc.lap_metrics(r, pl)
        out.append(dict(m, X=x[:, 0], Y=x[:, 1], v=np.hypot(x[:, 3], x[:, 4]), idx=idx, idx_raw=np.asarray(r["idx"]), s=np.asarray(pl["s"])[idx],
                        ey=np.asarray(r["err_true"])[:, 0], steer=np.asarray(r["applied"])[:, 0],
                        crash=str(r.get("crash", "")), rms_ey=lm.get("rms_ey"), pace=lm.get("pace"), laps=r.get("laps")))
    # plan x = the planned world state (X, Y, psi, vx, vy, r, ...), as the rollouts' x_true; xy = the track centreline
    cl = {pn: dict(X=np.asarray(p["x"])[:, 0], Y=np.asarray(p["x"])[:, 1], v=np.hypot(np.asarray(p["x"])[:, 3], np.asarray(p["x"])[:, 4]),
                   cx=np.asarray(p["xy"])[:, 0], cy=np.asarray(p["xy"])[:, 1], s=np.asarray(p["s"]), kappa=np.asarray(p["kappa_s"]),
                   steer=np.asarray(p["z"])[:, 7], length=float(p["length"])) for pn, p in plans.items()}
    np.savez_compressed(os.path.join(HERE, "car_traj_sim.npz"), runs=np.array(out, dtype=object), plans=np.array(cl, dtype=object))
    for o in out:
        print(o["tire"], o["net"], o["car"], o["k"], f"rms_ey {o['rms_ey']} pace {o['pace']} crash {o['crash']!r}")


if __name__ == "__main__":
    main()
