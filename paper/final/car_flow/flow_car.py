"""Car: the hardware stage re-converging the closed-loop flow inside the 0.5 m tube around the plan (2026-11-05; user: "the
flow diagram of the car retargeting in the 0.5 tube around the reference trajectory similar to the pogo stick"; the car
version of pogo_flow/flow_strip.py).
  frame       x = arc length s along the lap (m), y = lateral offset e_y from the plan (m), the tube = the room's safety
              envelope |e_y| <= 0.5 m (env_tight.sh F1T_SAFETY_EY; net_runner.py safety_ey on the car)
  rows        the nominal car (mu 0.6, the friction the network and plans were made at) under pi_0 (the reference), then
              the plastic car n04_real_nom (every friction x 0.4 / 0.6) under pi_0 .. pi_3 of one plastic stage run
              (runs/screen/PLSL_freePPON_s0: PPO+DR network seed 0, stage seed 9001; 3 iterations x 2 laps on
              square2fast); shown on the HELD-OUT peanut (clockwise, _a33), never practised
  arrows      where the car is H = 0.4 s later when started at (s, e_y) on the plan's state shifted sideways (heading,
              speeds on the plan): (delta s, delta e_y), one length factor for every row
  background  |delta e_y - delta e_y(reference)| in tolerance units (0.10 m, lap_metrics ey_ok)
  points      four laterally perturbed starts (e_y0 = +-0.30, +-0.15 m at four places), every 0.2 s for one lap
Rollouts: real_car.run, no sensing noise, the plastic runs' exact environment (plastic_sl.sh). The fields are computed once
into flow_car_data.npz (delete it to recompute). Writes car_flow.{pdf,png} to paper/final/.
    ~/miniconda3/envs/f1t/bin/python paper/final/car_flow/flow_car.py
"""
import glob
import math
import os
import sys

D = "/home/henry/ilc_ws/src/ilc_f1tenth/ilc2real"
os.environ.update(F1T_MU="0.6", F1T_PLANS=D + "/plans_nom06", F1T_TRACKS=D + "/../lifted_linear_tire_20261009/tracks_tight",
                  F1T_SAFETY_EY="0.5", F1T_NO_CRASH="1", NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1", JAX_PLATFORMS="cpu")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)                                          # paper/final
# variants (2026-11-05, user: "do both"): the plastic and rubber cars' lateral flow; the plastic car's SPEED flow (the plastic stage's
# measured gain is lap pace, not lateral tracking)
VARIANTS = {
    "plastic": dict(plan="mocap_peanutM_b0_free_a33", run="PLSL_freePPON_s0", car="n04_real_nom", var="lat", stem="car_flow",
                    label="plastic car"),
    "rubber": dict(plan="mocap_peanutM_b0_free_x120", run="SL50x120r2_freePPON_s0", car="ax:mu=1.667", var="lat",
                   stem="car_flow_rubber", label="rubber car"),
    "speed": dict(plan="mocap_peanutM_b0_free_a33", run="PLSL_freePPON_s0", car="n04_real_nom", var="speed",
                  stem="car_flow_speed", label="plastic car"),
    # the largest held-out score drop of PPO+DR's 108 plastic cells (9 runs x 12 cars; 0.82 -> 0.38, -54 %); SELECTED as the best
    # example -- disclose as such (the 216-cell median, both networks, is -19 %)
    # (2026-11-06, user: the plastic cost must "start a lot higher" and the car "get significantly faster"): of 36 plastic candidates
    # (12 cars x PLSL it3, LONG it8, PFAST x1.10 it3; plastic_J.py, two noise-free laps), the largest absolute drop of the stage's own
    # cost J among those without a failed lap -- PFAST_plastic_freePPON_s0_ss9001 n04_harsh_sense on the peanut x1.10 (practised):
    # J 617 -> 352 (-43 %), speed 2.10 -> 2.18 m/s, e_y 8.8 -> 6.0 cm. SELECTED as the best case; disclose as such
    "speed_hicost": dict(plan="mocap_peanutM_b0_free_a33_x110", run="PFAST_plastic_freePPON_s0_ss9001", car="n04_harsh_sense", var="speed",
                         stem="car_flow_speed_hicost", label="plastic car (harsh sense)"),
    "speed_best": dict(plan="mocap_peanutM_b0_free_a33", run="PLSLr2_freePPON_s0", car="n04_real_mass", var="speed",
                       stem="car_flow_speed_best", label="plastic car, mass changed (best of 108)"),
    # (2026-11-05, user: "show more than this, you said -54 %"): the same cell, the speed axis zoomed to where the car drives,
    # the reference's level in every row, the gap to the plan's speed as a share of the zero-shot gap
    "speed_best_zoom": dict(plan="mocap_peanutM_b0_free_a33", run="PLSLr2_freePPON_s0", car="n04_real_mass", var="speed",
                            stem="car_flow_speed_best_zoom", label="plastic car, mass changed (best of 108)",
                            grid=(-0.36, 0.12), starts=[(0.15, -0.30), (0.40, 0.08), (0.65, -0.20), (0.90, 0.0)]),
    # (2026-11-05, ETHZ_RULES.md): the PRACTISED ETHZMobil track (x4.3, 55 m, sim only), rubber car, the MEDIAN rubber PPO+DR run of
    # replicate 1 by score change (s0: -24.9 %; s2 -30.2, s1 -21.2). No reference row (a nominal-friction car cannot drive the rubber
    # plan); the shading is the lateral drift instead: moving away from the plan (warm) or back toward it (cool) over the 0.4 s
    "ethz": dict(plan="ethzmobil43_b0_free", run="ETHZ_rubber_freePPON_s0_ss9001", car="ax:mu=1.667", var="lat", stem="car_flow_ethz",
                 label="rubber car", ref=False, bg="drift", ns=60, track="practised ETHZMobil lap"),
    # the same, but the flow ON THE CAR'S OWN ORBIT: at each arc length the zero-shot car's actual state there (its speed, heading
    # error, sideslip, yaw rate, wheel speed; one noise-free lap) shifted sideways across the tube -- the plan-state slice above
    # misses the states this car drives (7 % over the plan's pace). Shading: the signed sideways push; the black line: where the push
    # is zero (the line the flow pulls the car onto)
    "ethz_orbit": dict(plan="ethzmobil43_b0_free", run="ETHZ_rubber_freePPON_s0_ss9001", car="ax:mu=1.667", var="lat",
                       stem="car_flow_ethz_orbit", label="rubber car", ref=False, bg="push", ns=60, base="orbit",
                       track="practised ETHZMobil lap"),
    # (2026-11-05) the same with a 1.0 s horizon: over 0.4 s the sideways drift is set by the heading the car already has (identical
    # starting states in every row), the policy acting through steering -> yaw -> heading needs longer; the pogo's arrow is one hop
    "ethz_orbit_1s": dict(plan="ethzmobil43_b0_free", run="ETHZ_rubber_freePPON_s0_ss9001", car="ax:mu=1.667", var="lat",
                          stem="car_flow_ethz_orbit_1s_w", label="rubber car", ref=False, bg="push", ns=60, base="orbit", H=1.0, tube=0.99,
                          track="practised ETHZMobil lap"),
    # (2026-11-05, user: "color shading ... should show magnitude of difference between the final version and the previous versions"):
    # the same flow, shaded by |flow - the FINAL policy's flow| (lateral, tolerance units of 0.1 m): pogo_flow's convention with the
    # converged policy as the reference (the last row is white by construction)
    "ethz_vs_final": dict(plan="ethzmobil43_b0_free", run="ETHZ_rubber_freePPON_s0_ss9001", car="ax:mu=1.667", var="lat",
                          stem="car_flow_ethz_vs_final", data="car_flow_ethz_orbit_1s", label="rubber car", ref=False, bg="final",
                          ns=60, base="orbit", H=1.0, track="practised ETHZMobil lap"),
    # (2026-11-05, user: "isn't it supposed to be against sim reference"): pogo_flow's convention -- the reference row is the nominal car
    # (mu 0.6, the friction the uncapped ETHZMobil plan was solved at; it drives the plan: 14.2 cm rms, pace 0.98, never out of the tube)
    # under pi_0; every row shaded by |its lateral flow - the reference's| (tol 0.1 m); the same starting states (the rubber car's
    # zero-shot orbit) in every row, so the rows differ only by the car and the policy
    # (2026-11-05, user: "show how the policy has changed as a residual from the initial zero-shot policy"): the shading is the policy's
    # own change, pi_k - pi_0 of the steering command (deg) at each starting state (policy_resid.py); arrows and the zero-push line as before
    "ethz_resid": dict(plan="ethzmobil43_b0_free", run="ETHZ_rubber_freePPON_s0_ss9001", car="ax:mu=1.667", var="lat",
                       stem="car_flow_ethz_resid", data="car_flow_ethz_orbit_1s_w", resid="car_flow_ethz_orbit_1s_w_resid", tube=0.99, label="rubber car",
                       ref=False, bg="resid", ns=60, base="orbit", H=1.0, track="practised ETHZMobil lap"),                                                         # (2026-11-05: all four perturbed starts shown, user)
    # (2026-11-05, for the stage videos): the held-out peanut, plastic nominal car (the same run and car as "plastic" / "speed"), the
    # 1.0 s horizon on its own zero-shot orbit, and the policy-residual shading (policy_resid.py peanut_orbit_1s)
    "peanut_orbit_1s": dict(plan="mocap_peanutM_b0_free_a33", run="PLSL_freePPON_s0", car="n04_real_nom", var="lat",
                            stem="car_flow_peanut_orbit_1s", label="plastic car", ref=False, bg="push", ns=40, base="orbit", H=1.0,
                            track="held-out peanut lap"),
    "peanut_resid": dict(plan="mocap_peanutM_b0_free_a33", run="PLSL_freePPON_s0", car="n04_real_nom", var="lat",
                         stem="car_flow_peanut_resid", data="car_flow_peanut_orbit_1s", resid="car_flow_peanut_orbit_1s_resid",
                         label="plastic car", ref=False, bg="resid", ns=40, base="orbit", H=1.0, track="held-out peanut lap"),
    # (2026-11-06, user: "show the rubber peanut"): the rubber car practising the peanut at x1.5 (PFAST2, median run s0: -23.3 %,
    # e_y 34.4 -> 26.4 cm), the 1.0 s horizon on its own zero-shot orbit, push shading (data) and the policy-residual shading (video)
    "peax150_orbit_1s": dict(plan="mocap_peanutM_b0_free_x150", run="PFAST2_rubber_peax150_freePPON_s0_ss9001", car="ax:mu=1.667",
                             var="lat", stem="car_flow_peax150_orbit_1s", label="rubber car", ref=False, bg="push", ns=40, base="orbit",
                             H=1.0, track="practised peanut lap (x1.5)"),
    "peax150_resid": dict(plan="mocap_peanutM_b0_free_x150", run="PFAST2_rubber_peax150_freePPON_s0_ss9001", car="ax:mu=1.667",
                          var="lat", stem="car_flow_peax150_resid", data="car_flow_peax150_orbit_1s",
                          resid="car_flow_peax150_orbit_1s_resid", label="rubber car", ref=False, bg="resid", ns=40, base="orbit",
                          H=1.0, track="practised peanut lap (x1.5)"),
    # (2026-11-06, user: the plastic video's cost is "insanely small" -- the nominal plastic car on the held-out peanut tracks to 2 cm and
    # its held-out gain is pace only): the plastic car where lateral tracking improves from a large start, the practised ETHZMobil.
    # Pick rule, fixed before rendering: the median run of the 9 (ETHZ_plastic_freePPON_s2_ss9003, -18.2 %); in it the car with the
    # largest zero-shot cost among those without failed laps (ax:mu=0.4667: score 4.14 -> 3.25, -21 %; e_y 41.4 -> 32.5 cm)
    "ethzp_orbit_1s": dict(plan="ethzmobil43_b0_free_a33", run="ETHZ_plastic_freePPON_s2_ss9003", car="ax:mu=0.4667", var="lat",
                           stem="car_flow_ethzp_orbit_1s", label="plastic car (mu 0.4667)", ref=False, bg="push", ns=60, base="orbit",
                           H=1.0, tube=0.99, track="practised ETHZMobil lap"),
    "ethzp_resid": dict(plan="ethzmobil43_b0_free_a33", run="ETHZ_plastic_freePPON_s2_ss9003", car="ax:mu=0.4667", var="lat",
                        stem="car_flow_ethzp_resid", data="car_flow_ethzp_orbit_1s", resid="car_flow_ethzp_orbit_1s_resid", tube=0.99,
                        label="plastic car (mu 0.4667)", ref=False, bg="resid", ns=60, base="orbit", H=1.0, track="practised ETHZMobil lap"),
    "ethz_simref": dict(plan="ethzmobil43_b0_free", run="ETHZ_rubber_freePPON_s0_ss9001", car="ax:mu=1.667", var="lat",
                        stem="car_flow_ethz_simref", label="rubber car", ref=True, line=True, ns=60, base="orbit", H=1.0,
                        track="practised ETHZMobil lap"),
}
V = VARIANTS[sys.argv[1] if len(sys.argv) > 1 else "plastic"]
DATA = os.path.join(HERE, f"{V.get('data', V['stem'])}_data.npz")
PLAN, RUN, CAR = V["plan"], D + "/runs/screen/" + V["run"], V["car"]
REF_CAR = "real_nom"
TOL, DT = 0.10, 0.025
# the tube: the mocap room's 0.5 m safety envelope; ETHZMobil (sim only) uses its own track edges (2026-11-06, user: "aren't the margins
# typically a lot larger in the ethzmobil track? probably about 2x"): BayesRace width 0.46 m x 4.3 = 1.98 m -> +-0.99 m
TUBE = V.get("tube", 0.5)
TW = "tube" if TUBE == 0.5 else "track"                              # ETHZMobil: the real track edges
H = V.get("H", 0.4)                                                  # the arrow's horizon (s)
NS, NE = V.get("ns", 40), 11
if V["var"] == "lat":
    GRID = np.linspace(-0.9 * TUBE, 0.9 * TUBE, NE)                  # lateral offset e_y0 (m): 90 % of the tube
    STARTS = [(0.15, 0.30), (0.40, -0.30), (0.65, 0.15), (0.90, -0.15)]   # (lap fraction, e_y0)
else:
    GRID = np.linspace(*V.get("grid", (-0.60, 0.30)), NE)            # speed error dV0 (m/s)
    STARTS = V.get("starts", [(0.15, -0.45), (0.40, 0.20), (0.65, -0.25), (0.90, 0.0)])    # (lap fraction, dV0)


def load_policy(path):
    """hw_stage.load_policy (hw_stage parses argv on import)"""
    d = np.load(path)
    n = len([k for k in d.files if k.startswith("W")])
    return [(np.asarray(d[f"W{i}"]), np.asarray(d[f"b{i}"])) for i in range(n)]


def compute():
    sys.path.insert(0, D)
    os.chdir(D)
    import bank
    import real_car as rc
    import sim_jax_np as sn
    pl = bank.load_bank([PLAN])[0]
    P = sn.PlanNP(pl)
    L = float(pl["length"])
    pols = [load_policy(f"{RUN}/its/it{k}_{CAR}_policy.npz") for k in range(4)]
    panels = ([(REF_CAR, pols[0])] if V.get("ref", True) else []) + [(CAR, p) for p in pols]

    orbit = None
    if V.get("base") == "orbit":                                     # the zero-shot car's own lap (noise-free), by plan index
        r0 = rc.run(CAR, pl, pols[0], laps=1.05, s0=0.0, pert=np.zeros(5), seed=0, noise=False)
        orbit = {int(i): (np.asarray(x, float), float(e[0])) for i, x, e in zip(r0["idx"], r0["x_true"], r0["err_true"])}
        oi = np.array(sorted(orbit))

    def start(s, e):
        i = int(round(s / P.ds)) % P.M
        x = pl["x"][i].copy()
        if orbit is not None:                                        # the orbit's state nearest this arc length, shifted to offset e
            j = int(oi[np.argmin(np.abs((oi - i + P.M // 2) % P.M - P.M // 2))])
            xo, eo = orbit[j]
            th = pl["x"][j][2]
            x = xo.copy()
            x[0] += -math.sin(th) * (e - eo)
            x[1] += math.cos(th) * (e - eo)
            return x, j
        if V["var"] == "lat":                                        # sideways shift, everything else on the plan
            x[0] += -math.sin(x[2]) * e
            x[1] += math.cos(x[2]) * e
        else:                                                        # on the path, speed V + e (vx, vy, r, wheel scaled)
            f = (math.hypot(x[3], x[4]) + e) / math.hypot(x[3], x[4])
            x[3:7] *= f
        return x, i

    ss = np.linspace(0, L, NS, endpoint=False)
    es = GRID
    col = 0 if V["var"] == "lat" else 5                              # err_true: 0 = e_y, 5 = speed error dV
    T = int(round(H / DT))
    jobs, keys = [], []
    for k, (car, pol) in enumerate(panels):
        for i, s in enumerate(ss):
            for j, e in enumerate(es):
                x, i0 = start(s, e)
                jobs.append(dict(car_name=car, plan=pl, policy=pol, x_start=x, idx_start=i0, T_max=T + 1, noise=False))
                keys.append(("f", k, i, j))
        for m, (fr, e) in enumerate(STARTS):
            x, i0 = start(fr * L, e)
            jobs.append(dict(car_name=car, plan=pl, policy=pol, x_start=x, idx_start=i0,
                             T_max=int(round(float(pl["lap_time"]) / DT)), noise=False))
            keys.append(("p", k, m, 0))
    R = rc.run_many(jobs, 16)
    K = len(panels)
    dS, dE = np.full((K, NS, NE), np.nan), np.full((K, NS, NE), np.nan)
    orb = [[None] * len(STARTS) for _ in range(K)]
    for (kind, k, i, j), r, jb in zip(keys, R, jobs):
        ey = r["err_true"][:, col]
        if kind == "f":
            n = min(T, len(ey) - 1)
            dS[k, i, j] = ((r["idx"][n] - r["idx"][0] + P.M // 2) % P.M - P.M // 2) * P.ds
            dE[k, i, j] = ey[n] - ey[0]
        else:
            sabs = (r["idx"][0] * P.ds + np.cumsum(np.r_[0, ((np.diff(r["idx"]) + P.M // 2) % P.M - P.M // 2) * P.ds]))
            orb[k][i] = np.stack([sabs, ey], 1)
    O = np.full((K, len(STARTS), max(len(o) for row in orb for o in row), 2), np.nan)
    for k in range(K):
        for m in range(len(STARTS)):
            O[k, m, :len(orb[k][m])] = orb[k][m]
    vbar = float(np.mean(np.hypot(pl["x"][:, 3], pl["x"][:, 4])))
    np.savez(DATA, ss=ss, es=es, dS=dS, dE=dE, O=O, L=L, xy=np.asarray(pl["x"])[:, :3], vbar=vbar)


def main():
    if not os.path.exists(DATA):
        compute()
    import matplotlib
    import matplotlib.colors
    import matplotlib.patches
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    sys.path[:0] = ["/home/henry/ilc_ws/log/dilc/flows"]
    import portrait_style as ps

    for f in glob.glob("/usr/share/fonts/truetype/msttcorefonts/[Tt]imes*.ttf"):
        font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family": ["Times New Roman", "TeX Gyre Termes", "Nimbus Roman", "STIXGeneral", "DejaVu Sans"],
                         "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42})
    Dd = np.load(DATA)
    ss, es, dS, dE, O, L = Dd["ss"], Dd["es"], Dd["dS"], Dd["dE"], Dd["O"], float(Dd["L"])
    KEEP = [m for m in range(O.shape[1]) if m not in V.get("drop_starts", ())]
    lat = V["var"] == "lat"
    tol = TOL if lat else 0.1 * float(Dd["vbar"])                    # 0.10 m, or 10 % of the plan's mean speed (pace_ok)
    YM = 1.2 * TUBE                                                  # the plot's lateral range: the tube plus a margin
    ylim = (-YM, YM) if lat else (es[0] - 0.08, es[-1] + 0.08)
    SS, EE = np.meshgrid(ss, es, indexing="ij")
    # one length factor for every row; in the speed panels the vertical axis is m/s, plotted on the same scale as m
    k_len = 0.85 * (ss[1] - ss[0]) * 2 / np.nanquantile(np.hypot(dS[0], dE[0]), 0.9)
    ref = V.get("ref", True)
    drift = V.get("bg") == "drift"
    laps_per_it = 1 if "ethz" in V["plan"] else 2                    # the ETHZMobil stage's one-lap trials (ETHZ_RULES.md)
    titles = ([r"nominal car ($\mu$ 0.6), $\pi_0$ (reference)"] if ref else []) + \
             [V["label"] + rf", $\pi_{k}$" + (" (zero-shot)" if k == 0 else f" (after {laps_per_it * k} lap{'s' if laps_per_it * k > 1 else ''})")
              for k in range(4)]
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#5598e7", "#f0efec", "#ec8a89", "#c23a39"])
    DNORM = TwoSlopeNorm(vcenter=0.0, vmin=-12.0, vmax=12.0)                          # cm per 0.4 s, toward (-) / away from (+) the plan
    outward = lambda k: 100 * np.where(EE == 0, np.abs(dE[k]), np.sign(EE) * dE[k])

    def settle_line(dEk):                                            # per arc length: the offset where the push crosses zero downward
        out = np.full(len(ss), np.nan)
        for i in range(len(ss)):
            y = dEk[i]
            for j in range(len(es) - 1):
                if y[j] > 0 >= y[j + 1]:
                    out[i] = es[j] + (es[j + 1] - es[j]) * y[j] / (y[j] - y[j + 1])
                    break
            else:
                out[i] = es[-1] if y[-1] > 0 else (es[0] if y[0] < 0 else np.nan)
        return out
    push = V.get("bg") in ("push", "final", "resid")
    resid = V.get("bg") == "resid"
    vsfinal = V.get("bg") == "final"
    settle = []
    fig, axs = plt.subplots(len(titles), 1, figsize=(7.16, 5.6), sharex=True)
    for k, (ax, ttl) in enumerate(zip(axs, titles)):
        dev = np.abs(dE[k] - dE[0]) / tol
        if V.get("bg") == "resid":                                   # the policy's steering change vs pi_0 (deg), and the zero-push line
            Rz = np.load(os.path.join(HERE, f"{V['resid']}.npz"))["U"]
            dst = np.degrees(Rz[k, :, :, 0] - Rz[0, :, :, 0])
            im = ax.pcolormesh(SS, EE, dst, cmap=DIV, norm=TwoSlopeNorm(vcenter=0.0, vmin=-6.0, vmax=6.0), shading="gouraud")
            ax.plot(ss, settle_line(dE[k]), color=ps.INK, lw=1.1, zorder=5)
        elif V.get("bg") == "final":                                   # |flow - the final policy's flow| (tolerance units), and the zero line
            devf = np.abs(dE[k] - dE[-1]) / TOL                      # own scale: these differences stay below ~1 tol
            im = ax.pcolormesh(SS, EE, devf, cmap=ps.CMAP, norm=matplotlib.colors.Normalize(0.0, 1.0), shading="gouraud")
            ax.plot(ss, settle_line(dE[k]), color=ps.INK, lw=1.1, zorder=5)
        elif V.get("bg") == "push":                                    # the signed sideways push (cm per 0.4 s) and its zero line
            im = ax.pcolormesh(SS, EE, 100 * dE[k], cmap=DIV, norm=DNORM, shading="gouraud")
            sl = settle_line(dE[k])
            ax.plot(ss, sl, color=ps.INK, lw=1.1, zorder=5)
        elif drift:
            im = ax.pcolormesh(SS, EE, outward(k), cmap=DIV, norm=DNORM, shading="gouraud")
        else:
            im = ps.background(ax, SS, EE, dev)
            if V.get("line"):                                        # the line the flow pulls the car onto
                ax.plot(ss, settle_line(dE[k]), color=ps.INK, lw=1.1, zorder=5)
        if lat:
            ax.axhspan(TUBE, YM, color="#e9e7e2", zorder=1)
            ax.axhspan(-YM, -TUBE, color="#e9e7e2", zorder=1)
            for y in (-TUBE, TUBE):
                ax.axhline(y, color="#777777", lw=0.6, ls="--", zorder=2)
        ax.axhline(0, color=ps.INK, lw=0.5, alpha=0.5, zorder=2)
        sub = (slice(None, None, 2), slice(None, None, 2))
        q = ps.arrows(ax, SS[sub], EE[sub], k_len * dS[k][sub], k_len * dE[k][sub])
        q.width, q.headwidth, q.headlength = 0.0022, 3.6, 4.0
        q.set_alpha(0.8)
        tail = []
        for m in KEEP:
            o = O[k, m]
            o = o[np.isfinite(o[:, 0])]
            out = np.nonzero(np.abs(o[:, 1]) > TUBE)[0] if lat else []
            if len(out):                                             # leaves the tube: on the car, the kill switch
                o = o[:out[0] + 1]
                ax.plot(np.mod(o[-1, 0], L), np.clip(o[-1, 1], -0.97 * YM, 0.97 * YM), "x", color=ps.RUN, ms=4, mew=1.0, zorder=7)
            tail.append(o[int(round(1.0 / DT)):, 1])                 # after 1 s: where the starts settle
            p = o[::8]                                               # every 0.2 s
            ax.plot(np.mod(p[:, 0], L), p[:, 1], "o", color=ps.RUN, mec="white", mew=0.3, ms=2.2, zorder=6)
        settle.append(float(np.mean(np.concatenate(tail))))
        if not lat:                                                  # the level the speed settles at
            ax.axhline(settle[-1], color=ps.RUN, lw=0.8, ls=":", zorder=5)
            if "grid" in V and k > 0:                                # zoomed: the reference's level, and the gap closing
                ax.axhline(settle[0], color=ps.INK, lw=0.6, ls=":", alpha=0.6, zorder=5)
                ax.fill_between([0, L], settle[-1], 0, color=ps.RUN, alpha=0.22, lw=0, zorder=3)
        ax.set_xlim(0, L)
        ax.set_ylim(*ylim)
        ax.set_yticks([-TUBE, 0, TUBE] if lat else ([-0.3, -0.2, -0.1, 0, 0.1] if "grid" in V else [-0.6, -0.3, 0, 0.3]))
        ax.tick_params(labelsize=6, length=2, width=0.5, pad=1.5)
        ax.set_title(ttl, fontsize=7.5, loc="left", pad=2)
        extra = "" if lat else f", settles at {settle[-1]:+.2f} m/s"
        if not lat and "grid" in V and k > 0:
            extra += f" (gap to the plan {100 * settle[-1] / settle[1]:.0f} % of zero-shot's)"
        if push:
            nout = sum(np.any(np.abs(O[k, m][np.isfinite(O[k, m][:, 1]), 1]) > TUBE) for m in KEEP)
            tr = O[k][KEEP][:, int(round(1.0 / DT)):, 1]
            txt = (f"pulled onto a line {100 * np.nanmean(np.abs(settle_line(dE[k]))):.1f} cm off the plan on average; perturbed laps "
                   f"{100 * np.sqrt(np.nanmean(tr ** 2)):.1f} cm rms, {nout} of {len(KEEP)} leave the {TW}")
            if resid:
                Rz = np.load(os.path.join(HERE, f"{V['resid']}.npz"))["U"]
                tr = O[k][KEEP][:, int(round(1.0 / DT)):, 1]
                txt = (f"steering change {np.degrees(np.abs(Rz[k, :, :, 0] - Rz[0, :, :, 0])).mean():.1f} deg mean "
                       f"(max {np.degrees(np.abs(Rz[k, :, :, 0] - Rz[0, :, :, 0])).max():.0f}); line {100 * np.nanmean(np.abs(settle_line(dE[k]))):.1f} cm "
                       f"off the plan; laps {100 * np.sqrt(np.nanmean(tr ** 2)):.1f} cm rms, {nout}/{len(KEEP)} leave the {TW}")
            if vsfinal:
                txt = f"difference from the final flow {np.nanmean(np.abs(dE[k] - dE[-1])) / TOL:.2f} tol (mean); " + txt
        elif drift:
            nout = sum(np.any(np.abs(O[k, m][np.isfinite(O[k, m][:, 1]), 1]) > TUBE) for m in KEEP)
            txt = f"flow away from the plan at {100 * np.mean(outward(k) > 0):.0f} % of points; {nout} of {O.shape[1]} perturbed starts leave the {TW}"
        else:
            txt = f"mean error {np.nanmean(dev):.2f} tol{extra}"
        ax.text(1.0, 1.02, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="#555555")
        for s_ in ax.spines.values():
            s_.set_linewidth(0.5)
            s_.set_color("#777777")
    axs[-1].set_xlabel(f"arc length along the {V.get('track', 'held-out lap')} (m)", fontsize=7, labelpad=1.5)
    fig.supylabel("lateral offset from the plan (m)" if lat else "speed minus the plan's speed (m/s)", fontsize=7, x=0.012)
    hs, ls_ = [], []
    if lat:
        hs += [Line2D([], [], color="#777777", lw=0.6, ls="--")]
        ls_ += [(f"{TUBE:.2f} m tube (safety envelope)" if TUBE == 0.5 else f"track edges (+-{TUBE:.2f} m)")]
    hs += [Line2D([], [], ls="none", marker="o", color=ps.RUN, mec="white", mew=0.3, ms=3)]
    ls_ += [("4 laterally perturbed" if lat else "4 speed-perturbed") + " starts, every 0.2 s"]
    if lat:
        hs += [Line2D([], [], ls="none", marker="x", color=ps.RUN, mew=1.0, ms=4)]
        ls_ += [f"leaves the {TW}"]
    else:
        hs += [Line2D([], [], color=ps.RUN, lw=0.8, ls=":")]
        ls_ += ["where they settle (after 1 s)"]
        if "grid" in V:
            hs += [Line2D([], [], color=ps.INK, lw=0.6, ls=":", alpha=0.6), matplotlib.patches.Patch(color=ps.RUN, alpha=0.22)]
            ls_ += ["reference's level", "gap to the plan's speed"]
    if push:
        hs += [Line2D([], [], color=ps.INK, lw=1.1)]
        ls_ += ["where the push is zero"]
    hs += [Line2D([], [], color=ps.INK, lw=0.8, marker=">", ms=2.5, alpha=0.8)]
    ls_ += [f"where the car is {H:.1f} s later"]
    fig.legend(hs, ls_, loc="lower center", bbox_to_anchor=(0.47, -0.005), ncol=len(hs), frameon=False, fontsize=6, handlelength=1.4,
               columnspacing=1.2, handletextpad=0.4)
    fig.subplots_adjust(left=0.065, right=0.915, top=0.965, bottom=0.10, hspace=0.42)
    p0, p1 = axs[0].get_position(), axs[-1].get_position()
    cax = fig.add_axes([p0.x1 + 0.012, p1.y0, 0.009, p0.y1 - p1.y0])
    cb = fig.colorbar(im, cax=cax, extend="both" if (resid or drift or (push and not vsfinal)) else "max")
    if resid:
        cb.set_label("steering change vs the zero-shot policy (deg; + = more to the left)", fontsize=6, labelpad=2)
    elif vsfinal:
        cb.set_label("difference from the final policy's flow (tol = 0.1 m)", fontsize=6, labelpad=2)
    elif push:
        cb.set_label(f"sideways push in {H:.1f} s (cm; + toward + offset)", fontsize=6, labelpad=2)
    elif drift:
        cb.set_label("lateral drift in 0.4 s (cm): back toward the plan  <  0  <  away from it", fontsize=6, labelpad=2)
    else:
        cb.set_label(("lateral" if lat else "speed") + f" flow deviation from the reference (tol = {tol:.2f} " + ("m)" if lat else "m/s)"),
                     fontsize=6, labelpad=2)
    cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4)
    cb.outline.set_linewidth(0.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{V['stem']}.{ext}"), dpi=300)
    for k, t in enumerate(titles):                                   # the numbers the caption quotes
        dev = np.abs(dE[k] - dE[0]) / tol
        out = np.mean([np.any(np.abs(O[k, m][np.isfinite(O[k, m][:, 1]), 1]) > TUBE) for m in KEEP]) if lat else 0.0
        if push:
            tr = O[k][KEEP][:, int(round(1.0 / DT)):, 1]
            print(f"{t:45s} settle line |offset| {100 * np.nanmean(np.abs(settle_line(dE[k]))):.1f} cm, perturbed laps rms "
                  f"{100 * np.sqrt(np.nanmean(tr ** 2)):.1f} cm, leave the tube {out:.2f}")
        elif drift:
            print(f"{t:45s} away from the plan at {100 * np.mean(outward(k) > 0):.0f} % of points, mean drift {np.nanmean(outward(k)):+.1f} cm, "
                  f"perturbed starts leave the tube {out:.2f}, settle at {settle[k]:+.3f}")
        else:
            print(f"{t:45s} mean dev {np.nanmean(dev):.2f} tol, perturbed starts settle at {settle[k]:+.3f}, leave the tube {out:.2f}")
    print("->", OUT, V["stem"])


if __name__ == "__main__":
    main()
