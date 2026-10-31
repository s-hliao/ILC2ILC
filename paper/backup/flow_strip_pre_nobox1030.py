"""Pogo: the hardware stage re-converging the closed-loop flow, one thin row read left to right (2026-10-30, the paper
version of presentation_2026-10-26/scripts/flow_iters_pogo.py; user: "it needs to be thin, along a line going left to
right", no orange lines, the individual points instead).
  panels      the real offset stick under pi_0 (corrupted) .. pi_4 (ours, nominal sim, servo leg: runs/v3/nom_s0_curve),
              then the sim under pi_0 (the reference the stage pulls toward)
  background  |V - V_sim,pi0| in tolerance units (2 cm apex, 0.1 m/s), white = the sim's field
  arrows      the apex-to-apex return map (next apex - apex) at the goal (0.8 m/s, 0.62 m)
  points      the apexes of 4 perturbed drops x 20 hops (where the old orbit lines turned); star = goal, box = tolerance
The fields are computed once into flow_strip_data.npz (delete it to recompute). Full text width (7.16 in), Times as the
other paper/final figures. Writes pogo_flow.{pdf,png} to paper/final/.
    POGO_LEG=servo python paper/final/pogo_flow/flow_strip.py      (the pogo env: needs log/dilc/flows/pogo/flows_pogo)
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)                                          # paper/final
DATA = os.path.join(HERE, "flow_strip_data.npz")
FL = "/home/henry/ilc_ws/log/dilc/flows"
SERVO = "/home/henry/ilc_ws/log/pogo_servo"
ROBOT, ARM = "real_offset", "nom_s0"
GOAL = np.array((0.8, 0.62))                                         # (apex speed m/s, apex height m)
N, K = 13, 4


def compute():
    sys.path[:0] = [FL, f"{FL}/pogo"]
    import flows_pogo as fp
    assert fp.C.LEG == "servo"
    R = f"{SERVO}/runs/v3/{ARM}_curve"
    pols = [f"{SERVO}/runs/nom_s0/policy_pick.npz"] + [f"{R}/it{k}/{ROBOT}_policy.npz" for k in (1, 2, 3)] + [f"{R}/{ROBOT}_policy.npz"]
    g = GOAL
    hs = np.linspace(g[1] - 0.08, g[1] + 0.08, N)
    vs = np.linspace(g[0] - 0.5, g[0] + 0.5, N)
    VV, HH = np.meshgrid(vs, hs)
    flat = np.stack([HH.ravel(), VV.ravel()], -1)
    Gl = np.repeat(g[None], len(flat), 0)
    X0, Gr = fp.drop_apexes(g[None])

    def field(p, system):
        return fp.fly_hop(p, flat[:, 0], flat[:, 1], Gl, system)[0] - flat, fp.orbit(p, X0, Gr, system, hops=fp.N_HOPS)

    Vs, Os = field(pols[0], "sim")
    V, O = zip(*[field(p, ROBOT) for p in pols])
    np.savez(DATA, vs=vs, hs=hs, Vs=Vs, Os=Os, V=np.array(V), O=np.array(O), tol=np.array(fp.TOL), goal=g)


def main():
    if not os.path.exists(DATA):
        compute()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    sys.path[:0] = [FL]
    import portrait_style as ps

    for f in glob.glob("/usr/share/fonts/truetype/msttcorefonts/[Tt]imes*.ttf"):
        font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family": ["Times New Roman", "TeX Gyre Termes", "Nimbus Roman", "STIXGeneral", "DejaVu Sans"],
                         "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42})
    D = np.load(DATA)
    vs, hs, Vs, tol, g = D["vs"], D["hs"], D["Vs"], D["tol"], D["goal"]
    VV, HH = np.meshgrid(vs, hs)
    xs, ys = vs[-1] - vs[0], hs[-1] - hs[0]
    rl = ps.ref_len(Vs[:, 1], Vs[:, 0], xs, ys)
    sub = slice(None, None, 2)                                       # arrows on every other grid node (7 x 7)
    panels = [(D["V"][k], D["O"][k], rf"$\pi_{k}$" + (" (real, before)" if k == 0 else "")) for k in range(K + 1)]
    panels.append((Vs, D["Os"], r"sim, $\pi_0$ (reference)"))

    fig, axs = plt.subplots(1, len(panels), figsize=(7.16, 1.52), sharey=True)
    for i, (ax, (V, O, ttl)) in enumerate(zip(axs, panels)):
        dV = np.linalg.norm((V - Vs) / tol, axis=1)
        im = ps.background(ax, VV, HH, dV.reshape(HH.shape))
        Vx, Vh = V[:, 1].reshape(HH.shape), V[:, 0].reshape(HH.shape)
        q = ps.arrows_scaled(ax, VV[sub, sub], HH[sub, sub], Vx[sub, sub], Vh[sub, sub], xs, ys, (N - 1) // 2, ref=rl)
        q.width, q.headwidth, q.headlength = 0.009, 3.2, 3.6
        q.set_alpha(0.75)
        pts = O.reshape(-1, 2)                                       # every drop's every apex (h, v)
        ax.plot(pts[:, 1], pts[:, 0], "o", color=ps.RUN, mec="white", mew=0.3, ms=2.4, zorder=6, clip_on=True)
        ax.plot(g[0], g[1], "*", color="white", mec=ps.INK, mew=0.6, ms=6.5, zorder=7)
        ax.add_patch(plt.Rectangle((g[0] - tol[1], g[1] - tol[0]), 2 * tol[1], 2 * tol[0], fill=False, ec=ps.INK, lw=0.6,
                                   ls=(0, (2.5, 1.5)), zorder=6))
        ax.set_xlim(vs[0], vs[-1])
        ax.set_ylim(hs[0], hs[-1])
        ax.set_title(ttl, fontsize=7.5, pad=2.5)
        dev = "" if i == len(panels) - 1 else f"{np.nanmean(dV):.1f} tol"
        ax.text(0.97, 0.04, dev, transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=ps.INK,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8), zorder=8)
        ax.set_xticks([0.4, 0.8, 1.2])
        ax.tick_params(labelsize=6, length=2, width=0.5, pad=1.5)
        for s in ax.spines.values():
            s.set_linewidth(0.5)
            s.set_color("#777777")
    axs[0].set_yticks([0.56, 0.62, 0.68])
    axs[0].set_ylabel("apex height (m)", fontsize=7, labelpad=1.5)
    fig.supxlabel("apex speed (m/s)", fontsize=7, y=0.015)
    fig.subplots_adjust(left=0.062, right=0.925, top=0.86, bottom=0.2, wspace=0.2)
    for a, b in zip(axs[:-1], axs[1:]):                              # left-to-right reading: an arrow between neighbours
        p, n = a.get_position(), b.get_position()
        fig.text((p.x1 + n.x0) / 2, p.y0 + p.height / 2, r"$\rightarrow$", ha="center", va="center", fontsize=7, color=ps.INK)
    p = axs[-1].get_position()
    cax = fig.add_axes([p.x1 + 0.012, p.y0, 0.009, p.height])
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label("deviation from sim (tol)", fontsize=6, labelpad=2)
    cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4)
    cb.outline.set_linewidth(0.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"pogo_flow.{ext}"), dpi=300)
    print("->", OUT)


if __name__ == "__main__":
    main()
