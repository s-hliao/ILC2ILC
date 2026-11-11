"""pogo_best_iter.py IN.json OUT.json (2026-11-11, user: "yes, best iteration rule on pogo as well"): the best-iteration rule of
fig5_best_iter.py / car_fig2_x8_data.py on the pogo's real-rollout panels -- every DistIL-C_hw stage curve (arms containing "DistILC")
stops, PER TIER PANEL (A / B / C; as stated to the user 2026-11-10: "on tiers A/B it would cut RMA + DistIL-C_hw back to its zero-shot
point"), at its lowest mean normalized error; the last point reached is kept in D["last"][arm][tier] for the caption. Baselines unchanged."""
import json
import sys

D = json.load(open(sys.argv[1]))
D["last"] = {}
for arm, per in D["real"].items():
    if "DistILC" not in arm:
        continue
    for t, pts in per.items():
        if not pts:
            continue
        k = min(range(len(pts)), key=lambda i: pts[i][1])
        D["last"].setdefault(arm, {})[t] = pts[-1]
        per[t] = pts[:k + 1]
        print(f"{arm:20s} {t}: best x = {pts[k][0]:.0f} ({pts[k][1]:.2f}); last x = {pts[-1][0]:.0f} ({pts[-1][1]:.2f})")
json.dump(D, open(sys.argv[2], "w"), indent=1)
