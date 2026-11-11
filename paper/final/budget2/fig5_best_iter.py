"""fig5_best_iter.py IN.json OUT.json (2026-11-10, user: "choose the best one it converges to and stop there (may be rebounds as is common
in ILC)"): the SELECTION RULE written before the extended runs' results -- every DistILC-stage curve (arms containing "DistILC") stops at
the point with the lowest mean score on the NOMINAL panel; the same point count is kept on the perturbed panel. The last point it reached is
printed (and kept in "last" for the caption). Baselines (PPO on robot, FADA, RMA zero-shot) are unchanged."""
import json
import sys

d = json.load(open(sys.argv[1]))
d["last"] = {}
for arm, per in d["real"].items():
    if "DistILC" not in arm or not per.get("nominal"):
        continue
    nom = per["nominal"]
    k = min(range(len(nom)), key=lambda i: nom[i][1])
    d["last"][arm] = {c: per[c][-1] for c in per}
    for c in per:
        per[c] = per[c][:k + 1]
    print(f"{arm}: best at x = {nom[k][0]:.0f} ({nom[k][1]:.2f}); last x = {d['last'][arm]['nominal'][0]:.0f} ({d['last'][arm]['nominal'][1]:.2f})")
json.dump(d, open(sys.argv[2], "w"), indent=1)
