"""stick.py OUT [OUT...]: how well each landing sticks, from the true foot contacts
(rec_true_contacts, else rec_contacts). Per trial, once all four feet have been loaded
after the flight: `unstick_ms`, the time any foot then reads unloaded (< 1 N) -- a foot
lifting, hopping or sliding off; `rear_ms`, the same for the rear feet alone. Fallen
trials are counted apart (their tail is the fall). Per run it prints the falls pattern,
the falls, and over the last 5 trials the worst unstick_ms / rear_ms and the landing error."""
import glob, json, os, sys
import numpy as np


def trial(p):
    d = np.load(p, allow_pickle=True)
    k = "rec_true_contacts" if "rec_true_contacts" in d.files else "rec_contacts"
    t, con = d["rec_t"], d[k]
    cfg = json.loads(str(d["config"])); Nc = sum(cfg["phases"][:2]); dt = cfg["dt"]
    res = json.loads(str(d["result"]))
    air = np.where((t > Nc * dt + 0.03) & (con.max(1) < 1))[0]
    un = rear = np.nan
    if air.size:
        after = np.arange(air[0], len(t))
        alld = after[con[after].min(1) >= 1]
        if alld.size:
            tail = np.arange(alld[0], len(t)); ddt = np.diff(t[tail], append=t[tail][-1])
            un = 1000 * ddt[con[tail].min(1) < 1].sum()
            rear = 1000 * ddt[con[tail][:, 2:].min(1) < 1].sum()
    return dict(fell=bool(res.get("fell")), un=un, rear=rear)


for out in sys.argv[1:]:
    rows = []
    for rd in sorted(glob.glob(f"{out}/*/")):
        files = sorted(glob.glob(f"{rd}trial_*.npz"))
        if not files or not os.path.exists(rd[:-1] + ".json"):
            continue
        tr = [trial(p) for p in files]
        h = json.load(open(rd[:-1] + ".json"))["history"]
        pat = "".join("F" if r["fell"] else "." for r in tr)
        t5 = [r for r in tr[-5:] if not r["fell"]]
        un = max((r["un"] for r in t5), default=np.nan); rr = max((r["rear"] for r in t5), default=np.nan)
        l = h[-1]
        name = os.path.basename(rd[:-1]); s, _, v = name.partition("__")
        rows.append((v, s, f"{pat:20s} F={pat.count('F'):2d} last5: F={pat[-5:].count('F')} "
                           f"unstick<={un:4.0f}ms rear<={rr:4.0f}ms | ILC last {l['pos_err']*100:4.1f}cm "
                           f"{np.degrees(l['theta_err']):4.1f}°"))
    for r in sorted(rows):
        print(f"{r[0]:10s} {r[1]:4s} {r[2]}")
