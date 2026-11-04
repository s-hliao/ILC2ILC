#!/usr/bin/env python3
"""gpusched.py JOBS.json (2026-10-30, matcha; user: "figure out how to schedule them. pogo goes first"): a GPU-memory-aware priority
queue. Each job: name, prio (lower first), mem (MiB it needs), done (file whose existence means finished), needs (files that must exist
first), cmd (shell, {gpu} = the GPU index chosen). A job starts only on a GPU with mem + MARGIN free (nvidia-smi), highest priority first,
one launch per STAGGER seconds; a job whose name is already in a running process's command line is left alone. JOBS.json is re-read
every loop, so jobs can be added while it runs. Log: JOBS.log."""
import json, os, subprocess, sys, time
F = sys.argv[1]; LOG = F.replace(".json", ".log"); MARGIN, STAGGER, SETTLE, TMAX = 600, 30, 90, 86
launched = {}
def log(m):
    open(LOG, "a").write(f"{time.strftime('%H:%M:%S')} {m}\n")
def free():
    o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    # (2026-10-30) a GPU at >= TMAX C takes no new job (the check-in limit is 88 C; all four ran at 84-86 C)
    return {int(a): (int(b) if int(t) < TMAX else 0) for a, b, t in (l.split(",") for l in o.strip().splitlines())}
def running(name, match=None):
    # (2026-10-30) a job may give "match", a string its live processes carry: bash -c execs a single command, dropping a "# name" tag
    o = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    pat = match or name
    return any(pat in l and "gpusched" not in l for l in o.splitlines())
while True:
    jobs = sorted(json.load(open(F)), key=lambda j: (j["prio"], j["name"]))
    pend = [j for j in jobs if not os.path.exists(j["done"]) and all(os.path.exists(n) for n in j.get("needs", []))]
    if not [j for j in jobs if not os.path.exists(j["done"])]:
        log("ALL_DONE"); break
    fr = free()
    # memory promised to jobs launched in the last SETTLE s (they have not allocated yet)
    for n, (g, m, t) in list(launched.items()):
        if time.time() - t < SETTLE: fr[g] -= m
    started = False
    for j in pend:
        if j["name"] in launched and time.time() - launched[j["name"]][2] < 600: continue
        if running(j["name"], j.get("match")): continue
        g = max(fr, key=fr.get)
        if fr[g] < j["mem"] + MARGIN:
            break                                             # strict priority: nothing lower jumps ahead of a waiting job
        cmd = j["cmd"].format(gpu=g)
        subprocess.Popen(["bash", "-c", cmd], start_new_session=True)
        launched[j["name"]] = (g, j["mem"], time.time()); log(f"start {j['name']} gpu {g} (free {fr[g]} MiB, needs {j['mem']})")
        started = True; break
    time.sleep(STAGGER if started else 20)
