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
SMI, CUDA, LOADMAX = os.environ.get("SMI_IDX"), os.environ.get("CUDA_IDX"), float(os.environ.get("LOADMAX", "1e9"))
CPULOAD = float(os.environ.get("CPULOAD", "28"))            # CPU-only jobs start only below this load (matcha: 36 CPUs)
MAXRUN = int(os.environ.get("MAXRUN", "1000000"))           # ube CPU-only jobs: at most this many of this file's jobs at once
MAXPERGPU = int(os.environ.get("MAXPERGPU", "1000000"))     # ube (user: "don't load them heavily"): at most this many live jobs per GPU
def free():
    if SMI:                                                   # ube: only one card works; nvidia-smi without -i fails on GPU 0
        o = subprocess.run(["nvidia-smi", "-i", SMI, "--query-gpu=memory.free,temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        b, t = o.strip().split(",")
        return {int(CUDA): int(b) if int(t) < TMAX else 0}
    o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    # (2026-10-30) a GPU at >= TMAX C takes no new job (the check-in limit is 88 C; all four ran at 84-86 C)
    return {int(a): (int(b) if int(t) < TMAX else 0) for a, b, t in (l.split(",") for l in o.strip().splitlines())}
def gpu_procs():
    # live jobs per GPU index: processes holding >= 300 MiB (a car stage's main process; its spawned workers hold ~160 MiB)
    idx = dict(l.split(", ") for l in subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id,index", "--format=csv,noheader"],
               capture_output=True, text=True).stdout.strip().splitlines())
    o = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    n = {}
    for l in o:
        b, p, m = l.split(", ")
        if int(m) >= 300: n[int(idx[b])] = n.get(int(idx[b]), 0) + 1   # a job's main process (workers hold ~160 MiB)
    return n
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
    if os.getloadavg()[0] > LOADMAX: pend = []                # ube: CPU cap (user: <= 34 of 36 CPUs)
    if sum(running(j["name"], j.get("match")) for j in jobs if not os.path.exists(j["done"])) >= MAXRUN: pend = []
    # CPU-only jobs (mem 0; 2026-10-31): no GPU slot, only a CPU-load cap; up to 4 per pass, ahead of and apart from the GPU queue
    ncpu = 0
    for j in [j for j in pend if j["mem"] == 0]:
        if ncpu >= 4 or os.getloadavg()[0] + ncpu > CPULOAD: break
        if j["name"] in launched or running(j["name"], j.get("match")): continue
        subprocess.Popen(["bash", "-c", j["cmd"].format(gpu=0)], start_new_session=True)
        launched[j["name"]] = (0, 0, time.time()); log(f"start {j['name']} cpu (load {os.getloadavg()[0]:.1f})"); ncpu += 1
    pend = [j for j in pend if j["mem"] > 0]
    for j in pend:
        if j["name"] in launched and time.time() - launched[j["name"]][2] < 600: continue
        if running(j["name"], j.get("match")): continue
        busy = gpu_procs()                                    # live processes per GPU (nvidia-smi), whoever started them
        for n, (g_, m_, t_) in launched.items():
            if time.time() - t_ < SETTLE: busy[g_] = busy.get(g_, 0) + 1   # launched, not on the GPU yet
        ok = {g_: f_ for g_, f_ in fr.items() if busy.get(g_, 0) < MAXPERGPU}
        if not ok: break
        g = max(ok, key=ok.get)
        if fr[g] < j["mem"] + MARGIN:
            break                                             # strict priority: nothing lower jumps ahead of a waiting job
        cmd = j["cmd"].format(gpu=g)
        subprocess.Popen(["bash", "-c", cmd], start_new_session=True)
        launched[j["name"]] = (g, j["mem"], time.time()); log(f"start {j['name']} gpu {g} (free {fr[g]} MiB, needs {j['mem']})")
        started = True; break
    time.sleep(STAGGER if started else 20)
