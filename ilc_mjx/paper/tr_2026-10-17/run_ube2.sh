#!/bin/bash
# run_ube2.sh (2026-10-22 22:00, ube): replaces run_ube.sh's tail so everything lands by ~04:00 (the user's deadline).
# run_ube.sh's driver was stopped after drC_s2 / drEft_s2 finished; its children keep running (the pool arm on drB_s2
# -> tr/pool_sd*, and drCft30_s2's training). Differences from run_ube.sh:
#   - the pick does not wait for drCft30_s2 (its CPU val is added afterwards, for the record only)
#   - stage_c runs 8 at a time
#   - a CPU schedule (<= 30 CPUs): fly tokens 3 while the quad goal-field flights + pogo flows run; 5 once the
#     goal-field flights end (then the car seed-0 flows start, 6 procs); 6 once the pogo flows end
set -u
P=/home/henry/ilc_ws/log/dilc/plane; T=/home/henry/ilc_ws/src/ilc_mjx/paper/tr_2026-10-17; F=/home/henry/ilc_ws/log/dilc/flows
PY=~/miniconda3/envs/ilcmjx/bin/python
log() { echo "$(date +%T) $*"; }
export DILC_FLY_JOBS=4 GPU0=0 NGPU=3
echo 3 > $P/fixbox/fly_tokens
# --- the CPU schedule, in the background
(
  while pgrep -f "goalfield.py fly" > /dev/null; do sleep 30; done
  echo 5 > $P/fixbox/fly_tokens; log "goal-field flights done: fly tokens 5; car seed-0 flows start"
  ( cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real && source env_tight.sh > /dev/null 2>&1
    cd $F/car
    export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    CPY=~/miniconda3/envs/f1t/bin/python
    for n in v3drBtL_s0 ppoL_task_s0 rmaL_task_s0; do
      d=runs/${n}_ladR_val_its
      [ -f fields/${n}_ladR_val_its.json ] || $CPY -u flows_car.py compute --stage $d --procs 6 > logs/flows_$n.log 2>&1 || log "FAILED car flows $n"
      for c in real_mu ax:mu=0.6; do
        [ -f "figs/portrait_${n}_ladR_val_its_$c.png" ] || $CPY -u flows_car.py portrait --stage $d --car $c --procs 6 >> logs/portrait_$n.log 2>&1
      done
      log "car $n flows + portraits done"
    done
    $CPY flows_car.py report > logs/report.log 2>&1; log "car report done" )
  while ! grep -q REMAINING_DONE $F/pogo/logs/run_remaining.log 2>/dev/null; do sleep 30; done
  echo 6 > $P/fixbox/fly_tokens; log "pogo flows done: fly tokens 6"
) >> $P/run_ube2_sched.out 2>&1 &
# --- CPU val of the finished candidates, the pick
CANDS="drC_s2:30 drCft_s2:15 drDft_s2:15 drEft_s2:15 drB_s2:30"
for n in $CANDS; do
  d=${n%%:*}; it=${n##*:}
  [ -f $P/$d/cpu_val_g0_plane_it$it.json ] || JOBS=12 bash $P/cpu_eval.sh $P/$d/policy g0_plane_it$it val 2
  [ -f $P/$d/cpu_val_e8_g0_plane_it$it.json ] || JOBS=12 bash $P/cpu_eval.sh $P/$d/policy g0_plane_it$it val 8
done
pick() { $PY - $P "$@" <<'PY'
import json, os, sys
import numpy as np
P = sys.argv[1]; best = None
for n in sys.argv[2:]:
    d, it = n.split(":")
    for tag in ("", "_e8"):
        f = f"{P}/{d}/cpu_val{tag}_g0_plane_it{it}.json"
        if not os.path.exists(f):
            print(f"{d} it{it}{tag}: missing", file=sys.stderr); continue
        r = json.load(open(f))
        s = np.array([0.01 * x["landing_cost"] + 20 * x["fell"] for x in r])
        bs = np.random.default_rng(0).integers(0, len(s), (5000, len(s)))
        lo, hi = np.percentile(s[bs].mean(1), [2.5, 97.5])
        ok = np.mean([(not x["fell"]) and abs(x["ex"]) <= 0.05 and abs(x["ez"]) <= 0.03 for x in r])
        rb = {c: np.mean([0.01 * x["landing_cost"] + 20 * x["fell"] for x in r if x["cond"] == c]) for c in ("real_r1", "real_s1", "real_r4", "real_r5")}
        print(f"{d:11s} it{it}{tag:3s} n={len(r):3d} score {s.mean():5.2f} [{lo:5.2f},{hi:5.2f}] ok {ok:4.0%} falls {sum(x['fell'] for x in r):3.0f}  "
              + " ".join(f"{c[5:]} {v:5.1f}" for c, v in rb.items()), file=sys.stderr)
        if tag == "_e8" and d != "drB_s2" and (best is None or s.mean() < best[0]):
            best = (s.mean(), d, it)
print(f"{best[1]}:{best[2]}" if best else "")
PY
}
NETSEL=$(pick $CANDS 2> $P/cpu_val_table.txt); cat $P/cpu_val_table.txt
log "picked: $NETSEL"
if [ -n "$NETSEL" ]; then
  d=${NETSEL%%:*}; it=${NETSEL##*:}; c=0.05; [ $d = drEft_s2 ] && c=0.07
  NET=$P/$d/policy MEMBER=g0_plane_it$it CLR=$c OUT=$P/stage_c_$d NPAR=8 bash $T/stage_c.sh > $P/stage_c_$d.out 2>&1 &
  PS=$!
fi
# drCft30_s2: CPU val once trained (record only)
while [ ! -f $P/drCft30_s2/policy/g0_plane_it30.npz ]; do sleep 60; done
sleep 30
JOBS=8 bash $P/cpu_eval.sh $P/drCft30_s2/policy g0_plane_it30 val 2
JOBS=8 bash $P/cpu_eval.sh $P/drCft30_s2/policy g0_plane_it30 val 8
pick $CANDS drCft30_s2:30 2> $P/cpu_val_table.txt > /dev/null; log "with drCft30_s2:"; cat $P/cpu_val_table.txt
[ -n "${PS:-}" ] && wait $PS
log "stage_c done"
while [ $(ls $P/tr/pool_sd*/*/eval_planeval_final.json 2>/dev/null | wc -l) -lt 12 ] || pgrep -f "OUT=$P/tr" > /dev/null; do sleep 60; done
cd $T
for d in $P/stage_c_*; do [ -d $d ] && $PY quad_table.py --dir $d | tee $d/table.txt; done
$PY quad_table.py --dir $P/tr recipe pool ctl_random nbr05 | tee $P/tr/table_pool.txt
log "RUN_UBE_DONE"
