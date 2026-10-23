#!/bin/bash
# run_ube.sh (2026-10-22, ube: 3 GPUs 0-2, <= 30 CPUs): run_remaining_quad.sh's steps for the machine the 10-22
# handoff landed on. matcha's logs were not transferred (drCft_s2 / drDft_s2 / tr ctl_random / cpu_eval.sh /
# bank_v3.json), so: bank_v3.json = drB_s2/bank.json (its byte copy), cpu_eval.sh rebuilt (reproduces drB_s2 it30's
# 19.36 exactly), the fine-tunes are retrained, and the drB_s2 control arm is reflown next to the pool arm.
#   A  (now)  the stage-side fix on drB_s2 it30 (step 4): pool + ctl_random -> tr/, paired with tr/recipe
#   B  (now)  retrain drC_s2 / drCft_s2 / drDft_s2 (retrain_c.sh's lines) + two extra candidates the handoff names:
#             drCft30_s2 (the fine-tune, 30 its) and drEft_s2 (the fine-tune with --clear 0.07)
#   C  CPU val zero-shot of every candidate: run_remaining's 64 jumps, plus 256 (8 episodes) for a tighter pick
#   D  pick the lowest 256-jump val score -> stage_c.sh (recipe / pool / ctl_random x 3 seeds x 4 robots), its clearance
#   E  tables
set -u
P=/home/henry/ilc_ws/log/dilc/plane; T=/home/henry/ilc_ws/src/ilc_mjx/paper/tr_2026-10-17
PY=~/miniconda3/envs/ilcmjx/bin/python
export DILC_FLY_JOBS=4 GPU0=0 NGPU=3
echo "$(date +%T) start"
( NET=$P/drB_s2/policy MEMBER=g0_plane_it30 CLR=0.03 OUT=$P/tr ARMS="pool ctl_random" NPAR=4 bash $T/stage_c.sh ) > $P/tr/stage_pool_ube.out 2>&1 &
PA=$!
R="--bank $P/bank_v3.json --beta 0.7 --cap 0.15 --replay 3 --steps 1000 --anchor-pool 20 --anchor-w 1.0 --lr-final 1e-4 --q-off 0.05 --a-off 0.03 --est-window 0.025 --dr"
FT="--seed 2 --eval-every 5 --init-npz $P/drB_s2/policy/g0_plane_it30.npz --lr 1e-4"
train() { n=$1; g=$2; shift 2
  [ -f $P/$n/policy/g0_plane_it$IT.npz ] && return
  [ -d $P/$n ] && mv $P/$n $P/${n}_partial_$(date +%s)
  ( cd /home/henry/ilc_ws/src/ilc_mjx
    ILC_QUAD_OVERLAY=/ilc_ws/log/dilc/plane/fixbox/overlay JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_ilc XLA_PYTHON_CLIENT_PREALLOCATE=false \
    OMP_NUM_THREADS=3 $PY scripts/plane_train.py --gpu $g $R --out $P/$n "$@" 2>&1 | grep --line-buffered -v "warp\|bfc_alloc" > $P/$n.out ) ; }
IT=30 train drC_s2 2 --clear 0.05 --seed 2 --iters 30 --dr-async &
IT=15 train drCft_s2 0 --clear 0.05 --iters 15 --dr-async $FT &
IT=15 train drDft_s2 1 --clear 0.05 --iters 15 $FT &
# the extras wait for GPU memory to be checked (EXTRAS file: written by hand once the first three fit)
while [ ! -f $P/.ube_extras ] && [ -n "$(jobs -rp | grep -v "^$PA$")" ]; do sleep 30; done
if [ "$(cat $P/.ube_extras 2>/dev/null)" = go ]; then
  IT=30 train drCft30_s2 0 --clear 0.05 --iters 30 --dr-async $FT &
  IT=15 train drEft_s2 1 --clear 0.07 --iters 15 --dr-async $FT &
fi
for j in $(jobs -rp); do [ $j != $PA ] && wait $j; done
echo "$(date +%T) training done"
CANDS="drC_s2:30 drCft_s2:15 drDft_s2:15 drCft30_s2:30 drEft_s2:15 drB_s2:30"
for n in $CANDS; do
  d=${n%%:*}; it=${n##*:}
  [ -f $P/$d/policy/g0_plane_it$it.npz ] || continue
  [ -f $P/$d/cpu_val_g0_plane_it$it.json ] || JOBS=12 bash $P/cpu_eval.sh $P/$d/policy g0_plane_it$it val 2
  [ -f $P/$d/cpu_val_e8_g0_plane_it$it.json ] || JOBS=12 bash $P/cpu_eval.sh $P/$d/policy g0_plane_it$it val 8
done
NETSEL=$($PY - $P $CANDS <<'PY'
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
)
echo "$(date +%T) picked: $NETSEL"
if [ -n "$NETSEL" ]; then
  d=${NETSEL%%:*}; it=${NETSEL##*:}; c=0.05; [ $d = drEft_s2 ] && c=0.07
  echo 5 > $P/fixbox/fly_tokens                     # the training's CPUs are free
  NET=$P/$d/policy MEMBER=g0_plane_it$it CLR=$c OUT=$P/stage_c_$d NPAR=6 bash $T/stage_c.sh > $P/stage_c_$d.out 2>&1
fi
wait $PA
cd $T
for d in $P/stage_c_*; do [ -d $d ] && $PY quad_table.py --dir $d | tee $d/table.txt; done
$PY quad_table.py --dir $P/tr recipe pool ctl_random | tee $P/tr/table_pool.txt
echo "$(date +%T) RUN_UBE_DONE"
