#!/bin/bash
# run_remaining.sh (2026-10-22): completes the pogo flow analysis on any machine; idempotent (every step skips outputs
# that exist), so on the original machine it only fills the gaps. Validation only (seed 301); never sets POGO_EVAL_SET.
#   1. hardware-stage reruns with per-iteration checkpoints (run_stage.sh; skips runs/<arm>_s<s>_PF with a summary.json)
#   2. the flows (flows_pogo.py compute; skips cache/<start>_s<seed>.npz that exist)
#   3. summary.json, summary_extra.json, fig_field_err.png, fig_task_err.png, the phase portraits
# Paths: absolute /home/henry/ilc_ws (POGO in flows_pogo.py, cd in run_stage.sh). On another machine either mirror
# that path or set ILC_WS below and fix the two absolute paths (flows_pogo.py: POGO; run_stage.sh: cd + OUT).
# Inputs that must exist (copy them with the repo):
#   src/ilc_pogo/*.py                                   (hw_stage.py with the append-only --save-at flag)
#   src/ilc_pogo/runs/{nom,drNomInit}_s{0,1,2}/policy_pick.npz, runs/ppo_dr_s{0,1,2}/policy_final.npz   (start nets)
#   src/ilc_pogo/runs/rmaR1_s0/{policy_final,rma_adapt}.npz, runs/rmaR1full_s{1,2}/{policy_final,rma_adapt}.npz (RMA)
#   src/ilc_pogo/runs/v3/fadaF2_s{0,1,2}_{zs,hw24}/real_*_policy.npz                                       (FADA)
#   log/dilc/flows/pogo/runs/*_PF (if already done here: copying them skips step 1, ~25 min x 9 runs)
# Runtime (36-core shared box, load ~38, CPU only): step 1 ~23 min per run, 8 in parallel -> ~50 min for all 9;
# step 2 ~15-20 min (RMA), ~30 min (FADA), ~90 min (a 7-iteration stage start) per (start, seed) job on that loaded box,
# 15 jobs, 8 in parallel -> ~3 h (less on an idle machine; one process per job); step 3 ~10 min.
set -e
ILC_WS=${ILC_WS:-/home/henry/ilc_ws}
D=$ILC_WS/log/dilc/flows/pogo
PY=${PY:-$HOME/miniconda3/envs/pogo/bin/python}
NJ=${NJ:-8}                              # <= 8 CPU processes (SPEC)
unset POGO_EVAL_SET POGO_TEST_SEED POGO_EVAL_TAG
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p $D/logs
bash $D/run_stage.sh
cd $D
# FLOWS_REDO_V1=1 (default here): caches written before 2026-10-22 19:00 (v1: no raw maps, no goal-flow error
# |F(x) - g|) are moved to cache_old/<start>_s<seed>_v1.npz and recomputed; set FLOWS_REDO_V1=0 to keep them
FLOWS_REDO_V1=${FLOWS_REDO_V1:-1} $PY -u flows_pogo.py compute --jobs $NJ >> logs/compute.log 2>&1
$PY -u flows_pogo.py aggregate >> logs/aggregate.log 2>&1
# phase portraits: ours + DR (headline) and ours nominal, seed 0, an inside-DR robot (mass) and an outside one (offset)
$PY -u flows_pogo.py portraits --portrait-start ours+DR --portrait-robots real_mass real_offset >> logs/portraits.log 2>&1
$PY -u flows_pogo.py portraits --portrait-start ours --portrait-robots real_mass real_offset >> logs/portraits.log 2>&1
echo "$(date '+%F %T') REMAINING_DONE" >> logs/run_remaining.log
