#!/bin/bash
# fig1_lambda_score.sh (2026-11-07 ~12:25): figure-1 zero-shot scoring of the DAgger diagnostic and fadaR on Lambda (dagger_run.sh /
# fadaR_all.sh's fig1 stage failed: fig1_eval.py was missing on Lambda). Key iterates first. F1T_PROCS 20 (Lambda <= 46 with fadaR fig2).
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_SPIN_RHO=3 F1T_PROCS=20 FADA_JIT=1
PY=$HOME/miniconda3/envs/f1t/bin/python
for s in 0 1 2; do for k in 1 2 5 10 20 40 75 150; do
  f=runs/fig1/fadaR_s$s/it${k}_policy.npz; [ -f $f ] || $PY -c "import numpy as np, os; np.savez('$f', mode='fada_tf', pkl=os.path.abspath('runs/fig1/fadaR_s$s/it$k.pkl'))"
done; done
nice -n 5 $PY -u fig1_eval.py runs/fig1/fig1_eval_dagger.json runs/fig1/dagger_s{0,1,2}/it8_policy.npz > logs/dagger/fig1_eval.log 2>&1
nice -n 5 $PY -u fig1_eval.py runs/fig1/fig1_eval_fadaR.json runs/fig1/fadaR_s{0,1,2}/it150_policy.npz > logs/fadaR/fig1_eval.log 2>&1
nice -n 5 $PY -u fig1_eval.py runs/fig1/fig1_eval_dagger.json runs/fig1/dagger_s{0,1,2}/it{1,2,4}_policy.npz >> logs/dagger/fig1_eval.log 2>&1
nice -n 5 $PY -u fig1_eval.py runs/fig1/fig1_eval_fadaR.json runs/fig1/fadaR_s{0,1,2}/it{1,2,5,10,20,40,75}_policy.npz >> logs/fadaR/fig1_eval.log 2>&1
touch logs/fadaR/FIG1_DONE
