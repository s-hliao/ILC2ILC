#!/bin/bash
# fig1J_post.sh (2026-11-09 ~01:35): after fig1J_train.sh: (1) ship the retrained networks to ube (its sbJ_sched.sh waits for them: figure 2),
# (2) score checkpoints it009/018/036/054/072/final of all 6 runs zero-shot on the practised panels (fig1_eval_prac.py; the matcha sq41 stage
# check is paused meanwhile), (3) swap them into figure 1's data (old ours/oursdr entries -> runs/fig1_preJ) and rebuild figure 1.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=~/miniconda3/envs/f1t/bin/python; U=henry@nsh1609server5.lan.local.cmu.edu; F=/home/henry/ilc_ws/src/paper/final/budget2
until [ -f logs/fig1J/TRAIN_DONE ]; do sleep 30; done
for r in ours_s0 ours_s1 ours_s2 oursdr_s0 oursdr_s1 oursdr_s2; do
  ssh $U "mkdir -p ilc_ws/src/ilc_f1tenth/ilc2real_fix/runs/fig1/$r"
  scp -q runs/fig1/$r/args.json runs/fig1/$r/log.jsonl runs/fig1/$r/policy_it0*.npz $U:ilc_ws/src/ilc_f1tenth/ilc2real_fix/runs/fig1/$r/
  scp -q runs/fig1/$r/policy_final.npz $U:ilc_ws/src/ilc_f1tenth/ilc2real_fix/runs/fig1/$r/
done
echo "$(date +%H:%M) shipped to ube" >> logs/fig1J/post.txt
SQ=$(ps -eo pid=,args= | awk '/hw_stage[.]py/ && /PRACB_plastic_sq41/ && !/spawn/ {print $1}')
for p in $SQ; do kill -STOP $p; pkill -STOP -P $p; done
L=""; for r in ours_s0 ours_s1 ours_s2 oursdr_s0 oursdr_s1 oursdr_s2; do for c in it009 it018 it036 it054 it072 final; do L="$L runs/fig1/$r/policy_$c.npz"; done; done
F1T_SPIN_RHO=3 F1T_PROCS=28 nice -n 5 taskset -c 0-31 $PY -u fig1_eval_prac.py runs/fig1/fig1J_eval_prac.json $L > logs/fig1J/eval.log 2>&1
for p in $SQ; do pkill -CONT -P $p; kill -CONT $p; done
echo "$(date +%H:%M) eval done; sq41 resumed" >> logs/fig1J/post.txt
mkdir -p $F/fig1_src/runs/fig1_preJ
python3 - $F/fig1_src/runs/fig1/fig1_eval_prac.json runs/fig1/fig1J_eval_prac.json <<'PYEOF'
import json, re, sys, shutil
p, q = sys.argv[1], sys.argv[2]
shutil.copy(p, p.replace('fig1/fig1_eval_prac.json', 'fig1_preJ/fig1_eval_prac.json'))
E = json.load(open(p)); n0 = len(E)
E = {k: v for k, v in E.items() if not re.search(r'fig1/(ours|oursdr)_s\d/', k)}
N = json.load(open(q)); E.update(N); json.dump(E, open(p, 'w'))
print('fig1 eval entries', n0, '->', len(E), '(new', len(N), ')')
PYEOF
for r in ours_s0 ours_s1 ours_s2 oursdr_s0 oursdr_s1 oursdr_s2; do
  [ -d $F/fig1_src/runs/fig1/$r ] && mv $F/fig1_src/runs/fig1/$r $F/fig1_src/runs/fig1_preJ/
  mkdir -p $F/fig1_src/runs/fig1/$r; cp runs/fig1/$r/args.json runs/fig1/$r/log.jsonl $F/fig1_src/runs/fig1/$r/
done
cd $F && python3 car_fig1_prac_data.py fig1_src/runs car_fig1_prac.json && python3 budget_figs2.py car_fig1_prac.json car_fig1prac >> /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix/logs/fig1J/post.txt 2>&1
echo "$(date +%H:%M) figure 1 rebuilt" >> /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix/logs/fig1J/post.txt
