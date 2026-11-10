#!/bin/bash
# fig2_queue.sh (2026-11-06, figure 2): our stage runs for figure 2 on ube's CPU (cores 0-29), at most 6 stage runs at once counting the
# step screen's: tier panel for PPO+DR and RMA (seeds 0-2 x 9001-9002); all three panels for figure 1's ours / oursdr networks (seeds
# 0-2 x 9001; each waits for its network). Then the held-out evaluation at every iteration (LADDER_ITS 0-4).
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_fix
npy() { ps -eo comm,args | awk -v p="$1" '$1 ~ /^python/ && $0 ~ p' | wc -l; }
J=(); for n in freePPON rmaN; do for s in 0 1 2; do for ss in 9001 9002; do J+=("$n tier $s $ss"); done; done; done
for n in ours oursdr; do for p in tier plastic rubber; do for s in 0 1 2; do J+=("$n $p $s 9001"); done; done; done
for j in "${J[@]}"; do
  set -- $j
  if [ $1 = ours ] || [ $1 = oursdr ]; then until [ -f runs/fig1/$1_s$3/policy_final.npz ]; do sleep 60; done; fi
  while [ $(npy "hw_stage.py") -ge 6 ]; do sleep 30; done
  bash fig2_stage.sh $j & sleep 20
done
wait
PY=~/miniconda3/envs/f1t/bin/python
PC="n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000"
TC=$($PY -c "import mb_car as m; print('|'.join(sorted(k for k in m.REAL_CARS if k[:5] in ('tiera','tierb','tierc'))))")
export F1T_SPIN_RHO=3 LADDER_ITS="0 1 2 3 4"
E() { taskset -c 0-29 env "$@"; }
E LADDER_OUT=fig2_eval.json LADDER_LAPS=5 F1T_PROCS=24 LADDER_PLANS=mocap_peanutM_b0_free LADDER_CARS="$TC" $PY -u ladder_eval.py runs/fig2/tier_* > logs/fig2/eval.log 2>&1
E LADDER_OUT=fig2_eval.json LADDER_LAPS=5 F1T_PROCS=24 LADDER_PLANS=mocap_peanutM_b0_free_a33 LADDER_CARS="$PC" $PY -u ladder_eval.py runs/fig2/plastic_* >> logs/fig2/eval.log 2>&1
E LADDER_OUT=fig2_eval.json LADDER_LAPS=5 F1T_PROCS=24 LADDER_PLANS=mocap_peanutM_b0_free_x120 $PY -u ladder_eval.py runs/fig2/rubber_* >> logs/fig2/eval.log 2>&1
echo FIG2_STAGE_DONE >> logs/fig2/eval.log
