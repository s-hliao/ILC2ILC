#!/bin/bash
# fig2_base.sh (2026-11-06, paper/FIGURE_PLAN_1106.md figure 2): the car baselines given real laps, on the fixed simulator (F1T_SPIN_RHO=3),
# each panel's practice plans as our stage's, seeds 0-2, real-lap checkpoints 8 / 24 / 96 / 384 per car (per-iter 8):
#   PPO+DR + PPO on the car   ppo_real_car.py --init runs/freePPON_s<s> (privileged critic)       -> runs/hw/fig2ppo_<panel>_s<s>_ep<K>
#   FADA + adaptation         fada_real_car.py --idm figure 1's fada_d60_s<s> (H 4, lam 0.8)       -> runs/hw/fig2fada_<panel>_s<s>_ep<K>
# matcha CPU (JAX on the CPU), after fig2_its12.sh; FADA waits for figure 1's IDMs (copied from ube into runs/fig1/).
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu F1T_PROCS=3
PY="nice -n 5 $HOME/miniconda3/envs/f1t/bin/python -u"
until [ -f logs/FIG2_ITS12_DONE ]; do sleep 60; done
TIER=$($HOME/miniconda3/envs/f1t/bin/python -c "import mb_car as m; print(' '.join(sorted(k for k in m.REAL_CARS if k[:5] in ('tiera','tierb','tierc'))))")
declare -A CARS=([tier]="$TIER" [plastic]="n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000" [rubber]="ax:mu=1.667 ax:mu=1.833,mass=1.1")
declare -A GOAL=([tier]=mocap_square2fast_b0_free [plastic]=mocap_square2fast_b0_free_a45 [rubber]=mocap_fig8R100_b0_free_x120)
mkdir -p logs/fig2
ppo() { $PY ppo_real_car.py --init runs/freePPON_s$2/policy_final.npz --arm fig2ppo_$1_s$2 --cars ${CARS[$1]} --goals ${GOAL[$1]} \
          --per-iter 8 --checkpoints 8 24 96 384 --seed $((4242 + $2)) > logs/fig2/ppo_$1_s$2.log 2>&1; }
fada() { mkdir -p runs/fig1; until rsync -a henry@nsh1609server5.lan.local.cmu.edu:ilc_ws/src/ilc_f1tenth/ilc2real_fix/runs/fig1/fada_d60_s$2 runs/fig1/ 2>/dev/null && [ -f runs/fig1/fada_d60_s$2/summary.json ]; do sleep 120; done
         $PY fada_real_car.py --idm runs/fig1/fada_d60_s$2/idm.npz --arm fig2fada_$1_s$2 --cars ${CARS[$1]} --goals ${GOAL[$1]} \
          --eval-plans ${GOAL[$1]} --horizon 4 --lam 0.8 --per-iter 8 --checkpoints 8 24 96 384 --seed $((5151 + $2)) > logs/fig2/fada_$1_s$2.log 2>&1; }
for p in rubber plastic tier; do for s in 0 1 2; do ppo $p $s & done; wait; done
for p in rubber plastic tier; do for s in 0 1 2; do fada $p $s & done; wait; done
touch logs/FIG2_BASE_DONE
