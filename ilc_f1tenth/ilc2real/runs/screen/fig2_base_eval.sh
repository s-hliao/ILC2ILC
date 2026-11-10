#!/bin/bash
# fig2_base_eval.sh (2026-11-06, figure 2): after fig2_base.sh, each baseline run's checkpoints (runs/hw/<arm>_<panel>_s<s>_ep<K>/<car>_policy.npz,
# K = 0 / 8 / 24 / 96 / 384 real laps per car) linked as runs/fig2base/<arm>_<panel>_s<s>/its/it<K>_<car>_policy.npz and scored as the main
# tables (ladder_eval.py, fixed simulator, F1T_SPIN_RHO=3) on the panel's held-out plan -> fig2_eval.json
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; PY=$HOME/miniconda3/envs/f1t/bin/python
until [ -f logs/FIG2_BASE_DONE ]; do sleep 120; done
for arm in fig2ppo fig2fada; do for p in tier plastic rubber; do for s in 0 1 2; do
  D=runs/fig2base/${arm}_${p}_s$s; mkdir -p $D/its
  for K in 0 8 24 96 384; do E=runs/hw/${arm}_${p}_s${s}_ep$K; [ -d $E ] || continue
    for f in $E/*_policy.npz; do c=$(basename $f _policy.npz); ln -sf $PWD/$f $D/its/it${K}_${c}_policy.npz; done
    [ -f $E/rma_adapt.npz ] && ln -sf $PWD/$E/rma_adapt.npz $D/its/rma_adapt.npz
  done
done; done; done
TC=$($PY -c "import mb_car as m; print('|'.join(sorted(k for k in m.REAL_CARS if k[:5] in ('tiera','tierb','tierc'))))")
PC='n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000'
export F1T_SPIN_RHO=3 F1T_PROCS=${F1T_PROCS:-6} LADDER_ITS="0 8 24 96 384"
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free LADDER_CARS="$TC" nice -n 5 $PY -u ladder_eval.py runs/fig2base/*_tier_s? > logs/fig2_base_eval.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_a33 LADDER_CARS="$PC" nice -n 5 $PY -u ladder_eval.py runs/fig2base/*_plastic_s? >> logs/fig2_base_eval.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_x120 nice -n 5 $PY -u ladder_eval.py runs/fig2base/*_rubber_s? >> logs/fig2_base_eval.log 2>&1
touch logs/FIG2_BASE_EVAL_DONE
