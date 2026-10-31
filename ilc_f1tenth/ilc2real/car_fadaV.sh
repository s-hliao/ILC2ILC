#!/bin/bash
# car_fadaV.sh SEED TRACK GPU (2026-10-30; user: on hardware our stage must beat RMA, FADA and PPO+DR zero-shot): the car FADA baseline
# in the CURRENT setup (3.74 kg plans_mu02_tight_m374, the _v08 plans, 1 m envelope, the 14 validation cars), the car's best
# non-oracle FADA recipe (fada_v2h4_real.sh: the v2 IDM, trunk dims 0-4, H 4, LoRA rank 4): IDM data over the SAME DR family as
# PPO+DR / RMA (sim_jax.dr_params 1.0) on the SAME 6 training plans (rmaV / freePPOV), behaviour = the PPO+DR network of this seed
# (freePPOV_s<SEED>; the quad's FADA likewise uses its PPO teacher), then adaptation on each car with OUR stage's lap budget on
# TRACK (hw_stage: 5 iterations x 2 laps = 10 laps; here 10 one-lap episodes, 10.5 laps). -> runs/hw/fadaV_s<SEED>_<TRACK>/
# (<car>_policy.npz, mode fada) + idm.npz (the zero-shot IDM).
cd "$(dirname "$0")"; source env_tight.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=${F1T_PROCS:-3}
PY=${PY:-~/miniconda3/envs/f1t/bin/python}
s=$1 t=$2 g=$3
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.9 ax:mu=0.7 ax:mu=0.6 harsh_combo harsh_steer harsh_sense harsh_drift ax:mu=4.5 ax:mu=5.5,mass=1.1"
PLANS="mocap_square2fast_b0_free_v08 mocap_square2fast_b0_free_v09 mocap_square2fast_b0_free mocap_figfast_b0_free_v08 mocap_figfast_b0_free_v09 mocap_figfast_b0_free"
$PY -u fada_car.py --out runs/hw/fadaV_s${s}_$t --init runs/freePPOV_s$s/policy_final.npz --plans $PLANS --goals mocap_${t}_b0_free_v08 \
  --cars $CARS --dims 0 1 2 3 4 --horizon 4 --iters 10 --seed $s --gpu $g > logs/fadaV_s${s}_$t.log 2>&1
