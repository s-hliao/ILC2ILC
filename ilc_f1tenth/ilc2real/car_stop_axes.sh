#!/bin/bash
# The car's perturbation suite (scheduler_lap10.py's *_l10 suites) redone for the paper's final rows (2026-10-11): our headline
# network = the STOPPING-RULE network (v3drBt_stop; zero-shot and its 10-lap arm), and the RMA / FADA baselines = only the
# best non-oracle and best oracle of each (pick_nets.py <- src/paper/baselines.json). Same cars / plans / seeds / runs as the
# *_l10 suites -> runs/axes/<suite>_l10s.json; then the stop network adapted 10 laps ON each perturbed car
# (ax_*_drBtS_lap10, reeval'd) and its trained x tested matrix (tp_matrix_l10s).
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python; PR=${F1T_PROCS_AX:-12}
EV4="--plans mocap_square2fast_b25 mocap_figfast_b25 mocap_square2fast_b0 mocap_figfast_b0"
NETS="ours_stop=arm:runs/hw/v3drBt_stop_lap10 ours_stop_zs=zs:runs/v3drBt_stop/policy_final.npz $(python3 pick_nets.py)"
SWEEP="mass=0.8 mass=0.9 mass=1.1 mass=1.4 mu=0.7 mu=0.9 mu=1.1 ky=0.7 ky=0.85 ky=1.2 kt=0.7 kt=0.85 kt=1.15 tau=0.06 tau=0.08 tau=0.1 d_delay=1 d_delay=2 i_delay=1 i_delay=2 d_off=0.03 d_off=-0.05"
SENSE="mocapbad latency2 lowrate ekflag noisy3"
REAL="real_nom real_mass real_mu real_act real_lag"
ev() { local tag=$1; shift; [ -f runs/axes/$tag.json ] || F1T_PROCS=$PR $PY -u eval_axes.py --tag $tag --net $NETS "$@" > logs/axes_$tag.log 2>&1; echo "$(date +%T) $tag"; }
ev sweep_l10s --cars $(for x in $SWEEP; do echo -n "ax:$x "; done) $EV4
ev sense_l10s --cars $(for c in $REAL; do for s in $SENSE; do echo -n "$c+$s "; done; done) $EV4
ev start_direct_l10s --cars $REAL $EV4 --laps 3 --launch 0.6 0.6 0
ev start_launch_l10s --cars $REAL $EV4 --laps 3 --launch 0.6 1.0 15
ev heldout_l10s --cars $REAL --plans mocap_squareH_b25 mocap_squareH_b21 mocap_squareH_b14 mocap_squareH_b0
# the stop network adapted ON each perturbed car (10 laps: the two b25 plans alternated, 2-lap chained trials)
ARMS=""
for car in $(for x in $SWEEP; do echo -n "ax:$x "; done) $(for s in $SENSE; do echo -n "real_nom+$s "; done); do
  t=$(echo "$car" | sed 's/ax:/ax_/; s/real_nom+/sense_/; s/=//; s/,/_/')_drBtS_lap10
  [ -f runs/hw/$t/summary.json ] || F1T_PROCS=8 $PY -u hw_stage.py --policy runs/v3drBt_stop/policy_final.npz --out runs/hw/$t \
    --iters 5 --goals mocap_square2fast_b25 mocap_figfast_b25 --goal-cycle 1 --cars "$car" --gpu 1 > logs/hw_$t.log 2>&1
  ARMS="$ARMS $t"
done
echo "$(date +%T) adapted arms"
F1T_PROCS=$PR $PY -u reeval.py $ARMS > logs/reeval_drBtS_lap10.log 2>&1
TP="ax:mass=1.4 ax:mu=0.7 ax:ky=0.7 ax:kt=0.7 ax:d_delay=2 ax:i_delay=2 real_nom+mocapbad real_nom+latency2"
TPN=""; for c in $TP; do t=$(echo "$c" | sed 's/ax:/ax_/; s/real_nom+/sense_/; s/=//'); TPN="$TPN $t=file:runs/hw/${t}_drBtS_lap10/${c}_policy.npz"; done
[ -f runs/axes/tp_matrix_l10s.json ] || F1T_PROCS=$PR $PY -u eval_axes.py --tag tp_matrix_l10s --cars real_nom $TP \
  --net nominal_adapted=arm:runs/hw/v3drBt_stop_lap10 $TPN $EV4 > logs/axes_tp_matrix_l10s.log 2>&1
echo "$(date +%T) STOP_AXES_DONE"
