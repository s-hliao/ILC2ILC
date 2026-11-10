#!/bin/bash
# hw_sync_1107.sh (2026-11-07, run ON MATCHA): bring the car-session workstation (nsh1609@172.26.79.206, ~/ros2_ws/src/ilc_f1tenth/ilc2real)
# up to matcha's ilc2real CODE for HW_RUNBOOK_N6_1031.md. Checked before writing (md5): the networks (freePPON_s0, rmaN_s0 + rma_adapt),
# the 7 runbook plans in plans_nom06 and the 4 track files are already identical -> not copied. Code that differs (matcha newer):
# hw_stage.py (the runbook's --fit-stall 0 / --z-path 1 do not exist on the workstation: section 4 would stop with "unrecognized
# arguments"), ilc_core.py (the RMA z-path), real_car.py (its env check only accepted the old mu-0.2 plans), policy_runtime.py /
# reeval.py (added policy modes), sim_jax.py / mb_car.py / mb_fiala.py / plan_lqr.py / ppo_dr.py (sim-side), the runbook; + 75 new files.
# NOT touched: ilc_f1tenth_hw (the workstation's net_runner.py is the NEWER one: Nov 4 start check, track offset, mocap-loss stop),
# plans, runs, ~/car_runs, the car. Every file it overwrites is saved first (WS:~/car_runs/backup_pre_sync_1107/). No deletions.
set -e
WS=nsh1609@172.26.79.206
SRC=/home/henry/ilc_ws/src/ilc_f1tenth/ilc2real
DST=ros2_ws/src/ilc_f1tenth/ilc2real
cd $SRC
EXC=(--exclude='*/' --exclude='*.pkl' --exclude='*.npz' --exclude='*.mp4' --exclude='*.gif' --exclude='*.png' --exclude='*.pdf' --exclude='*.json')
echo ">>> would change (checksum):"
rsync -anic "${EXC[@]}" ./ $WS:$DST/ | grep '^<fc' | awk '{print "   ", $2}'
rsync -ac --backup --backup-dir=/home/nsh1609/car_runs/backup_pre_sync_1107 "${EXC[@]}" ./ $WS:$DST/
echo ">>> backups of the overwritten files: $(ssh $WS 'ls ~/car_runs/backup_pre_sync_1107 | wc -l') in ~/car_runs/backup_pre_sync_1107"
echo ">>> verify (md5 matcha vs workstation):"
for f in hw_stage.py hw_eval_requests.py ilc_core.py real_car.py policy_runtime.py bank.py sim_jax.py car_model.py; do
  l=$(md5sum $f | cut -c1-32); r=$(ssh $WS "md5sum $DST/$f" | cut -c1-32)
  [ "$l" = "$r" ] && echo "    ok   $f" || { echo "    FAIL $f"; exit 1; }
done
echo ">>> the runbook's section-4 options on the workstation (f1t env):"
ssh $WS "source ~/anaconda3/etc/profile.d/conda.sh && conda activate f1t && cd $DST && source env_tight.sh >/dev/null && export F1T_MU=0.6 F1T_PLANS=\$PWD/plans_nom06 && \
  python hw_stage.py --help | grep -oE -- '--(fit-stall|z-path|skip-launch|cap) ' | sort -u | tr '\n' ' ' && echo && \
  python -c \"import real_car, policy_runtime as pr, bank; pr.load_policy('runs/rmaN_s0/policy_final.npz'); pr.load_policy('runs/freePPON_s0/policy_final.npz'); \
bank.load_bank(['mocap_peanutM_b0_free_a33', 'mocap_fig8R100_b0_free_x120']); print('    imports, both networks and the plans load')\""
echo ">>> done. Next: source ~/car_runs/car_env.sh plastic|rubber, then the runbook from section 2."
