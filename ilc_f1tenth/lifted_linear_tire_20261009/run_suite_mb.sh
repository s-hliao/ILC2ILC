#!/bin/bash
# Multi-body plant (mb_sim/mb_bridge.py up, ROS_DOMAIN_ID=42), 40 Hz: the converged Fiala
# NMPC TO (initial_to.py --model fiala) as trial 1, then the lifted linear-tire ILC, and the
# snapshot's defect-aware iLQR from the same TO controls.
set -e
cd "$(dirname "$0")"
source /opt/ros/foxy/setup.bash
source /workspaces/lla_drive_ws/install/setup.bash
export ROS_DOMAIN_ID=42 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg ACADOS_SOURCE_DIR=/acados
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/acados/lib
for ref in s_curve figure_eight; do
  python3 -u run_lifted.py --method lifted --model linear --trials 12 \
      --history references/${ref}_40hz_fiala_to.npz --output results/mb_40hz_${ref}_lifted
  python3 -u run_lifted.py --method ilqr --alpha 0.5 --trials 12 \
      --history references/${ref}_40hz_fiala_to.npz --output results/mb_40hz_${ref}_ilqr
done
echo SUITE_DONE
