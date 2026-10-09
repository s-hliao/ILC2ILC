#!/bin/bash
# 40 Hz: nonlinear TO -> lifted linear-tire ILC, then the snapshot's defect-aware iLQR from
# the same TO controls, on the S-curve and the figure-eight. Run inside docker_lla_new with
# the gym bridge up (sim_open40.yaml, ROS_DOMAIN_ID=42).
set -e
cd "$(dirname "$0")"
source /opt/ros/foxy/setup.bash
source /workspaces/lla_drive_ws/install/setup.bash
export ROS_DOMAIN_ID=42 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg ACADOS_SOURCE_DIR=/acados
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/acados/lib
for ref in s_curve figure_eight; do
  python3 -u run_lifted.py --method lifted --model linear --reference $ref --rate 40 \
      --trials 12 --output results/40hz_${ref}_lifted
  python3 -u run_lifted.py --method ilqr --alpha 0.5 \
      --history results/40hz_${ref}_lifted/initial_to.npz --trials 12 \
      --output results/40hz_${ref}_ilqr
done
echo SUITE_DONE
