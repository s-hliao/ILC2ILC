#!/bin/bash
# MB plant (mb_sim/mb_bridge.py up, ROS_DOMAIN_ID=42), 40 Hz.
#  1. ablations of the lifted-vs-iLQR gap, from the same Fiala TOs as run_suite_mb.sh:
#     lifted ILC with blend-model G, and the snapshot iLQR at alpha = 1
#  2. mocap tracks from the IPOPT TO (trajopt.py): lifted linear-tire ILC, snapshot iLQR
set -e
cd "$(dirname "$0")"
source /opt/ros/foxy/setup.bash
source /workspaces/lla_drive_ws/install/setup.bash
export ROS_DOMAIN_ID=42 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg ACADOS_SOURCE_DIR=/acados
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/acados/lib
run() { echo "RUN $*"; python3 -u run_lifted.py "$@"; }
for ref in s_curve figure_eight; do
  run --method lifted --model blend --trials 12 --history references/${ref}_40hz_fiala_to.npz \
      --output results/mb_40hz_${ref}_lifted_blendG
  run --method ilqr --alpha 1.0 --trials 12 --history references/${ref}_40hz_fiala_to.npz \
      --output results/mb_40hz_${ref}_ilqr_alpha1
done
for ref in mocap_square2fast mocap_figfast; do
  run --method lifted --model linear --trials 12 --history references/${ref}_40hz_ipopt_to.npz \
      --output results/mb_40hz_${ref}_lifted
  run --method ilqr --alpha 0.5 --trials 12 --history references/${ref}_40hz_ipopt_to.npz \
      --output results/mb_40hz_${ref}_ilqr
done
echo QUEUE_DONE
