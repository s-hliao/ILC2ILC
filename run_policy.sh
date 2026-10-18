#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
[[ $(git branch --show-current) == quad ]] || { echo 'ERROR: quad branch required'; exit 1; }
source /opt/ros/foxy/setup.bash
source "$PWD/install/setup.bash"
unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ILC2REAL_DIR="$PWD/ilc_f1tenth/ilc2real"
source "$ILC2REAL_DIR/env_tight.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec flock -F -n -E 73 /tmp/ilc2ilc-manual-policy.lock python3 "$PWD/ilc_f1tenth_hw/scripts/manual_policy.py" "$@"
