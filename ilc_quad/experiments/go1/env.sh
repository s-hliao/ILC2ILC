# Settings shared by the experiment scripts (sourced, not run). Override any of them in
# the environment, e.g.  JOBS=12 LOG=/data/ilc_log ./validate.sh /data/ilc_log/final2
EXP=${EXP:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}   # this folder
WS=${WS:-/ilc_ws}                    # colcon workspace; install/ must be built
LOG=${LOG:-$WS/log}                  # where runs are written
SUITE=${SUITE:-$LOG/final2}          # the nominal suite: transfer sources for the robustness grid
JOBS=${JOBS:-3}                      # runs at once (one process each)
THREADS=${THREADS:-2}                # BLAS / OpenMP threads per run
export EXP WS LOG SUITE JOBS THREADS
export ROS_LOCALHOST_ONLY=1 OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS \
    MKL_NUM_THREADS=$THREADS
source "$WS/install/setup.bash"
