#!/bin/bash
# car_eval.sh (2026-11-04), ON THE WORKSTATION: hw_eval_requests.py with the policy COPIED into the session folder first.
#   bash ~/car_runs/car_eval.sh --session-dir $S --policy runs/rmaN_s0/policy_final.npz --tag ... --plan ... --laps 5 --runs 2
#   (--score calls pass straight through)
# Why: hw_eval_requests.py writes the policy's absolute path into req_<k>.json and the car's net_runner loads that path; a
# workstation path does not exist on the car. The copy goes to $S/policies/<tag>/ (+ rma_adapt.npz, which
# policy_runtime.load_policy reads from next to an RMA policy). hw_stage.py --backend manual already writes its policies
# into the session folder and needs no wrapper.
bash $HOME/car_runs/place_view/ensure_view.sh   # (2026-11-22) live placement window
ARGS=("$@"); SD=; POL=; TAG=
for ((i=0; i<${#ARGS[@]}; i++)); do case "${ARGS[i]}" in
  --session-dir) SD=${ARGS[i+1]};; --policy) POL=${ARGS[i+1]}; PI=$((i+1));; --tag) TAG=${ARGS[i+1]};; --plan) PLAN=${ARGS[i+1]};; esac; done
[ -n "$PLAN" ] && { bash $HOME/car_runs/check_offset_1122.sh "$PLAN" || exit 1; }   # (2026-11-22) per-track offset, RUNBOOK_1122
H=$HOME/ros2_ws/src/ilc_f1tenth/ilc2real/hw_eval_requests.py
[ -z "$POL" ] && exec python $H "$@"
[ -n "$SD" ] && [ -n "$TAG" ] && [ -f "$POL" ] || { echo "!!! need --session-dir, --tag and an existing --policy"; exit 1; }
case "$(realpath -m "$SD")" in /home/kathy/car_sessions/*) ;; *) echo "!!! --session-dir must be under /home/kathy/car_sessions (the car's folder)"; exit 1;; esac
D=$SD/policies/$TAG; mkdir -p $D
cp "$POL" $D/ || exit 1
AD=$(dirname "$POL")/rma_adapt.npz; [ -f "$AD" ] && cp "$AD" $D/
cmp -s "$POL" $D/$(basename "$POL") || { echo "!!! copy differs"; exit 1; }
ARGS[PI]=$D/$(basename "$POL")
echo ">>> policy copied to $D ($(ls $D | tr '\n' ' '))"
exec python $H "${ARGS[@]}"
