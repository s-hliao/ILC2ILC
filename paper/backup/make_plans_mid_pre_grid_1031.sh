#!/bin/bash
# make_plans_mid.sh (2026-10-31, user: "just fix the plans"): make_plans_sbf.sh with the box face CENTRED between the standing front toes
# (+0.19) and the landing rear toes (x - 0.19): face = x / 2. The old rule max(x - 0.25, x/2 + 0.015) landed the rear feet only 2.7-4.4 cm
# past the face (info_feet_rel), and the robots land 5-6.5 cm short: on boxes >= 11 cm a rear foot off the edge falls (every arm).
# (2026-10-31 evening) each margin's check rollout gets its own run name (a retry reused it and the rollout died with
# FileExistsError), and a plan is kept only if that rollout ran and did not fall.
# was: make_plans_sbf.sh (2026-10-25): see below.
# fix L1): the swinging front feet stay box top + 4 cm high whenever within SBF of the box face on the approach side, so
# a robot running a few cm ahead of the plan no longer strikes the face during push-off. SBF = min(CAP, face - 0.205)
# (the standing front toes are at +0.19: they must be outside the zone when they lift). Everything else = plans_v3
# (default 4 / 3 cm clearance, face at max(x - 0.25, x/2 + 0.015), box margins 0.9 / 0.95 / 1.0). Flat plans: plans_v3's.
#   CAP=0.08 OUT=plans_mid8 bash make_plans_mid.sh
OUT=${OUT:?}; CAP=${CAP:?}
export P=/ilc_ws/log/dilc/plane/$OUT H=/home/henry/ilc_ws/log/dilc/plane/$OUT CAP
mkdir -p $H; for f in /home/henry/ilc_ws/log/dilc/plane/plans_v3/ref_x*_h00.npz; do cp -n $f $H/; done
one() {
  x=$1; h=$2; n=x$(printf %02d $(python3 -c "print(round($x*100))"))_h$(printf %02d $(python3 -c "print(round($h*100))"))
  [ -f $H/ref_$n.npz ] && { echo "$n exists"; return; }
  xf=$(python3 -c "print(round($x / 2, 4))"); sbf=$(python3 -c "print(round(min($CAP, $xf - 0.205), 4))")
  for m in 0.9 0.95 1.0; do
    sg docker -c "docker exec ilc_quad bash -lc 'cd /ilc_ws && source install/setup.bash && export WS=/ilc_ws OMP_NUM_THREADS=1 ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=$((RANDOM % 90 + 5)) && \
      timeout 900 python3 src/ilc_quad/experiments/go1/sweep/solve_plan.py --robot go1 --max-trials 1 --jump $x $h --box $xf $h \
      --param phases:=[30,30,30] --param margin:=$m --param box_setback_front:=$sbf --reference-file $P/ref_tmp_$n.npz --log-dir $P/runs --run-name ${n}_m$m'" > $H/${n}_m$m.log 2>&1; cp $H/${n}_m$m.log $H/$n.log
    miss=$(grep -o 'planned landing \[[^]]*\] (goal \[[^]]*\]' $H/$n.log | python3 -c "import sys,re; v=[float(t) for t in re.findall(r'-?[0-9.]+', sys.stdin.read())]; print(int(len(v) == 5 and max(abs(v[0] - v[3]), abs(v[1] - v[4])) <= 0.01))")
    if grep -q "TO: success=True" $H/$n.log && [ -f $H/ref_tmp_$n.npz ] && [ "$miss" = 1 ] && grep -q "landing miss" $H/$n.log && ! grep -q "FELL" $H/$n.log; then
      mv $H/ref_tmp_$n.npz $H/ref_$n.npz; echo "$n m$m sbf$sbf $(grep -o 'planned landing.*' $H/$n.log | cut -c1-160) | $(grep -o 'landing miss.*' $H/$n.log | head -1)"; return
    fi
    rm -f $H/ref_tmp_$n.npz
  done
  echo "$n FAILED sbf$sbf"
}
export -f one
for x in 0.50 0.55 0.60 0.65; do for h in 0.05 0.10 0.15 0.20; do echo "$x $h"; done; done | grep -vE "^(0.65 0.20|0.60 0.15|0.65 0.15)$" \
  | grep -E "${ONLY:-.}" | xargs -P ${NPAR:-15} -L 1 bash -c 'one "$@"' _
echo done
