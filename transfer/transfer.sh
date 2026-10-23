#!/bin/bash
# transfer.sh: move the deep-ILC work (SD2) to another machine. Read HANDOFF.md first.
# (2026-10-22: this file and HANDOFF.md live in the git repo, src/transfer/; ~/ilc_ws/transfer.sh and ~/ilc_ws/HANDOFF.md
#  are symlinks to them. The flow analysis's code is mirrored in src/transfer/flows by sync_flows.sh -- pack refreshes
#  it, restore installs it into log/dilc/flows.)
#
#   on the old machine:   ./transfer.sh pack [OUT_DIR] [--with-checkpoints] [--with-image]
#   copy OUT_DIR to the new machine, e.g.  rsync -avP OUT_DIR/ newhost:~/sd2_transfer/
#   on the new machine:   ~/sd2_transfer/transfer.sh restore [BUNDLE_DIR]
#
# pack writes (default OUT_DIR ~/sd2_transfer):
#   ilc_ws_code.tar.gz     src/ (the git repo incl. .git, uncommitted edits and the f1tenth_gym submodule),
#                          install/, HANDOFF.md, transfer.sh                                        (~30 MB)
#   ilc_ws_results.tar.gz  log/dilc without episode recordings: every result, table, log, script and
#                          config; the sim-stage policies (pre_*/policy); off10/ (the sim stage's init)
#                          and final701/ (robot and perturbation files); log/f1tenth                (~1 GB)
#   ilc_ws_checkpoints.tar.gz  (--with-checkpoints) every hardware-stage iteration's policy
#                          (*/it<k>/policy), for budget / checkpoint evaluations                    (~2.5 GB)
#   ilc_ws_quad_episodes.tar.gz  (--with-quad-episodes) the 2026-10-19..22 quad hardware-stage recordings the flow
#                          analysis reads (tr/recipe, tr/ctl_random, baseline_audit/after/hw24_on_ppo*, clean/,
#                          the drC/drD stage runs if any): real/*.npz with rec_* states                 (~5 GB)
#   ilc_quad_image.tar.gz  (--with-image) the container's image as it is now (docker commit + save)  (several GB)
#   envs/                  conda exports of the host envs dilc and ilcmjx
#   memory/                Claude's project memory for this workspace (see restore)
#   MANIFEST.txt           what was packed, from where, and versions
# Not packed (regenerable or too big, ~110 GB): episode recordings (*.npz under runs), log/sweep, older log/*,
# ~/mujoco_menagerie (public: cloned at a pinned commit on restore).
set -euo pipefail

WS=/home/henry/ilc_ws
MEM_SRC=/home/henry/.claude/projects/-home-henry-ilc-ws/memory
MENAGERIE_COMMIT=da76818
cmd=${1:-}
shift || true

pack() {
  local out=$HOME/sd2_transfer ckpt=0 image=0 qep=0
  for a in "$@"; do
    case $a in
      --with-checkpoints) ckpt=1 ;;
      --with-image) image=1 ;;
      --with-quad-episodes) qep=1 ;;
      *) out=$a ;;
    esac
  done
  mkdir -p "$out/envs" "$out/memory"
  cd "$WS"
  echo "== code (the flow analysis's code mirrored into the repo first)"
  bash "$WS/src/transfer/sync_flows.sh" to-repo
  tar czf "$out/ilc_ws_code.tar.gz" src install HANDOFF.md transfer.sh

  echo "== results (no episode recordings)"
  ( cd "$WS" && {
      # every non-npz file under log/dilc and log/f1tenth
      # (runall/hangs: only real stack dumps; the small stubs come and go with running jobs)
      find log/dilc -type f ! -name '*.npz' ! -path 'log/dilc/runall/hangs/*'
      find log/dilc/runall/hangs -type f -size +300c 2>/dev/null || true
      find log/f1tenth -type f ! -name '*.npz' 2>/dev/null || true
      # the npz that matter: sim-stage policies, the init, robot/perturbation data, f1tenth runs
      find log/dilc -path 'log/dilc/pre_*/policy/*' -type f
      find log/dilc/off10 log/dilc/final701 -type f -name '*.npz'
      find log/f1tenth -type f -name '*.npz' 2>/dev/null || true
      # (2026-10-22) the plane quad: the plan bank's plans, every sim-stage network (*/policy/*.npz, incl. the
      # baselines' ppo_dr_fix_s0/it145/policy and the retrained drC / drD), the box overlay; the flow analysis
      find log/dilc/plane/plans_v3 log/dilc/plane/fixbox/overlay -type f 2>/dev/null || true
      find log/dilc/plane -path '*/policy/*' -name '*.npz' ! -path '*/real/*' ! -path '*/real_*/it*' 2>/dev/null || true
      find log/dilc/flows -type f 2>/dev/null || true
    } | sort -u > "$out/results_files.txt" )
  tar czf "$out/ilc_ws_results.tar.gz" --ignore-failed-read -C "$WS" -T "$out/results_files.txt"

  if [ $ckpt = 1 ]; then
    echo "== hardware-stage checkpoints"
    ( cd "$WS" && find log/dilc -path '*/it*/policy/*' -type f ) > "$out/checkpoint_files.txt"
    tar czf "$out/ilc_ws_checkpoints.tar.gz" --ignore-failed-read -C "$WS" -T "$out/checkpoint_files.txt"
  fi

  if [ $qep = 1 ]; then
    echo "== quad hardware-stage recordings for the flow analysis"
    ( cd "$WS" && find log/dilc/plane/tr/recipe_sd* log/dilc/plane/tr/ctl_random_sd* log/dilc/plane/clean \
        log/dilc/plane/baseline_audit/after/hw24_on_ppo_dr_fix_s0_sd* log/dilc/plane/stage_c* -type f 2>/dev/null ) \
      > "$out/quad_episode_files.txt" || true
    tar czf "$out/ilc_ws_quad_episodes.tar.gz" --ignore-failed-read -C "$WS" -T "$out/quad_episode_files.txt"
  fi

  echo "== conda envs"
  for e in dilc ilcmjx pogo f1t; do
    "$HOME/miniconda3/bin/conda" env export -n $e --no-builds > "$out/envs/$e.yml" 2>/dev/null || true
    "$HOME/miniconda3/envs/$e/bin/python" -m pip freeze > "$out/envs/$e.pip.txt" 2>/dev/null || true
  done

  echo "== memory"
  cp -r "$MEM_SRC/." "$out/memory/" 2>/dev/null || true

  if [ $image = 1 ]; then
    echo "== container image (this takes a while)"
    sg docker -c "docker commit ilc_quad ilc_quad:transfer >/dev/null && docker save ilc_quad:transfer" | gzip > "$out/ilc_quad_image.tar.gz"
  fi

  cp "$WS/transfer.sh" "$WS/HANDOFF.md" "$out/"
  {
    echo "packed $(date -Is) on $(hostname) from $WS"
    echo "git: $(git -C "$WS/src" rev-parse --abbrev-ref HEAD) $(git -C "$WS/src" rev-parse --short HEAD); uncommitted: $(git -C "$WS/src" status --porcelain | grep -vc pycache || true)"
    echo "submodule f1tenth_gym: $(git -C "$WS/src/f1tenth_gym" rev-parse --short HEAD 2>/dev/null || echo not-initialized)"
    echo "mujoco_menagerie: $MENAGERIE_COMMIT"
    echo "container: image ilc_quad:humble, mounts $WS->/ilc_ws and /home/henry/mujoco_menagerie->/mujoco_menagerie, --cpus 28, shm 64 MB, no GPU, PID 1 'sleep infinity'"
    echo "results files: $(wc -l < "$out/results_files.txt")"
    ls -la "$out"
  } > "$out/MANIFEST.txt"
  du -sh "$out"/*.tar.gz
  echo "done: $out (copy it, then run transfer.sh restore on the new machine)"
}

restore() {
  local b=${1:-$(cd "$(dirname "$0")" && pwd)}
  if [ "$HOME" != /home/henry ] && [ ! -d /home/henry ]; then
    cat <<EOF
NOTE: about 80 scripts hardcode /home/henry/... (log/dilc/*.sh, deploy.py, dilc_train.py, the job lists).
Easiest: use a user named henry, or (as root) symlink:  ln -s $HOME /home/henry
Otherwise:  grep -rl /home/henry ~/ilc_ws/src ~/ilc_ws/log/dilc | xargs sed -i "s#/home/henry#$HOME#g"
EOF
  fi
  mkdir -p "$WS"
  echo "== unpacking into $WS"
  tar xzf "$b/ilc_ws_code.tar.gz" -C "$WS"
  tar xzf "$b/ilc_ws_results.tar.gz" -C "$WS"
  [ -f "$b/ilc_ws_checkpoints.tar.gz" ] && tar xzf "$b/ilc_ws_checkpoints.tar.gz" -C "$WS"
  [ -f "$b/ilc_ws_quad_episodes.tar.gz" ] && tar xzf "$b/ilc_ws_quad_episodes.tar.gz" -C "$WS"
  bash "$WS/src/transfer/sync_flows.sh" install          # the flow code from the repo (newer files win)
  [ -e "$WS/transfer.sh" ] || ln -s src/transfer/transfer.sh "$WS/transfer.sh"
  [ -e "$WS/HANDOFF.md" ] || ln -s src/transfer/HANDOFF.md "$WS/HANDOFF.md"

  echo "== mujoco_menagerie @ $MENAGERIE_COMMIT"
  if [ ! -d /home/henry/mujoco_menagerie ]; then
    git clone https://github.com/google-deepmind/mujoco_menagerie.git /home/henry/mujoco_menagerie
    git -C /home/henry/mujoco_menagerie checkout $MENAGERIE_COMMIT
  fi

  echo "== conda envs (needs miniconda at ~/miniconda3; the GPU env needs CUDA 12 drivers)"
  for e in dilc ilcmjx pogo f1t; do
    if [ ! -d "$HOME/miniconda3/envs/$e" ]; then
      "$HOME/miniconda3/bin/conda" env create -n $e -f "$b/envs/$e.yml" \
        || echo "  conda create for $e failed: create it by hand from envs/$e.pip.txt (python 3.11 for dilc, 3.12 for ilcmjx)"
    fi
  done

  echo "== container (needs docker; prefix with 'sg docker -c' if your user is not in the docker group yet)"
  if [ -f "$b/ilc_quad_image.tar.gz" ]; then
    gunzip -c "$b/ilc_quad_image.tar.gz" | docker load && docker tag ilc_quad:transfer ilc_quad:humble
  else
    docker build -t ilc_quad:humble "$WS/src/ilc_quad/docker"   # (src/ilc_quad/REPRODUCE.md: the same)
  fi
  docker run -d --name ilc_quad --cpus 28 --shm-size 64m \
    -v "$WS":/ilc_ws -v /home/henry/mujoco_menagerie:/mujoco_menagerie ilc_quad:humble sleep infinity
  # install/ came with the code; rebuild it inside the container so it matches this machine
  docker exec ilc_quad bash -lc 'cd /ilc_ws && source /opt/ros/humble/setup.bash && colcon build --packages-select ilc_quad'

  echo "== Claude memory"
  # the project memory lives under ~/.claude/projects/<workspace path with / -> ->/memory
  local mdst="$HOME/.claude/projects/$(echo "$WS" | sed 's#/#-#g')/memory"
  mkdir -p "$mdst" && cp -rn "$b/memory/." "$mdst/" && echo "  -> $mdst"

  cat <<EOF

Restored. Checks:
  1. GPU sim:  cd $WS/src/ilc_mjx && ~/miniconda3/envs/ilcmjx/bin/python -c "import jax; print(jax.devices())"
  2. CPU robots (expect 32 jumps, scores as in log/dilc/snapsel/pre_coilcex1_g0_coilc.s0_e300.json):
     docker exec ilc_quad bash -lc 'cd /ilc_ws && source install/setup.bash && export ROS_LOCALHOST_ONLY=1 OMP_NUM_THREADS=1 && \\
       python3 install/ilc_quad/lib/ilc_quad/dilc_execute.py run --policy /ilc_ws/log/dilc/pre_coilcex1/policy \\
       --member g0_coilc.s0_e300 --conds real_r1,real_s1,real_r4,real_r5 --jump 0.45 0 --jump 0.4625 0 \\
       --jump 0.5375 0 --jump 0.55 0 --episodes 2 --seed 301 --jobs 8 --json /tmp/check.json'
  3. Integrity of the transferred results:  ~/miniconda3/envs/dilc/bin/python $WS/log/dilc/runall/validate.py
  4. Then read $WS/HANDOFF.md (section 9: open threads).
  If the new machine's GPUs or CPU count differ, edit the --gpus / --cpu defaults in log/dilc/runall/runall.py
  and the container's --cpus.
EOF
}

case $cmd in
  pack) pack "$@" ;;
  restore) restore "$@" ;;
  *) sed -n '2,25p' "$0"; exit 1 ;;
esac
