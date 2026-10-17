#!/bin/bash
# Apply the LLA-MPC-onboard modifications of f1tenth_system to the upstream submodules (src/f1tenth_system at its pinned
# humble-devel commit, with its own vesc and ackermann_mux submodules). Idempotent: a patch that is already applied is
# skipped; one that neither applies nor is applied stops the script (the submodule moved: regenerate the patch).
#   bash src/ilc_f1tenth_hw/scripts/apply_hw_patches.sh        # then colcon build
#   bash src/ilc_f1tenth_hw/scripts/apply_hw_patches.sh --revert
# The ROS distro picks the patch set ($ROS_DISTRO, or --distro foxy|humble): humble -> patches/*.patch against
# f1tenth_system humble-devel; foxy -> patches/foxy/*.patch against foxy-devel (its ackermann_mux already publishes
# ackermann_drive_out and it already ships tf_publisher.py, so the foxy set has no mux patch and no new file).
set -e
DISTRO=${ROS_DISTRO:-humble}; ARGS=()
while [ $# -gt 0 ]; do case $1 in --distro) DISTRO=$2; shift 2;; *) ARGS+=("$1"); shift;; esac; done
set -- "${ARGS[@]}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYS="$(cd "$HERE/../f1tenth_system" && pwd)"
[ -d "$SYS/vesc/vesc_ackermann" ] || git -C "$SYS" submodule update --init --recursive
case $DISTRO in
  foxy) PDIR="$HERE/patches/foxy"; PATCHES="f1tenth_system vesc";;
  humble) PDIR="$HERE/patches"; PATCHES="f1tenth_system vesc ackermann_mux";;
  *) echo "ERROR: no patch set for ROS distro '$DISTRO' (foxy, humble)"; exit 1;;
esac
want=$([ "$DISTRO" = foxy ] && echo foxy-devel || echo humble-devel)
have=$(git -C "$SYS" rev-parse --abbrev-ref HEAD 2>/dev/null || true)
git -C "$SYS" branch -a --contains HEAD 2>/dev/null | grep -q "$want" || \
  echo "WARNING: $SYS (at $have) does not look like $want; the $DISTRO patches expect it"
echo "patch set: $DISTRO ($PDIR)"
for p in $PATCHES; do
  d="$SYS"; [ "$p" != f1tenth_system ] && d="$SYS/$p"
  patch="$PDIR/$p.patch"
  if [ "$1" == "--revert" ]; then
    if git -C "$d" apply --reverse --check "$patch" 2>/dev/null; then git -C "$d" apply --reverse "$patch"; echo "reverted  $p"
    else echo "not applied: $p"; fi
  elif git -C "$d" apply --reverse --check "$patch" 2>/dev/null; then
    echo "already applied: $p"
  elif git -C "$d" apply --check "$patch"; then
    git -C "$d" apply "$patch"; echo "applied   $p"
  else
    echo "ERROR: $p.patch does not apply to $d (its commit moved?)"; exit 1
  fi
done
