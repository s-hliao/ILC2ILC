#!/bin/bash
# Apply the LLA-MPC-onboard modifications of f1tenth_system to the upstream submodules (src/f1tenth_system at its pinned
# humble-devel commit, with its own vesc and ackermann_mux submodules). Idempotent: a patch that is already applied is
# skipped; one that neither applies nor is applied stops the script (the submodule moved: regenerate the patch).
#   bash src/ilc_f1tenth_hw/scripts/apply_hw_patches.sh        # then colcon build
#   bash src/ilc_f1tenth_hw/scripts/apply_hw_patches.sh --revert
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYS="$(cd "$HERE/../f1tenth_system" && pwd)"
[ -d "$SYS/vesc/vesc_ackermann" ] || git -C "$SYS" submodule update --init --recursive
for p in f1tenth_system vesc ackermann_mux; do
  d="$SYS"; [ "$p" != f1tenth_system ] && d="$SYS/$p"
  patch="$HERE/patches/$p.patch"
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
