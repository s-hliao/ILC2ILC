#!/bin/bash
# sync_flows.sh (2026-10-22): the flow analysis's CODE (log/dilc/flows: SPEC.md, run_all.sh, merge_flows.py and the
# pogo / car / quad scripts + FLOWS.md) lives outside the git repo; this keeps a copy in the repo (src/transfer/flows)
# so it travels with git. Data (runs/, cache/, logs/, figures) stays in log/ and goes with transfer.sh's tarballs.
#   bash sync_flows.sh to-repo     log/dilc/flows -> src/transfer/flows   (before committing)
#   bash sync_flows.sh install     src/transfer/flows -> log/dilc/flows   (on a new machine; newer files win)
set -e
WS=/home/henry/ilc_ws
LOG=$WS/log/dilc/flows; REPO=$WS/src/transfer/flows
INC=(--exclude='runs/***' --exclude='cache*/***' --exclude='logs/***' --exclude='smoke/***' --exclude='backup/***'
     --exclude='__pycache__/***' --include='*/' --include='*.py' --include='*.sh' --include='*.md' --exclude='*')
case ${1:-} in
  to-repo) mkdir -p $REPO && rsync -a --update --prune-empty-dirs "${INC[@]}" $LOG/ $REPO/ && echo "flows code -> $REPO" ;;
  install) mkdir -p $LOG && rsync -a --update "${INC[@]}" $REPO/ $LOG/ && echo "flows code -> $LOG" ;;
  *) sed -n '2,8p' "$0"; exit 1 ;;
esac
