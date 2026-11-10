#!/bin/bash
# rmaCov_push.sh (2026-11-07, matcha): push each finished rmaCov_s<s> teacher to Lambda (rmaD_oracle.sh's init)
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real
for s in 0 1 2; do
  until [ -f runs/rmaCov_s$s/policy_final.npz ] && [ -f runs/rmaCov_s$s/agent_final.npz ]; do sleep 60; done
  sleep 30; rsync -a runs/rmaCov_s$s henry@172.26.86.229:ilc_ws/src/ilc_f1tenth/ilc2real/runs/
done
