# Car-side fixes from the 11-08 / 11-09 hardware sessions (2026-11-15)

`orig/` holds the car-side scripts **as they ran**, copied from
`nsh1609:~/ros2_ws/src/hw_data/car_results_stage/code/car_side/`. `fixed/` holds the patched versions. Nothing has
been installed on the car yet.

## What failed, and what fixes it

56 of the 255 flights were voided (`src/ilc_f1tenth/hw_summary_1115.txt`; reasons are in each `req_*.json`).

| failure (example runs) | already fixed in the as-ran code | fixed here |
|---|---|---|
| one-frame heading flips of 140-149 deg, which ended runs as false "spin" stops (00032, 00046, 00064) | `optitrack_node` drops yaw flips (11-09) | moved unchanged into `PoseGate`, now unit-tested |
| upside-down rigid-body solves, tilt 169-172 deg (00081) | dropped by the tilt check (11-09) | same |
| late or out-of-order Motive frames (Motive take-video stalls) | `drop_late` (11-08) | — |
| **position jumps of 0.5-1.9 m in one frame** (00043, 00088, 00090, 00092) | **not handled** | `PoseGate` drops any step beyond 0.05 m + 7 m/s x dt since the last accepted frame. The check lapses after 0.3 s without an accepted frame, so a real re-acquisition is accepted |
| **the runner's mocap-loss check counted rejected frames as fresh** | — | `net_runner` now watches `/optitrack/pose_ok`, the frames `optitrack_node` accepted, instead of the raw `/dl_f1tenth/pose` |
| no braking (VESC `current_min: 0`) | `vesc.yaml` `current_min: -25` since 11-09 08:16 | — (verified in the as-ran `vesc.yaml`) |
| VESC serial errors and driver deaths (00116-00125) | — | **hardware:** the frayed control-board wire. Replace it; software can only stop safely on 'stale', which it already does |
| a battery died mid-flight (00047) | — | **procedure:** check the voltage before each block of flights |

## Tests

```
cd fixed && python3 -m pytest test_pose_gate.py -q     # 6 passed; no ROS needed
```

The tests feed synthetic 240 Hz power-slide streams (2.5 m/s at 2.5 rad/s, the hardest logged motion) and check:
- a real slide is never dropped;
- a one-frame 140 deg flip is dropped, and the stream recovers on the next frame;
- a persistent upside-down solve is always dropped;
- 1.9 m and 0.49 m position jumps are dropped;
- alternating 1.2 m mis-solves on a straight are dropped frame by frame;
- a re-acquisition after a 0.5 s dropout 1.25 m away is accepted.

## Install on the car (when you choose to)

```
scp fixed/optitrack_node.py fixed/net_runner.py kathy@<car>:~/ILC2ILC/ilc_f1tenth_hw/scripts/
# then rebuild or re-install ilc_f1tenth_hw as usual, and restart car_stack.sh and car_runner.sh
```

New parameters, all optional: `optitrack_node` `jump_tol_m` (0.05) and `jump_max_speed` (7.0). `net_runner`'s
`mocap_topic` now defaults to `/optitrack/pose_ok`; pass `-p mocap_topic:=/dl_f1tenth/pose` to get the old behaviour.

**Limit:** if the first frame after a 0.3 s dropout is itself a mis-solve, the gate locks onto it and rejects the true
frames for up to 0.3 s before re-accepting. The runner's 0.5 s mocap-loss stop still bounds this.
