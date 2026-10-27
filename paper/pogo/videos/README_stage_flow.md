# Pogo stage + flow videos (2026-10-27; log/pogo_servo/video_stage_flow.py)

Offset stick (calibration offset 0.05 rad, outside the DR family), seed 0, servo leg, the frozen PF stage (6 batches of
6 training trials, then the final network). Each video, 1127 frames at 30 fps (~38 s), a 2 s pause after every batch.

- Left: the 6 training trials of batch k, re-flown exactly (every re-flown trial cost J matches the stage's log to 3
  decimals; the per-iteration networks are from runs/video/<arm>_PF, the frozen stage rerun with --save-at 1..5, which
  reproduces the frozen stage's evaluation except ours+DR's spring stick, 0.72 vs 0.78 validation success).
- Right: the stick's apex-to-apex flow map at the goal (0.8 m/s, 0.62 m) under pi_k; background = deviation from the
  sim field under pi_0; orange = the 4 validation drops' 20-hop orbits and their fixed point; dashed box = tolerance.
- Below: fixed-point error and field deviation per iteration (tolerance units). The field deviation does not go to 0:
  it is measured against the SIM field, and the stage corrects toward the real stick's goal, not back to the sim.

| video | policy type | fixed-point error at pi_0 (tol) | at pi_6 (tol) | trials on target, batch pi_0 | final network |
|---|---|---|---|---|---|
| stage_flow_nom_s0_real_offset.mp4 | Ours (nominal sim) | 3.14 | 0.04 | 0/6 | 6/6 |
| stage_flow_drP_s0_real_offset.mp4 | Ours + DR | 1.90 | 0.84 | 0/6 | 4/6 (falls at goal 0.0 m/s, 0.58 m) |
| stage_flow_ppo_dr_s0_real_offset.mp4 | PPO + DR, our stage | 1.20 | 0.18 | 0/6 | 6/6 |
