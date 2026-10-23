# media/ (2026-10-23): slides material, one folder per domain, one look (log/dilc/flows/portrait_style.py)
Status 08:20: complete. Videos are H.264 (crf 28, <= 1280 px, 24 fps, faststart), each < 10 MB, GitHub-ready.
A copy in the paper layout (pogo/ car/ quadruped/ x figures/ videos/ + tables.md) is in src/paper/presentation_2026-10-23/media/ (not git-added).

## Headline numbers (tables)
- Pogo (frozen test 3301, 3 seeds): ours 7.6 -> 90.8 % with the stage (36 trials); ours+DR 48.8 -> 97.6 %; PPO+DR 73.6 -> 97.2 %. Direction control: flipped Jacobian 0.0 %, open-loop 5.2 % vs 79.7 % (validation, 24 trials).
- Car (validation 301, 3 seeds, 8 cars): ours 40.8 -> 89.9 % b25 with the 10-lap ladder stage (test 2701: 91.1 %), crashes 45.8 -> 1.9 %; PPO+DR + our stage 85.9 %; ours - PPO+DR (both + ladder) +4.0 [-6.8, +16.7] pts (n.s.).
- Quad (validation, at 08:19: drB_s2 36 paired cells, drCft_s2 21): stage vs random control, success +1.6 [-0.9, +3.9] pts (n.s.) / +7.3 [+2.2, +12.9]; capped score -1.05 [-1.53, -0.61] / -0.99 [-1.51, -0.48] (capped = POST HOC); raw mean score n.s. on drB. r5 alone: +2.8 [+0.0, +5.6] (drB) / -3.8 [-7.5, +0.0] (drCft). More seeds finish ~08:45: rerun make_tables.py.

## Tables (make_tables.py; rerun to refresh the quad numbers)
- pogo/tables.md, pogo/table_pogo.png: success on the frozen test seed 3301 (zero-shot vs + stage for 3 starts, baselines) + direction ablations.
- car/tables.md, car/table_car.png: b25 drift success (validation 301, test 2701), ladder stage vs baselines + paired diffs + flipped-Jacobian control.
- quad/tables.md, quad/table_quad.png: pre-registered success, raw and capped score, stage vs random-direction control, paired CIs (drB_s2 + drCft_s2 replication).

## Flow fields (copied from A_current_mu0.6 / B_friction_fix_mu0.45)
- fig_flows_all.png: the three domains' field error vs iteration side by side.
- pogo/fig_portraits_{ours,oursDR,PPODR}.png: apex-to-apex flow, sim pi0 | real pi0 (corrupted) | pi1, pi2, pi6; the field re-corrects.
- pogo/fig_goalflow_err.png, fig_task_err.png: field / fixed-point error vs iteration (IN / HULL / OUT).
- car/portrait_<start>_<car>.png: 0.5 s closed-loop flow slices; car/field_err.png, task_err.png: the field barely moves while the lap error falls.
- quad/fig_goalfield_*.png: landing field over the goal plane per iteration (ours, PPO+DR, random control, clean twins, mean); fig_goalfield_r5*.png: r5 under friction 1.0 / 0.59 / the 0.45 fix.

## Stage videos (sim-to-sim hardware stage, batch by batch; blue = this batch, gray = the previous batch; one style)
- quad/stage_video_real_r5.mp4, stage_video_real_s1.mp4 (22 s each, 0.5x real time; r5 = the heavy low-friction robot, s1 = a good one): LOGGED trials (tr/recipe_sd9001, no re-fly); the 6 training jumps of each batch side by side (side view, trunk + legs, box, goal star), batch result line (within 5/3 cm, falls, median landing error). *_midframe.png / *_lastframe.png = stills.
- pogo/stage_video_real_offset_nom_s0.mp4 (28 s, real time; batch 0: 0/6 trials on target, final network: 6/6): stage trials flown again exactly (the stage logged no states; re-fly reproduces the logged trial costs J to 3 decimals); offset stick (outside the DR family), 6 trials per batch side by side, batches pi0..pi5 + the final network. *_batch0.png / *_final.png = stills.
- car/stage_video_v3drBtL_s0_ladR_val_its.mp4 (~33 s, 3x real time; batch 0: 2 crashes of 8, batch 4: 0 crashes, mean RMS lateral error 5.5 -> 4.0 cm; gray = the previous trial on the same track): stage trials flown again (re-fly matches the logged RMS lateral error within 0.3 cm on every trial, crashes identical; start perturbation set to 0); all 8 cars' trials of each batch at once, plan line dashed, ladder rung in the panel title. car/stage_trials_*.npz = the re-flown trajectories.

## Animated flow fields (one frame per stage iteration, 1.2 s each; same style as the static portraits)
- pogo/flow_anim_pogo.{mp4,gif}: sim pi0 vs the offset / mass sticks under pi_k, k = 0..6 (ours, seed 0); *_first.png / *_last.png.
- quad/flow_anim_quad.{mp4,gif}: sim pi0 vs the 4 robots under pi_k, k = 0..4 (ours+DR start, seed 9001; logged fields).
- car: the 0.5 s state field (portrait_*.png) is NOT the map the stage corrects: its gap is mostly in the drift states (vy, r), which the stage's cost (e_y + pace) never asks for, so it barely moves. The lap-level checkpoint return map is the one the stage corrects:
- car/fig_car_returnmap.png (454 KB): the car's trial-level field (analogue of pogo's apex map / the quad's goal field) in (e_y / 0.10 m, speed error / 0.30 m/s); orange loop = the repeating error (mean state at 8 checkpoints/lap, per track), arrows = smoothed segment map, star = plan; ours seed 0, pi_0..pi_5 re-flown from the 3 validation starts (seed 301) x 2 laps. Repeating error 0.70 -> 0.52 (real_nom), 0.58 -> 0.30 (real_mu), 0.72 -> 0.57 (real_lag); all 8 cars 0.55 -> 0.44; the axle-friction cars ax:mu=0.6/0.7/0.9 do not contract (their gain is the ladder rung). Script: log/dilc/flows/car/car_returnmap.py.
- car/fig_car_returnmap.png: THE car flow field -- the lap-level checkpoint map in the task coordinates (lateral error, speed error; tolerance units) smoothed from the actual trials; the repeating error orbit contracts toward the plan: real_mu 0.58 -> 0.30, real_nom 0.70 -> 0.52, real_lag 0.72 -> 0.57, all 8 cars 0.55 -> 0.44 (ax:mu=0.7 / 0.9 do not contract).
- car/fig_car_localmap.png: the same map FLOWN around the actual trials (multi-body car started from a 3x3 grid of (lateral, speed) offsets at each checkpoint state of the trial, one segment ahead); mean distance of the next checkpoint state: real_nom 0.62 -> 0.53, real_mu 0.52 -> 0.46, real_lag 0.69 -> 0.61 tol (modest, 12-15 %).
- (why the 0.5 s portraits barely move: that field is all 5 error states vs sim pi_0's; most of the gap is in sideslip / yaw rate, which the stage's cost (lateral error + pace) never targets -- the stage corrects the lap-level task map, above.)

## Scripts (all in media/): make_tables.py, video_quad.py, video_pogo.py, car_fly.py + video_car.py, flow_anim.py

## File sizes (media/, 08:19:27)
```
     81321  car/field_err.png
    471917  car/portrait_ppoL_task_s0_ladR_val_its_ax:mu=0.6.png
    509222  car/portrait_ppoL_task_s0_ladR_val_its_real_mu.png
    500561  car/portrait_rmaL_task_s0_ladR_val_its_ax:mu=0.6.png
    519235  car/portrait_rmaL_task_s0_ladR_val_its_real_mu.png
    453247  car/portrait_v3drBtL_s0_ladR_val_its_ax:mu=0.6.png
    481029  car/portrait_v3drBtL_s0_ladR_val_its_real_mu.png
   1217526  car/stage_trials_v3drBtL_s0_ladR_val_its.npz
    122165  car/stage_video_v3drBtL_s0_ladR_val_its_batch0.png
    144148  car/stage_video_v3drBtL_s0_ladR_val_its_last.png
    685453  car/stage_video_v3drBtL_s0_ladR_val_its.mp4
    162607  car/table_car.png
      1328  car/tables.md
    111672  car/task_err.png
    234838  fig_flows_all.png
    193142  pogo/fig_goalflow_err.png
    664048  pogo/fig_portraits_oursDR.png
    634440  pogo/fig_portraits_ours.png
    662796  pogo/fig_portraits_PPODR.png
    206239  pogo/fig_task_err.png
    235628  pogo/flow_anim_pogo_first.png
   1059815  pogo/flow_anim_pogo.gif
    283264  pogo/flow_anim_pogo_last.png
    145131  pogo/flow_anim_pogo.mp4
    109154  pogo/stage_video_real_offset_nom_s0_batch0.png
    123214  pogo/stage_video_real_offset_nom_s0_final.png
   1380394  pogo/stage_video_real_offset_nom_s0.mp4
    181522  pogo/table_pogo.png
      1448  pogo/tables.md
    858249  quad/fig_goalfield_mean.png
   1019960  quad/fig_goalfield_ourspDR_clean_twins.png
   1004741  quad/fig_goalfield_ourspDR.png
   1053964  quad/fig_goalfield_PPOpDR_p_our_stage.png
    692227  quad/fig_goalfield_r5.png
    692227  quad/fig_goalfield_r5_with_fix045.png
    997426  quad/fig_goalfield_random-dir_control.png
    192014  quad/flow_anim_quad_first.png
    208843  quad/flow_anim_quad.gif
    185103  quad/flow_anim_quad_last.png
     76117  quad/flow_anim_quad.mp4
    116468  quad/stage_video_real_r5_lastframe.png
    106667  quad/stage_video_real_r5_midframe.png
    732765  quad/stage_video_real_r5.mp4
    120505  quad/stage_video_real_s1_lastframe.png
    108795  quad/stage_video_real_s1_midframe.png
    780023  quad/stage_video_real_s1.mp4
    178117  quad/table_quad.png
      1495  quad/tables.md
```

- car/fig_car_xyfield.png: the car's correction field in its own x-y space on the figure-8 (multi-body car started 6/12 cm off the racing line at 44 points, offset after 1 s; orange = the stage's actual figure-8 trials, iterations 1 and 3). The local feedback field is the same at every iteration (offset kept 0.31-0.34 of the start on real_nom, 0.41-0.43 on real_mu): the stage changes the lap, not the local steering response.
- tables.pptx (one native PowerPoint table per slide: copy a table into your deck), tables.html (open in a browser, select, copy, paste), tables_tsv/*.tsv (tab-separated): all 6 tables, built by tables_to_ppt.py from */tables.md.
