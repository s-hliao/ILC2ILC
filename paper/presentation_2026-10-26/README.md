# Presentation media, 2026-10-26: the stage iteration by iteration

Every video: one representative case, every iteration's run kept on screen as a ghost in its own colour (light → dark
blue = early → late), the iteration being flown drawn on top, a pause with the result after each iteration
("π_k flown → update → π_k+1"), and the task error by iteration filling in. 1600 × 900, 30 fps.

| file | what | iterations |
|---|---|---|
| media/pogo/videos/pogo_iters_real_offset_nom_s0.mp4 | pogo, offset stick (0 % zero-shot), goal 0.6 m/s / apex 0.74 m; servo leg | π₀..π₄: training goals on target 0 → 1 → 6 → 6 → 6 of 6 |
| media/pogo/videos/pogo_iters_real_mass_nom_s0.mp4 | pogo, mass stick | π₀..π₄ |
| media/pogo/videos/flow_iters_pogo.mp4 (+ .gif in the same folder) | the apex-to-apex flow after each iteration, offset and mass sticks, vs the sim's | fixed point 14.7 → 0.02 tol (offset), 5.1 → 0.02 (mass); field deviation near the goal 4.2 → 0.7 / 1.7 → 0.5 tol |
| media/car/videos/car_iters_real_mu.mp4 | drift car, low-friction car, figure-8 (each iteration's network re-flown 2 laps) | RMS lateral error 16.7 → 15.9 → 6.0 → 6.2 → 3.8 → 3.7 cm (plan eased to the friction-0.8 ladder rung from π₂) |
| media/car/videos/car_iters_real_nom.mp4 | drift car, nominal car, figure-8 | 9.2 → 5.7 cm |
| media/quadruped/videos/quad_iters_real_s1_g0.575_0.15.mp4 | quad s1, 0.575 m onto a 15 cm box (stage_c_drCft_s2 recipe seed 9004) | landing error 12.3 → 12.0 → 2.7 → 1.0 cm |
| media/quadruped/videos/quad_iters_real_r4_g0.450_0.00.mp4 | quad r4, flat 0.45 m (stage_sbfCft_s2 avg4_it8 seed 9001) | 15.3 → 3.8, then 3-6 cm: one big correction, then the jump-to-jump noise floor |

Caveats for the talk:
- Pogo and car are sim-to-sim "hardware" stages; the cases are the clear ones (pogo offset/mass stick; car mu).
- The pogo flow: the fixed point and the field around the goal return to the sim's; far from the goal the field stays
  ~1.4 tol off (the stage only corrects where the task is).
- The quad cases are among the best of the stage runs (picked by their per-trial landing errors), not typical: the
  quad stage gains only +0.04..+0.09 success over random steps on average (noise-limited; NOTES 2026-10-25/26). The r4
  video shows exactly that: one correction, then scatter of 3-6 cm at a fixed network.

Scripts (absolute paths to the workspace logs; run from log/dilc/presentation_2026-10-26/media): scripts/iter_style.py (shared look), video_pogo_iters.py, video_car_iters.py, video_quad_iters.py, flow_iters_pogo.py.
