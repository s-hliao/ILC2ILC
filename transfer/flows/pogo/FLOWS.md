# Pogo flows: apex-to-apex return map (2026-10-22, matcha; validation only)

Spec: `../SPEC.md`. Code: `flows_pogo.py` (compute / aggregate / portraits). Stage reruns: `run_stage.sh`. The rest: `run_remaining.sh`.

## What was computed
- **The map.** F(h, vx | g) takes the flight apex to the next apex.
  - The flight starts at the apex on a control-period boundary. The next apex is ballistic from liftoff.
  - sim is the nominal model (common.substep, 0.2 ms steps). real is the real_pogo sticks (0.1 ms, damping 15 N s/m, plus each stick's perturbation).
  - The MLP and FADA policies are memoryless, so their map is exact.
  - RMA has a 50-period history (longer than a hop), so each lane first flies one warm-up hop (see caveats).
- **Units.** Everything is in tolerance units: 2 cm of apex height and 0.1 m/s of speed.
- **field_err.** The mean of |V^real_{pi_i} − V^sim_{pi_0}| over a tube.
  - The tube holds the apexes that sim pi_0 and real pi_0..pi_6 visit from the 4 validation drops per goal over 20 hops: 100 sampled states plus 100 perturbed by ±1 tolerance.
  - "far" uses ±3 tolerances.
- **task_err.** |fixed point − goal|, where the fixed point is the mean apex of hops 10–20 from each drop.
- **Goals.**
  - IN: the 6 TRAIN_GOALS.
  - HULL: 9 goals.
  - OUT: 7 goals.
  - HULL and OUT are the 16 goals of the validation seed 301, split by whether they lie inside the IN convex hull (with a 0.10 margin, normalized goal box). The test seeds were not touched.
- **Start policies.**
  - **ours, ours+DR (drNomInit), PPO+DR.** The frozen PF stage (6 iterations × 6 trials, `--train-pert 1 --pert-fixed 1`) was re-run on the CPU with an append-only `--save-at 1..5` flag in `hw_stage.py`. The reruns reproduce the original val success to ±1 run (for example ours+DR s0 100/100, ours s0 90.6/90.1).
  - **RMA.** Zero-shot only, because the stage adapts MLPs only.
  - **FADA.** Its own LoRA adaptation, where it4 = hw24 (24 trials).
  - Every start has 3 seeds × 6 robots = 18 units, with 95 % bootstrap CIs.
- **No plan in this domain.** The reference flow is V^sim_{pi_0}. A second reference is the stage's own target, F(x) = g (`goal_err`). It is in the caches for RMA only, and `run_remaining.sh` recomputes it for the rest.

## Main numbers (it0 → it6; FADA it0 → it4; mean [95 % CI])

| start | set | field_err | task_err (fixed point) | fixed point within tolerance |
|---|---|---|---|---|
| ours (nominal) | IN | 2.29 [1.79, 2.90] → **0.57** [0.49, 0.67] | 2.37 [2.08, 2.65] → **0.45** [0.31, 0.61] | 1 % → 89 % |
| | HULL | 2.29 → 0.55 [0.47, 0.64] | 2.35 → 0.45 [0.32, 0.61] | 2 → 93 % |
| | OUT | 1.92 → 0.48 [0.39, 0.58] | 2.21 → 0.36 [0.31, 0.42] | 6 → 100 % |
| ours + DR | IN | 2.15 [1.59, 2.81] → 1.03 [0.90, 1.16] | 1.32 [0.96, 1.70] → **0.36** [0.30, 0.43] | 45 → 97 % |
| | HULL | 2.14 → 0.98 | 1.31 → 0.37 | 45 → 96 % |
| | OUT | 1.82 → 0.84 | 1.29 → 0.41 | 50 → 97 % |
| PPO+DR + our stage | IN | 2.03 [1.55, 2.60] → 1.45 [1.32, 1.58] | 0.89 [0.62, 1.21] → **0.19** [0.14, 0.25] | 69 → 100 % |
| | HULL | 2.02 → 1.45 | 0.88 → 0.22 | 68 → 100 % |
| | OUT | 1.68 → 1.15 | 0.77 → 0.23 | 73 → 100 % |
| FADA (own adaptation) | IN | 2.29 → 1.59 [1.45, 1.72] | 1.29 → 0.70 [0.59, 0.81] | 50 → 83 % |
| RMA zero-shot | IN / HULL / OUT | 1.80 / 1.86 / 1.16 | 1.74 / 1.89 / 1.51 (unreliable, see caveats) | 45 % |

- **The sim pi_0 fixed point is not at the goal for the DR-trained starts.** Mean task_err of sim pi_0 on IN:

  | start | sim pi_0 task_err (IN) |
  |---|---|
  | ours | 0.21 |
  | ours + DR | 0.66 |
  | PPO+DR | 2.0 |
  | FADA | 2.2 |

- **Robots inside vs outside the DR family** (IN, field_err it0 → it6):

  | start | inside (nom / mass / spring) | outside (act / offset / prog) |
  |---|---|---|
  | ours | 1.88 → 0.54 | 2.71 → 0.60 |
  | ours + DR | 1.63 → 0.99 | 2.67 → 1.07 |

  - The offset stick is the worst corruption: field_err 4.8–5.1 at it0. It falls to 0.8–1.5 by it6.

## Reading
- **The real sticks corrupt the flow.** dV_0 is 1.7–2.3 tolerances on the tube for every start.
  - For ours (nominal), the fixed point sits 2.4 tolerances off the goal: apex about 4.5 cm low, which is the missing leg damping.
- **For ours (nominal), the stage re-corrects the flow back toward the designed sim flow.**
  - field_err falls 2.3 → 0.57, about 4×. Most of the drop comes in the first 2 iterations.
  - The fixed point comes back to the goal: 2.4 → 0.45, 1 % → 89 % of goals within tolerance.
- **The correction holds equally on HULL and OUT validation goals** (OUT 1.9 → 0.48). On pogo, the network carries the correction over the whole goal box.
  - "far" states (±3 tolerances off the tube) also improve, but less: ours 2.4 → 0.96.
- **For the DR-trained starts, V^sim_{pi_0} is the wrong yardstick.** Their nominal-sim flow is itself off the goal.
  - The stage moves the real flow to the goal, not to that sim flow. The fixed-point error falls 1.3 → 0.36 (ours + DR) and 0.89 → 0.19 (PPO+DR).
  - field_err to sim pi_0 falls only partly: 2.1 → 1.0 and 2.0 → 1.45.
  - For PPO+DR on real_nom it even grows, 1.06 → 1.52.
  - The goal-target flow error (`goal_err`) is the measure that fits these starts. It is not yet computed for them (see Status).
- **The task link agrees with the real evaluation.** The fraction of fixed points within tolerance follows the val success rates:

  | start | fixed points within tolerance | val success |
  |---|---|---|
  | ours | 1 → 89 % | 0 → 95 % |
  | ours + DR | 45 → 97 % | 54 → 99.5 % |
  | PPO+DR | 69 → 100 % | 77 → 98.8 % |

  The exception is RMA (see caveats).
- **FADA's own adaptation corrects less than our stage at the same 24 trials:** task_err 0.70 vs 0.36–0.45 at it4.

## Caveats
- **RMA's map is not faithful.**
  - Its history state does not fit a 2-D apex map. The warm-up hop helps but is not enough.
  - The map gives a fixed-point error of 1.1–2.9 on the high-apex goals, while the continuous 10-s val runs give an RMS apex error of 0.2–0.5 cm (99.8 % success).
  - The earlier no-warm-up version is in `cache_old/RMA_s*_nowarm.npz`.
  - Treat RMA's flow numbers as indicative only.
- **The control-period phase is fixed at the apex.** In continuous hopping it varies, which adds a small sampling noise to the map.
- **Iteration-to-iteration CIs are over the 18 robot-seed units.** Goals are averaged inside each unit, so the units are not independent across goals.

## Status / remaining (the matcha jobs were killed 2026-10-22 ~19:50)

**Finished:**
- the 9 stage reruns, `runs/{nom,drNomInit,ppo_dr}_s{0,1,2}_PF/` (it1..it5 + final checkpoints, final val eval);
- all 15 flow caches, `cache/<start>_s<seed>.npz`:
  - RMA is v2: raw maps plus goal_err, with warm-up;
  - the other 12 are v1: aggregates only;
- `summary.json`, `summary_extra.json`, `fig_field_err.png`, `fig_task_err.png`.

**Missing:**
- the phase portraits (`fig_portraits_oursDR.png`, `fig_portraits_ours.png`; the processes were killed);
- `goal_err` and the raw maps for the 12 v1 caches, and with them `fig_goalflow_err.png`.

**`run_remaining.sh` (idempotent).** It runs these steps, using ≤ 8 processes and `JAX_PLATFORMS=cpu`:
1. `run_stage.sh`, which skips every run that is done.
2. `flows_pogo.py compute` with `FLOWS_REDO_V1=1`. This moves the 12 v1 caches to `cache_old/*_v1.npz` and recomputes them. Set it to 0 to keep them.
3. `aggregate`, which also writes `fig_goalflow_err.png` once every cache is v2.
4. The two portrait figures.

**Runtime on loaded matcha:**
- a 7-iteration start: about 90 min per (start, seed) job;
- FADA: about 30 min per job;
- RMA: about 20 min per job;
- portraits: about 10 min.

That is roughly 2.5–3 h with 8 processes, and less on an idle machine. Portraits only (`FLOWS_REDO_V1=0`) take about 10 min.

**Inputs to copy.** Paths are absolute under `/home/henry/ilc_ws`: `POGO` in `flows_pogo.py`, and `cd` / `OUT` in `run_stage.sh`. Mirror that path or edit those lines.
- `src/ilc_pogo/*.py`, including `hw_stage.py` with `--save-at`;
- `src/ilc_pogo/runs/{nom,drNomInit}_s{0,1,2}/policy_pick.npz`;
- `src/ilc_pogo/runs/ppo_dr_s{0,1,2}/policy_final.npz`;
- `src/ilc_pogo/runs/rmaR1_s0/`, `rmaR1full_s{1,2}/` (`policy_final.npz` and `rma_adapt.npz`);
- `src/ilc_pogo/runs/v3/fadaF2_s{0,1,2}_{zs,hw24}/`;
- this directory, including `runs/` and `cache/`.
