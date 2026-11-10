# Figure plan (2026-11-06, revised with the user's structure)

Templates: the pogo pair (paper/final/pogo_budget/budget_figs.py): pogo_budget_sim (zero-shot error vs simulated hours of training)
and pogo_budget_real (error vs real rollouts per robot, 0 = zero-shot). Metric: the continuous normalized score (car: max(rms e_y /
10 cm, (1 - pace) / 0.1); quad: max(|ex| / 5 cm, |ez| / 3 cm); failure = 5), mean and 95 % cluster-bootstrap CI over (seed, car or
robot, goal). Car numbers on the FIXED simulator. "Real" = the simulated real cars / robots (sim-to-sim), except figure 7.

| # | figure | arms / content | panels | data now | to do |
|---|---|---|---|---|---|
| 1 | CAR, simulation training (as pogo_budget_sim) | zero-shot error on the real cars vs simulated hours: DistILC_sim, DistILC_sim + DR, PPO, PPO + DR, RMA, FADA; 3 seeds each, all from the same BC clone (freeBCN) | plastic (12 cars) / rubber (2 cars), held-out peanut | PPO + DR (freePPON) and RMA (rmaN): 3 seeds, checkpoints only at it 100 / 200 / 300; the other four arms: no current runs | train every arm with checkpoints log-spaced in simulated hours (same configs as the deployed networks); evaluate each checkpoint zero-shot (ladder_eval) |
| 2 | CAR, real laps (as pogo_budget_real) | error vs real laps per car: PPO + DR + DistILC_hw, RMA + DistILC_hw, DistILC_sim (+ DR) + DistILC_hw, PPO + DR + PPO on the car (privileged critic), FADA + adaptation, RMA zero-shot (marker) | plastic / rubber | ours: the headline runs' saved iterations it 0-3 = 0 / 2 / 4 / 6 laps (9 plastic / 18 rubber runs per network; few-shot, NO long practice -- user: at most 8 laps); 8 laps only if the cars are rerun with --iters 4 (e.g. after the step-size screen); baselines: Oct runs on the OLD setup only | port ppo_real_car.py and fada_real_car.py to the current networks, cars and score; stage runs on the DistILC_sim networks of figure 1 |
| 3 | CAR, domain-specific: the flow | flow fields with the policy-change shading (car_flow_*.pdf), 4 videos (full laps, side log, trace panel, dotted previous lap) | - | DONE on the current networks | re-render only if the step-size screen changes the networks; captions |
| 4 | QUAD, simulation training (as pogo_budget_sim) | zero-shot error on the 4 sim robots vs simulated hours: DistILC_sim, DistILC_sim + DR, PPO, PPO + DR, RMA, FADA; damping 0.1, memoryless observation | held-out goals / perturbed starts | PPO + DR m0: init + it 100 + it 300, one seed; fl01z/ours_s2 (our learner, with memory); RMA / FADA at d = 0.1: none | train every arm, 3 seeds, dense checkpoints (a quad run takes 2-3 h on one GPU); evaluate with planeval nominal + perturbed |
| 5 | QUAD, real jumps (as pogo_budget_real) | error vs real jumps per robot: PPO + DR + DistILC_hw (the picked arm, it 0-6 = 0-36 jumps), DistILC_sim (+ DR) + DistILC_hw, PPO + DR + PPO on the robot, FADA + adaptation, RMA zero-shot | held-out goals / perturbed starts | ours: it 0 and it 6 evaluated (R / RS / RX); baselines: Oct runs on the old networks only | evaluate it 2 / 4 of the pick; port quad_real_ppo.py, fada_adapt.py, quad_eval_ckpts.py to d = 0.1 + memoryless + the score |
| 6 | QUAD, domain-specific: perturbed starts + videos | heatmap: zero-shot, sim-rollout Jacobian, R, RS, RX (+ memory-network panel FRAM / FRMR) x nominal + 8 start conditions; videos: zero-shot vs stage jumps, nominal and perturbed stances | - | heatmap DRAFT (pert_heatmap.py); video quad_d01_stage_s1.mp4 is the OLD arm | final heatmap after the R / RS / RX second copies finish (~14:45) and FRMR (~18:00); re-render the quad videos from the pick |
| 7 | HARDWARE, 5 experiments | car: PPO + DR plastic, PPO + DR rubber, RMA plastic, RMA rubber; quad: jumping (PPO + DR + stage). Improvement over each network's own zero-shot, paired; error over real laps / jumps, before / after, track and landing overlays | one panel per experiment | car: car_hw_figs.py template on the desk rehearsal; quad: none | quad template on the hwsm sim-robot sessions; fill from the sessions |

## Compute and order (target: all simulation figures by Nov 10)
- Today: quad pick (~14:45), heatmap, car re-score, step-size screen (~16:30); write the training configs for figures 1 / 4 and port
  the baseline scripts; launch the figure 1 / 4 training on the GPUs freed after FRMR (~18:00): car 6 arms x 3 seeds (~20-60 min
  each), quad 6 arms x 3 seeds (~2-3 h each) on ube 3 GPUs + matcha 4 GPUs.
- Nov 7: car training done -> figure 1; launch the figure 2 real-lap arms; quad it 2 / 4 evaluations; figure 3 / 6 videos.
- Nov 8: quad training done -> figure 4; launch the figure 5 real-jump arms; figure 2.
- Nov 9-10: figure 5. Figure 7 the day after each hardware session.

## Open questions
- (user 2026-11-06) Few-shot only: ours is shown up to 8 real laps per car; the LONG (16-lap) study is not in the paper.
- Panels: plastic / rubber (car) and held-out / perturbed (quad) as above, or pogo's coverage tiers (inside DR / extrapolated / outside)?
- Figure 1 / 4 PPO + DR and RMA: the deployed networks have only 3 saved checkpoints. Retrain them with dense checkpoints (same configs
  and seeds; the deployed networks marked as their endpoints) -- recommended -- or plot only the saved checkpoints?
- DistILC_sim for the car: the October car learner (v3drBt) predates the current plans and cars; it is retrained in the current setup.
- Hardware: has the car session (Nov 2-8 in the handoff) happened?

## Decisions (2026-11-06 ~14:00)
- Dense checkpoints: YES (user) -- every arm of figures 1 / 4 retrained, same configs and seeds, log-spaced checkpoints in simulated
  hours; the deployed networks marked as the endpoints.
- Panels (my recommendation, awaiting the user): car figures 1 / 2 = the pre-registered coverage tiers of mb_car.py (2026-10-29, 12
  cars each: A inside the DR range, B extrapolated 1.5 half-ranges, C outside the DR class) + a 4th panel "the hardware tires"
  (12 plastic + 2 rubber cars of the main tables; friction -33 % / +67-83 % vs the plan, outside the 20 % DR half-range). Quad figures
  4 / 5 = held-out goals / perturbed starts (4 robots: too few for tiers; none were pre-registered).

## Figure 6 revised (user 2026-11-06 ~14:10): the heatmap compares METHODS, sim and sim-to-sim arms (not the stage variants)
Rows: sim-trained zero-shot (figure 4 finals: DistILC_sim, DistILC_sim + DR, PPO, PPO + DR, RMA, FADA) and sim-to-sim after 36 real jumps
per robot (figure 5: PPO + DR + DistILC_hw = the picked arm, DistILC_sim + DistILC_hw, DistILC_sim + DR + DistILC_hw, PPO + DR + PPO on
the robot, FADA + adaptation). Columns: nominal + the 8 start conditions (goals < 10 cm, 4 seeded repeats, the same seeds for every row).
Cell colour = absolute score (sequential); adapted rows also show the change vs their own zero-shot (* = CI excludes 0). Every final
network gets the FULL perturbed evaluation (deploy.py's start / final for the stage arms; fada_adapt.py --perturbed; quad_eval_ckpts.py
with the perturbed set for PPO on the robot and RMA zero-shot). Ready with figure 5 (Nov 9-10). The R / RS / RX / FRAM / FRMR heatmap
(pert_heatmap.py) moves to the supplementary as the stage-variant ablation.

## Figure 6 revised again (user ~14:30): retargeting onto perturbed stances (stage_fl/RETARGET_1106.md)
Columns: the 5 hardware-realizable stances (blk15, crouch, noseup, nosedn, tall) + nominal. Rows: each sim-trained network zero-shot and
after retargeting from that stance -- our stage (arm R, --train-cond) on PPO + DR, RMA, DistILC_sim, DistILC_sim + DR; FADA with its own
adaptation (--train-cond). 4 robots x 1 stage seed per cell (100 runs, jobs_RT.json on matcha; RMA / FADA / DistILC_sim wait for figure 4's
seed-0 networks). Cell: score from that stance, change vs the network's own zero-shot, * = CI excludes 0; nominal-stance cost alongside.

## RMA with the ORACLE latent (user 2026-11-07 ~01:10: "add rma with oracle latent to the ablations, and have them as the 'ultimate
## performer' in sim-to-sim transfer, can have a dashed line"), written before any result
The RMA teacher's actor fed the TRUE latent of each test system (no adaptation module, no real data): not deployable, a reference.
- Car (figures 1-2 panels): rmaN_s<s> actor (sim_jax's DR, ppo_dr.py --mode rma) with z = zvec of the car's own parameters mapped onto
  the DR keys (mb_car.raw_car: mass -> m, mu -> muf, ky -> Cf, kt, tau -> tau_s, d_off, d_delay, i_delay; normalized by ppo_dr.py's
  ZNOM / ZSC). Effects outside the DR keys (multi-body geometry, sensing presets, wear, the rear-axle split) are invisible to z -- the
  oracle sees exactly what the teacher was trained to read. Scored as figure 2's zero-shot points (ladder_eval, held-out plan, seed 701,
  5 laps) -> a dashed horizontal line per panel.
- Quad (figures 4-6): the figure-4 teacher rmat_s<s>/it300 with z = JumpEnv.dyn_from_cond of each robot's condition string (the stance
  perturbations of figure 6 are not dynamics: z unchanged), flown on the robots (planeval held-out goals, perturbed starts) -> dashed line.

### RMA oracle-latent RESULT (2026-11-07 ~02:25, MEASURED; rma_oracle_fig2.sh, ladder_eval 5 laps seed 701, held-out peanut)
mean score, oracle (teacher + true z) vs RMA zero-shot (teacher + adaptation module's z_hat; fig1_eval.json, same protocol):
  tier A  0.468 / 0.531 / 0.415  vs 0.501 / 0.546 / 0.416   (s0 / s1 / s2)     oracle slightly better
  tier B  0.532 / 0.604 / 0.489  vs 0.585 / 0.642 / 0.518                      oracle better
  tier C  0.833 / 0.988 / 0.837  vs 0.733 / 0.858 / 0.735                      oracle WORSE
  plastic 1.022 / 1.016 / 1.073  vs 0.886 / 0.960 / 0.950                      oracle WORSE
  rubber  1.185 / 1.291 / 1.242  vs 0.580 / 0.826 / 0.534                      oracle WORSE
The true latents of the plastic / rubber cars lie far outside rmaN's DR box (friction z: plastic -1.67 .. -3.0, rubber +3.3 / +4.2;
the DR covers [-1, 1]; harsh_steer tau z 3.8, d_off z 3.3). So the teacher fed the TRUE z extrapolates, while the adaptation module's
z_hat (trained on in-box histories) stays in-range. INFERRED (not tested): out-of-coverage extrapolation of the actor in z.
Consequence: the oracle-latent RMA is an upper reference only inside DR coverage (tiers A / B); on tier C / plastic / rubber it is NOT an
upper bound. Reported as measured; the dashed line stays as the user specified, with this caveat in the caption. Decision for the user:
keep, or add the alternative they raised (PPO trained on each car's true dynamics).

### RMA oracle-latent v2: a DEPLOYMENT-COVERING teacher (user 2026-11-07 ~02:35: "train a separate rma that directly covers deployment
### dynamics instead of the original dr set"), written before any result
rmaCov_s<s> = ppo_dr.py --mode rma with rmaN_s<s>'s exact arguments (init freeBCN_s0, 300 iterations, 1024 lanes, same plans / reward / PPO
settings) except the DR family: sim_jax F1T_DR_BOX = the union box of all 50 deployment cars + ~5 % margin: mass x0.6-1.4 (Iz alike),
friction x0.38-1.93 log-uniform, tire stiffness x0.68-1.32, motor constant x0.68-1.32, servo tau 0.015-0.155 s, steering offset +-0.105 rad,
delays 0/1 as before. Checked: all 50 cars' z inside the sampled box. Phase 2 skipped (the oracle reads the true z). Oracle = teacher +
true z (rma_oracle_car.py, same ZNOM / ZSC), scored as figure 2's zero-shot points -> runs/fig2base/rmaCovOracle_<panel>_s<s>
(rmaCov_oracle2.sh). Figures: this is the dashed "ultimate performer" line; the rmaN-teacher oracle (v1, above) reported in the supplement.
Not covered by any latent (both versions): multi-body geometry, sensing presets, tire wear, the rear-axle split.

### RMA oracle-latent v3 = the FIGURES' dashed line (user 2026-11-07 ~02:45: "no, it has to cover all deployment dynamics"), before results
rmaD_s<s> (rmaD_train.py, rmaD_latent.py): the RMA teacher trained IN THE DEPLOYMENT SIMULATOR (mb_car via real_car.run: multi-body car,
sensing, tire wear; the evaluation's F1T settings) over EVERY axis the deployment cars vary, 14-d latent = ppo_dr's 8 + rear-stiffness split,
sensing noise / delay / hold / velocity lag, tire wear; one random car per episode from a box containing all 50 deployment cars (checked),
the sensing block and the wear nominal w.p. 0.5. PPO (ppo_real_car.py's update, one shared actor / asymmetric critic), 5.05-lap episodes on
rmaN's 6 plans, start perturbation N(0, 0.5), 200 iterations x 128 episodes, init = the planar covering teacher rmaCov_s<s> (new z inputs at
0). Oracle = rmaD + each car's true z, scored as figure 2's zero-shot points -> runs/fig2base/rmaDOracle_<panel>_s<s> (rmaD_oracle.sh on
Lambda). Reported with its training cost; checked for a plateau. v1 (rmaN teacher) and v2 (rmaCov, planar covering) -> supplement.
Comparison metric: share of the zero-shot -> oracle gap closed (a method may pass the oracle: one network over the whole box vs a per-car
optimum; proposed check (asked): per-car PPO on the 2 rubber cars).

### RMA oracle in ALL ablations (user 2026-11-07 ~02:55: "just add the rma oracle to all ablations"; no per-car PPO check)
Reference row / dashed line in every comparison, scored by that comparison's own protocol (zero-shot, no stage, no real data):
car: figure 1 (panels A / B / C / plastic / rubber from the rmaDOracle per-car scores), figure 2 (dashed per panel), every car table in the
supplement (SB, refit screen, stage variants: same held-out plans / cars -> the same per-car scores reused); quad: figure 4 / 5 (dashed),
figure 6 heatmap (an oracle column: the oracle flown from the nominal + 8 perturbed stances), quad supplement tables (memory ablation etc.).
Quad oracle = rmat_s<s>/it300 + true z (quad_rma_oracle.py); whether the quad needs the deployment-simulator version (as the car's v3)
is decided from its v1 result vs RMA zero-shot (the car's v1 fell below RMA zs outside coverage) -- reported either way.

### RMA oracle-latent v2 RESULT (rmaCov, planar covering teacher; MEASURED 2026-11-07 ~03:25; s0 | s1 | s2)
tier A 0.554 | 0.626 | 0.528, B 0.624 | 0.685 | 0.600, C 0.786 | 0.901 | 0.731; plastic 1.135 | 1.132 | 1.010; rubber 0.691 | 1.009 | 0.719.
vs v1: better on tier C and rubber, worse on tiers A / B and plastic; still above RMA zero-shot on plastic (0.89-0.96) and on rubber for
s0 / s2 (zs 0.58 / 0.53). The planar teacher with an 8-key latent does not reach a ceiling -> v3 (rmaD, the deployment simulator) stays the
dashed line; v1 / v2 in the supplement. rmaD training (Lambda): reward / step 0.55 -> 0.57-0.58 by it 40-60 of 200, falls ~9 % (noisy:
one random car per episode over the whole box); plateau checked before use.

### QUAD RMA oracle v1 RESULT (MEASURED ~04:15; quad_rma_oracle.py, fig4_eval.sh: planeval nominal / perturbed-stance score, lower better)
s0 r1 5.0/51 r4 26.2/83 r5 3.9/36 s1 12.1/63 (mean 11.8 / 58); s1 mean 7.3 / 52; s2 mean 11.3 / 102. RMA zero-shot (rma_s0_ck300, the
retargeting runs' start evaluation, same protocol): mean 15.0 / 76; PPO + DR zero-shot (ppo_m0): mean 4.6 / 55. The oracle beats RMA but
NOT PPO + DR: not a ceiling. As the car: the teacher trained in MJX, the robots are the ilc_quad MuJoCo simulator (async timing, mocap,
IMU, estimator, ground) with params the 8-key latent does not encode -> a quad v3 teacher trained in the robots' simulator over its own
random-robot family (dilc_execute.sample_condition + the _m500 sensing = how the 4 robots were drawn), latent = all its parameters.

### Per-car PPO ceiling v1 RESULT (MEASURED 2026-11-07 ~11:15; ceil_car.sh, Lambda) -- NOT A CEILING
rmaD_train.py --car, init rmaD_s0, rmaN's 6 training plans, 100 its; held-out peanut (fig2 protocol): plastic mean 1.025 (12 cars, 0 fails),
rubber 0.677 -- WORSE than its own init (rmaD_s0 oracle: plastic 0.941, rubber 0.649; worse on 8 / 12 plastic cars) and than RMA zero-shot
(plastic 0.932, rubber 0.647). Training-plan reward rose (ax:mu=0.4000 train fail 0.52 -> 0.20); the held-out score is mostly pace-limited
(n04_real_nom rms e_y 0.012, pace 0.944 -> 0.556). Inferred (not tested): training-plan PPO does not transfer to the held-out plan.
v2 (ceil2_car.sh, launched ~11:15): same PPO trained ON the evaluation plan (peanut a33 / x120; a reference may see the test plan), 200 its.

### FADA in ALL car results = RMA-matched-data FADA (fadaR; user decision 2026-11-07 ~12:10, runs/screen/FADA_AUDIT_1107.md)
fadaT (leaky) invalid; fadaL (paper recipe, leak fixed) diagnostic only. fadaR_all.sh: figure 1 (it 1 2 5 10 20 40 75 150), figure 2 (all panels).

### QUAD figure 5 with the PAPER METRIC (MEASURED 2026-11-07 ~13:25; normalized max(|ex|/5cm, |ez|/3cm), fall = 5; success |ex|<=5cm & |ez|<=3cm)
(Correction: summary.json's planeval_* = 0.01 landing cost + 20 fell, the training cost -- not the paper metric; numbers quoted from it today were wrong.)
PPO+DR zero-shot nominal 2.17 (13 %) / perturbed 2.98 (9 %). After 36 jumps: R 1.42-1.49 (38-41 %) / 2.59-2.67; RS 1.22-1.37 (39-52 %) /
2.29-2.61; RX 1.03-1.61 (39-60 %) / 2.49-3.19. Zero-shot others: RMA 2.19 (14 %) / 3.04; RMA + true z 2.23-2.49 / 3.05-3.36; FADA full-fidelity
(leaky) it8 2.32-2.56 (0-12 %) / 3.19-3.50; old memoryless FADA (fadaP) 2.92 / 3.94, after its adaptation 3.34-3.77 / 4.87-5.29 (worse).

### DECISION (user 2026-11-07 ~14:05): FADA moves INTO OUR TRAINING REGIME ("we're just using a faster training method, the adaptation was the
main part of the method anyways"); no supplementary paper-recipe FADA, no stage-on-FADA. ONE FADA network per domain for every comparison:
RMA's teacher / DR / plans / phase-2 distillation data, MEMORYLESS input (current state; H 1) like PPO+DR, our stage's networks and GPS;
FADA's adaptation exactly (planner frozen, LoRA on the IDM from one round of FADA-zs rollouts, same budgets). Car = fadaR1 (the a_{t-1}
token: kept or zeroed by the daggerR / daggerRp test, ~14:50). Quad: rebuild the same way (memoryless, the quad RMA's phase-2 data).
GPS: already in our regime (starts from our PPO+DR networks freePPON, the same memoryless MLP observation, same cars / plans / scoring;
its real-lap budget per iteration is its own, plotted on the real-laps axis) -> unchanged.

### FADA in our regime, ADAPTATION (MEASURED ~16:40; fig2 protocol, held-out, it 0 / 8 / 24 / 96 / 384 episodes, mean of 3 seeds)
car (fadaR1): tier 0.451 0.503 0.497 0.521 0.525 (+17 %), plastic 0.781 0.847 0.839 0.820 0.849 (+8 %), rubber 0.317 0.418 0.390 0.438 0.445 (+39 %)
-> FADA's adaptation WORSENS every car panel. quad (fadaRq, paper metric, 4 robots): nominal 2.39 (5 %) -> 2.34 (5 %) at 36 jumps (seeds 0-1 at
24 / 36 so far), perturbed 3.26 -> 3.02 -> flat. Ours (SB, 6 laps): plastic -24 / -33 %, rubber -5 %; quad RS 2.17 -> 1.22-1.37 (39-52 %).
QUAD FADA (fadaRq) adaptation FINAL (MEASURED 17:10, 3 seeds x 4 robots, paper metric): jumps 0 / 6 / 12 / 24 / 36: nominal 2.39 2.36 2.37 2.36
2.37 (success 5 5 4 5 4 %), perturbed 3.26 3.32 3.22 3.12 3.08 (5-6 %). Flat nominal, -6 % perturbed. Ours RS: 2.17 (13 %) -> 1.22-1.37 (39-52 %).

### DECISIONS (user 2026-11-07 ~17:50-18:00)
- Refit settings: "keep it as close to the paper's theory as possible, 3-4 points isn't particularly important" -> SB stays everywhere
  (AW80 -4.2 / AD1 -2.8 pts NOT adopted; REFIT_SCREEN_1106.md reported in the supplement only).
- "yes, no-teacher fada is fine": the FADA in the ADAPTATION figures (2, 5) has NO privileged teacher; it is distilled from the same base
  our stage starts from, so both start equal and only the adaptation differs. Everything else = fadaR1 / fadaRq (RMA phase-2 data, H 1).
  car fadaNP_s<s> (teacher freePPON_s<s>, obs only) and fadaNR_s<s> (teacher rmaN_s<s> actor + its adaptation module on the trajectory's own
  40-step history; real_car.py's loop) -- fada_car_tf.py --teacher-kind ppodr / rma; Lambda fadaN_all.sh (train, fig1 check, fig2 3 panels).
  quad fadaNPq_s<s> (teacher mem/ppo_m0/it300, mem0_R / RS's base) -- fada_quad_tf.py --teacher-kind ppodr; matcha fadaNPq_all.sh (train,
  fig5 adaptation). Figures 1 / 4 (sim sample-efficiency) keep fadaR1 / fadaRq charged with their teacher's sim cost.
- GPS: car fig2 done on all three panels (from freePPON), no improvement; no quad GPS exists.

### QUAD figure 4 FADA (fadaRq) zero-shot, PAPER METRIC (MEASURED 2026-11-07 20:40; fig4/eval/fadaRq_s<s>_it<k>, 3 seeds x 4 robots)
it 1 / 5 / 10 / 25: nominal 2.16 / 2.48 / 2.28 / 2.39 (success 10 / 6 / 7 / 5 %); perturbed 3.15 / 3.32 / 3.13 / 3.26. (PPO+DR zs 2.17 / 2.98.)
Note: 20:15 a duplicate scoring driver (fadaRq_eval.sh, mine) was found writing the same folders as fadaRq_all.sh's loop; killed by PID.
All 48 outputs verified (done marker + valid JSON); the two double-written ones re-run cleanly into fadaRq_s1_it1_rerun: identical (2.919 / 1.946).

### NO-TEACHER FADA, ADAPTATION (MEASURED 2026-11-07 ~22:15; fig2 protocol held-out, mean of 3 seeds; quad paper metric 3 seeds x 4 robots)
car episodes 0 / 8 / 24 / 96 / 384 (change at 384, per seed):
  rubber  fadaNP 0.485 0.506 0.513 0.512 0.522 (+8 %: +13 +5 +5)  | fadaNR 0.427 0.472 0.491 0.468 0.471 (+10 %: +10 -5 +25)
  plastic fadaNP 1.375 1.324 1.308 1.311 1.312 (-5 %: -3 -8 -3)   | fadaNR 0.861 0.852 0.845 0.828 0.852 (-1 %: -3 -4 +4)
  tier    fadaNP 0.736 0.742 0.730 0.734 0.744 (+1 %: +5 -8 +5)   | fadaNR 0.569 0.579 0.594 0.573 0.568 (+0 %: -4 -3 +8)  (MEASURED 23:10)
quad fadaNPq jumps 0 / 6 / 12 / 24 / 36: nominal 2.12 (14 %) 2.22 2.22 2.27 2.24 (14 %); perturbed 2.94 (9 %) 3.01 3.02 3.03 2.93 (10 %).
Ours (SB, 6 laps): plastic PPO+DR -33 % / RMA -24 %, rubber -5.5 % / -5.3 %; quad RS 2.17 -> 1.22-1.37.
Zero-shot of the students differs from their bases only on the held-out track (runs/screen/FADA_AUDIT_1107.md).

### QUAD figure 5: NO-TEACHER FADA (fadaNPq, teacher = mem/ppo_m0/it300) ADAPTATION, PAPER METRIC (MEASURED 2026-11-07 22:15; 3 seeds x 4 robots)
jumps 0 / 6 / 12 / 24 / 36: nominal 2.12 2.22 2.22 2.27 2.24 (success 14 12 14 12 14 %); perturbed 2.94 3.01 3.02 3.03 2.93.
Per seed nominal 0 -> 36: +5 / +4 / +9 % (worse on every seed). Equal start with our base (PPO+DR zs 2.17, 13 %), as designed.
Ours (RS, same base): 2.17 -> 1.22-1.37 (39-52 %). fadaRq (RMA-data FADA, figure 4 only) for reference: 2.39 -> 2.37, perturbed 3.26 -> 3.08.

### CAR figure 2: NO-TEACHER FADA ADAPTATION (MEASURED 2026-11-07 ~22:40; fig2 protocol, held-out peanut, ladder_eval 5 laps, 3 seeds)
Score mean over cars, episodes 0 / 8 / 24 / 96 / 384 (8 episodes ~ 6.4-7.3 laps ~ our 6-lap budget):
  fadaNP (PPO+DR-taught) rubber 0.485 0.506 0.513 0.512 0.522 | plastic 1.375 1.324 1.308 1.311 1.312 | tier 0.736 0.742 0.730 0.734 0.744
  fadaNR (RMA-taught)    rubber 0.427 0.472 0.491 0.468 0.471 | plastic 0.861 0.852 0.845 0.828 0.852 | tier 0.569 0.579 0.594 0.573 0.568
Per-car % vs OWN zero-shot (the paper statistic), ep 8 / 24 / 384, mean of 3 seeds:
  fadaNP rubber +4.6 / +6.0 / +7.6; plastic -4.1 / -4.9 / -4.9; tier +1.0 / -1.0 / +1.0
  fadaNR rubber +9.3 / +15.2 / +10.8; plastic -1.6 / -3.2 / -1.8; tier +2.5 / +5.9 / +0.3
Ours (SB, 6 laps, same bases): plastic PPO+DR -33.1 %, RMA -23.6 %; rubber -5.5 % / -5.3 %; tier PPO+DR -5.4 %, RMA +16.4 % (worse).

### CORRECTION (2026-11-08 ~00:10): quad stage numbers quoted on 2026-11-07 were the UNMASKED stage
"RS 1.22-1.37 (39-52 %)", "R 1.42-1.49", "RX 1.03-1.61" (sections above, ~13:25 onward) are stage_fl/mem0_{R,RS,RX}: the stage whose refit
read the recorded observation history (MEMORY_ABLATION_1106.md "BUG FOUND 15:00"), not the paper's memoryless stage. The valid runs are
mem0k_{R,RS} (masked), and by the pre-written pick rule only RS passes -> figure 5 / heatmap / runbook = mem0k_RS. PAPER METRIC (MEASURED
2026-11-08 00:05; 3 seeds x 4 robots; nominal / perturbed):
  mem0k_RS  2.17 (13 %) -> 1.64 (31 %), seeds 1.74 1.60 1.57 | perturbed 2.98 -> 2.68 (seeds 2.64 2.69 2.72)
  mem0k_R   2.17 -> 1.84 (24 %) | perturbed 2.98 -> 2.98
  PPO+DR + PPO on robot (fig5/eval_ppo_robot, from mem/ppo_m0): jumps 12 / 36 / 96 / 384 nominal 2.17 / 3.65 / 3.62 / 6.61, perturbed
  2.98 / 4.27 / 4.29 / 7.09 (12 = unchanged: critic warm-up).  No-teacher FADA (fadaNPq): 2.12 -> 2.24 / 2.94 -> 2.93.

### QUAD figure 5: the picked stage (mem0k_RS) at it 2 / 4 (MEASURED 2026-11-08 03:45; fig5_ours_mid.sh, 3 stage seeds x 4 robots, paper metric)
jumps 0 / 12 / 24 / 36: nominal 2.17 / 1.26 / 1.43 / 1.64 (success 13 / 47 / 39 / 31 %, falls 0-1 %); perturbed 2.98 / 2.57 / 2.65 / 2.68.
The stage is best after 12 jumps on 3 of 4 robots (r1 r5 s1: every stage seed best at it 2, worse by it 6; r4 improves slightly to it 6).
Cause NOT investigated (trajectories not inspected). Relevant to the quad hardware runbook (6 iterations = 36 jumps): open question for the user.

## 2026-11-08 07:55 -- real PLASTIC practice stages (session 1108_plastic, scored by scratchpad hw_score2.py: nerr from hand-over, speed > 0.5 m/s mask; signed sideslip in corners, rear-out = share < -3 deg)
| track / plan | net | zero-shot nerr (rear-out) | trials t1/t2/t3 | adapted eval nerr (rear-out) | change |
|---|---|---|---|---|---|
| square2fast a45 | PPO+DR | 2.61 (6 %), pace 0.74 | 2.75 / 2.33 / 2.07 | 1.85 (35 %), pace 0.81, beta p50 -0.1 -> -2.3 | -29 % |
| square2fast a45 | RMA | 2.36 (27 %), pace 0.76 | 2.44 / 2.30 / 2.36 | 2.24 (24 %), pace 0.78 | -5 % |
| fig8R115 a36 | PPO+DR | 2.04 / 1.99 (5 %), pace 0.80 | 2.06 / 2.13 / 2.19 | 2.08 (6 %), pace 0.79 | +3 % (vs mean 2.02) |
| fig8R115 a36 | RMA | 1.88 (3 %), pace 0.81 | 1.97 / 2.37 / 2.20 | 1.90 (4 %), pace 0.81 | +1 % |
VOID flights (stall / wall / mocap lost) excluded: 00006 00011 00015 00019 00023 00027. fig8R115 a36 on plastic stays in grip (beta p50 +2.5..+3.5) and the stage does not raise pace -- matches sim (plastic fig8 never converges at 8 settings).

## 2026-11-08 ~08:05 -- plastic second track switched fig8R115 -> oval165 (user: make the stage work on another track)
Cause measured on hardware: fig8R115 lobes need ~93 % of the real car's steering curvature -> full lock, the stage has no authority.
Sim fig8R115 a30 (6 runs): PPO+DR -1.9 % (1/3), RMA -2.4 % (2/3) -- flat like every plastic fig8 setting.
Oval zero-shot sim sweep (12 plastic cars, launched, 5 laps): a26/a28/a30/a34/a40 PPO+DR nerr 1.30/1.32/1.35/1.46/1.72 (rear-out 4/6/10/18/36 %),
RMA 1.62/1.64/1.67/1.75/1.91 (1/2/3/8/21 %); no crash, e_y <= 0.11 -> rule 1 picks a40. Cells queued: oval30 (Lambda), oval34 (ube), oval40 (matcha).
Rules: ilc2real/OVAL_RULES_1108.md. Hardware: runbook MODIFICATION d (zero-shots a30 + a34, then stage on the faster feasible one).
09:40 oval sim, seed 0 (it0->it3, launched scorer, 12 plastic cars): a30 PPO+DR -18.8 % (3/3, rear-out 10->21 %); a34 PPO+DR -9.2 % (3/3, 18->27 %);
a40 PPO+DR +3.1 % (1/3; harsh_combo / mu 0.47 / mu 0.40 lose the line), a40 RMA -9.7 % (3/3, 21->33 %). RMA a30 / a34 running.
10:10 oval sim RMA s0: a30 -28.8 % (2/2, rear-out 3->16 %), a34 -20.4 % (2/2, 8->19 %); a40 PPO+DR now n=5 -1.4 % (3/5).
12:40 oval40 cell COMPLETE (18): PPO+DR -2.1 % (6/9), RMA -3.1 % (7/9). oval30 PPO+DR complete: -19.7 % (9/9, rear-out 8->17 %). Last oval34 RMA run (s2 ss9003) moved to matcha.
13:40 oval30 cell COMPLETE: PPO+DR -19.7 % (9/9, rear-out 8->17 %), RMA -20.3 % (9/9, 3->10 %). oval34 PPO+DR -10.5 % (9/9, 15->24 %), RMA -17.5 % (8/8 + 1 on matcha -16.1 %). RUBBER same plans (grip contrast): oval34 PPO+DR -22.6 % (5/5), RMA -14.5 % (3/3), rear-out 0->0 %.
14:10 oval34 COMPLETE: plastic RMA 9/9 (-17.5 % x8, -16.1 %). Rubber oval34 COMPLETE: PPO+DR -23.1 % (9/9), RMA -17.7 % (7/9), rear-out 0 %. Rubber oval30 so far PPO+DR -30.3 % (3/3), RMA -26.8 % (1/1). Plastic ETHZ x075 PPO+DR -27.2 % (2/2), rear-out 69->65 %.
14:50 rubber oval30 COMPLETE: PPO+DR -26.5 % (9/9), RMA -20.6 % (9/9), rear-out 0 %. => same-plan contrast complete on oval a30 + a34 (plastic slides more, rubber stays in grip, all improve). Plastic ETHZ remainder moved Lambda->matcha (one duplicate started on matcha killed by PID; Lambda's copy kept).
15:30 INSPECTED: "ours" (fig1, no DR, nominal-sim-trained) on plastic square, car n04_real_nom (mu x0.67), sim section rollouts:
zero-shot runs at plan pace (2.45-2.79 m/s, pace ~0.91), achieved curvature 0.49-0.58 vs plan 0.5-1.0, sideslip -5..-8 deg the whole lap,
e_y +0.35..+0.43 m (outside) everywhere; steering NOT saturated (0.36-0.41 of 0.70). After the stage (it3): steering +15-25 % (0.41-0.49),
speed unchanged, achieved curvature unchanged (0.49-0.57), e_y 0.38-0.46 -> the correction went into steering, which no longer turns a
sliding car; slowing down would. PPO+DR on the same car: 2.1-2.4 m/s, e_y 0.04-0.08. INFERRED (Jacobian not inspected): the stage's
sensitivities come from the nominal sim, where the car grips at this speed, so they credit steering, not speed.
16:20 user: hardware = SPLIT-FRICTION OVAL (oval165 a34; turn 1 = lap 0.056-0.556 on the second surface) + faster square2fast for both tires
(plastic nominal 2.79 m/s = TRACK square; rubber nominal = squarenom, or a45). Sim: rubber oval34split (F1T_MU_PATCH 0.056,0.556,0.36),
rubber square45, rubber squarenom queued on matcha (prac_one4.sh / sched10.sh). Zero-shot sim check (ax:mu=1.667, a34): the slick turn
changes the state the network sees (signed sideslip +4.8 -> +0.3..+2.3 deg, e_y +0.02 -> -0.05 m, 0.1 m/s slower), the grip turn is unchanged.
Scoring caveat found: the sim scorer has no envelope (cars 1.4-4.2 m off the path score 13-36, not crash 5); re-score w/ 1.0 m = user decision.
16:35 user final hardware list: plastic split oval (oval34split), plastic square nominal, rubber square (nominal else a45). Plastic split sim screens: oval34grip (x1.67, Lambda) + oval34slick (x0.6, ube), 6 runs each. Runbook MODIFICATION g.
16:30 user: second surface = FOAM (grippier) + unmodelled jump at the foam edge (hardware only). oval34grip extended to 18 (Lambda s0-1, ube s2); oval34slick dropped after 1 run.

## 2026-11-08 ~17:50 -- user: "make sure everything is publishable by tonight except hardware"; "fig2 is only sim2sim transfer"; "fix fig8 (longer loops)"
- FIG 2 practice-only: every method's checkpoints re-scored on its panel's PRACTISED plan with fig 2's own scorer (ladder_eval, seed 701,
  4 starts, 5 laps): ours = SB_* (sb_practised.json, ube), FADA no-teacher NP/NR + PPO-on-car + RMA-oracle (fig2_practised.json, Lambda),
  GPS (gps_practised.json, Lambda). Panels/plans unchanged (plastic square2fast_a45, rubber fig8R100_x120, tier square2fast). Builder:
  budget2/car_fig2_prac_data.py + stage_fig2_prac.sh -> car_fig2prac(_rel). x label = laps on the deployment car (sim-to-sim).
- FIG 1 practice-only: fig1_eval_prac.py (= fig1_eval.py, panels on the practised plans) over the 160 fig1 checkpoints (matcha sandbox
  ilc2real_ubefix = ube's exact code) + fadaR1's 9 (Lambda). Builder budget2/car_fig1_prac_data.py.
- BEHAVIOUR panel: paper/final/behaviour/fig_behaviour.py (collect_beh.py over PRACB_*): car_behaviour_sim.pdf + _table.txt.
- FIG8 fix (sim only): projection ruled out (local window -0.1..+0.3 m, cannot jump branches). New mocap_fig8R165 (lobes R 1.65 = oval
  curvature, fig8R115's crossing angle, 20.7 m). Zero-shot: plastic a34 nerr 1.52/1.47, e_y <= 0.055, no crash (nominal crashes) -> rule
  picks a34; rubber rule would pick nominal (pace 1.08-1.10) -- a34 used for both (same-plan contrast, as the oval). Cells fig8big34 queued.
- Deferred (fig 2 uses SB): PRACB tier / ours / oursdr items not yet started.
17:00 OUR NETWORK on plastic, MEASURED (scratchpad sens_check.py, car n04_real_nom, plastic square nominal, 0.25 s holds from flown states):
at our network's sliding corner states (sideslip -5..-7 deg, 2.4-2.7 m/s) +0.05 rad steering changes the yaw rate by +0.001..+0.005 in the stage's
nominal model and +0.000..+0.002 in the car (both: no authority); -4 A of drive current changes it by -0.08..-0.13 in the model but +0.05..+0.11
in the car (3 of 5 states) -> the model has the WRONG SIGN for "slow down to turn", so the stage never slows down. PPO+DR (2.1-2.4 m/s, in grip):
steer gain model 0.24 / car 0.11 (same sign). Where our network starts moderately off the line (held-out peanut a33, e_y 0.17) the stage helps
(-16..-28 %, 3/3); on the practised a45 (e_y 0.37-0.45) -14 / -2 / +5 %. USER DECISION: report as a limitation (the precondition: the nominal
model's sensitivities must have the right sign at the start), no method change, no reruns. (Supersedes my 15:30 inference "credits steering".)
17:45 FIGURES 1/2 PRACTICE-ONLY BUILT: final/car_fig1prac_sim, car_fig2prac_real, car_fig2pracrel_real (re-scoring complete: fig1 160 + fadaR1 9,
fig2 SB 84 + baselines). Layout: fig1 per-panel y; fig2 abs plastic cap 4.0; rel ylim -50..47; y labels shortened ("practised track" -> captions).
"Our model better zero-shot" screens (seed 0, matcha sandbox, fig1_eval_prac panels A/B/C/plastic/rubber):
  published ours 0.45/0.69/0.59/3.50/0.54; oursdr 0.47/0.74/0.64/4.11/0.69; ppodr 0.34/0.42/0.45/1.34/0.84
  L (90 its) 0.46/2.33/0.89/7.90/0.68 | W (friction DR x0.5-1.2) 0.46/0.66/0.58/3.17/0.94 | WL 0.37/0.55/0.54/2.63/0.97
  PW (PPO+DR with the same widened DR, fairness) 0.54/0.62/0.67/1.82/0.85. Running: WS / WSL (+ speed error half weight).
  WS (W + speed error half weight, 30 its) 0.37/0.57/0.46/3.06/0.91 | WSL (90 its) 0.36/0.52/0.49/2.34/0.88 = best ours variant (seed 0).
19:30 queued on Lambda (top): stage on oursdrWSL (zero-shot-best screen) vs published oursdr, same PRACB protocol, plastic square45 + rubber fig8 x120, s0 x 3 stage seeds each.
20:30 "fix our stage" check (sens_check_mu.py): with the stage's model friction scaled to x0.67 (the plastic car's), slowing down still gives
the WRONG sign (model yaw-rate -0.04..-0.16 vs car +0.05..+0.11 at 3/5 states) -> not a friction level: the planar model saturates the REAR
in the slide, the multibody plastic car the FRONT. A friction (one-parameter) identification would not fix it. WSL stage on plastic a45:
first run -2.8 % (median +0.1; only the two lowest-grip cars improve). Figures: final/FIGURES_STATUS_1108.md.
  (x0.5 model friction too: slowing -> yaw rate -0.03..-0.17 in the model at every state; the car +0.05..+0.11 at 3/5. Sign never matches.)
