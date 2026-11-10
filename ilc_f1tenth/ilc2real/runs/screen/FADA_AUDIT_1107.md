# FADA baseline audit (2026-11-07 ~00:30; user: "do a comprehensive evaluation of fada ... check their trajectories and corrections")

Reference: Xie et al., "FADA: Few-Shot Domain Adaptation via Dynamics Alignment for Humanoid Control", arXiv 2606.28476v2 (read from the
arXiv HTML; LoRA rank / lr / steps / batch only from a summary of Appendix B.3, unverified).

## Deviation table (paper -> ours)
| item | paper | car (fada_car.py / fada_real_car.py) | quad (fada_train.py / fada_adapt.py) |
|---|---|---|---|
| planner | learned, trained THROUGH the frozen IDM to reproduce the oracle's first action (Eq. 4.3) | NONE: fixed decay d* = 0.8^H d (H 4: 41 % of the current deviation) | learned, regressed to the teacher's realized future error (the paper's ablated variant, Sec 5.5: 7/10 vs 10/10) |
| IDM inputs | obs history H=30 + action history + K=6 future obs chunk | o_t + d_{t+4} (5 deviation dims) | o_t (memoryless, ILC_OBS_MEM=0 like every arm) + e_{t+H} |
| IDM output / loss | K=6 action chunk, first-action MSE | one action, MSE | one action, masked MSE |
| architecture | transformers | MLP 256x256 | MLP (the policy's, warm-started) |
| adaptation | LoRA on the IDM only, planner frozen, realized (o, future obs) -> executed action, one round ~6000 steps | same principle; LoRA r4 every layer, 500 Adam steps lr 1e-3 per round, refit on all data every 8 episodes; relabelling with the TRUE state (--privileged 1) | same principle; LoRA r4, 500 steps, lr 1e-3 per iteration (deploy.py's budget) |
| LoRA r / alpha / lr / steps / batch | 8 / 16 / 3e-4 / 800 / 4096 [summary] | 4 / - / 1e-3 / 500 / 2048 | 4 / - / 1e-3 / 500 |

## Measured (car, pure-numpy rollouts, fig2's env; scratchpad fadaaudit/audit.py, lamtest.py)
- Consistency gap (FADA Sec 5.7): RMS(reached d_{t+4} - requested) / RMS(d) = 0.59-0.66 on every car, panel and checkpoint (ep0 ... ep384):
  exactly the value if the deviation simply persists (|d - 0.41 d| = 0.59 d). LoRA adaptation does not shrink it.
- The IDM does use its target (|d a / d target| ~ 0.7-1.1 on steering, as large as on the current deviation); asking lam_eff 0 / 0.41 / 1.0
  changes the lap little, 1.5 a lot. So the IDM is not a behaviour clone; the decay planner asks for unreachable targets.
- Tracking gets WORSE with adaptation on most cars (rubber train plan 0.035 -> 0.166 m RMS e_y at ep384; tier mocapbad held-out 0.188 ->
  2.67 at ep96; plastic ax:mu=0.4 held-out 0.36 -> 1.17). Cause NOT verified (hypothesis: the adapted IDM is fitted to reachable targets and
  then queried at unreachable ones).
- The paper's learned planner (fada_car_planner.py, Eq. 4.3, oracle = the IDM data's source network freeBCN_s0, its state distribution):
  held-out first-action MSE vs the oracle 0.00034 (learned) vs 0.138 (our decay planner) vs 0.048 ("hold the deviation"), fada_d60_s0.
  => the car FADA baseline was handicapped by our planner; the fix follows the paper.

## RERUN RULES (written before any result)
Car, every FADA cell: the planner replaced by fada_car_planner.py's (one per IDM: figure 1's fada_d{5,10,20,40,60}_s{0,1,2}), frozen;
everything else unchanged (IDM, data, LoRA settings, --privileged 1, checkpoints 8 / 24 / 96 / 384, seeds 5151 + s).
- Figure 1: fig1_eval.py on runs/fig1/fada_d*_s?/idm_planner.npz (the same evaluation as every figure-1 network) -> fig1_eval_fadaP.json.
- Figure 2: fada_real_car2.py --planner runs/fig1/fada_d60_s<s>/planner.npz, the fig2 protocol (panels tier / plastic / rubber, seeds 0-2)
  -> runs/hw/fig2fadaP_<panel>_s<s>_ep<K>; scored as figure 2's other arms (ladder_eval, held-out peanut, seed 701, 5 laps).
- The old decay-planner FADA results are kept and reported as "FADA with a fixed decay planner (our earlier implementation)" in the
  supplement; the learned-planner FADA is the figure's FADA. Every result reported.
Quad: the planner deviation (regression vs through-IDM) needs the teacher's actions at the planner's states (not saved by the running
trainings) -> a new planner-data collection; decided after the car result (does the planner matter there).

## QUAD RERUN (user 2026-11-07 ~00:30: "go ahead and fix the quad fada planner too"), written before any result
log/dilc/plane/fada_quad_fix.sh: per figure-4 FADA cell (seeds 0-2, teacher it100 / it300):
 A the planner rollouts WITH the teacher's actions (fada_train_pthru.py --collect-only; fada_train.py's planner distribution, 25600 jumps
   as 25 x 1024, seed s + 100);
 B the IDM refitted at H = 6 from the running run's own data.npz (fada_train.py --load-data --relabel-h 6): the paper's K = 6 future
   chunk; the baseline audit measured H = 1 as not identifiable (the IDM explains 2 % more than without its target);
 C the planner trained THROUGH that frozen IDM to the teacher's action (Eq. 4.3) -> fig4/fadaP_s<s>_ck<k> = the figures' FADA.
Figure 4: fadaP points (the H = 1 regression-planner FADA reported as the earlier implementation). Figure 5: FADA adaptation on fadaP.
Figure 6: the 20 FADA retargeting jobs now run fadaP_s0_ck300 (retarget_job_fadaP.sh; jobs_RT.json, old entries backed up). fada_adapt.py
relabels at the IDM's horizon P_h (checked: obs[min(k + H, Nc)]).

## SECOND PASS (2026-11-07 ~00:50; arXiv HTML v1 + v2 appendix read verbatim; project page empty, no code released)
- Adaptation (v2 B.3, verbatim): "Adaptation is performed once, offline, after the full target dataset is collected, for a fixed 800 steps
  with learning rate 3x10^-4 and batch size 4096; no online or iterative updates are used." LoRA r 8, alpha 16, dropout 0.05 (grid
  r {4,8,16} x alpha {8,16,32}, "not sensitive"); LoRA "inserted into the IDM" (layers unspecified). Target data: ~2 min at 50 Hz (~6000
  control steps) collected with FADA-zs (T1). User 2026-11-07: "keep the one offline round if that's what fada does" -> our iterative
  per-8-episode refit is replaced; the fig2fadaP runs (iterative) were stopped (00:50) before any score.
- Architecture (C.1, Table 8, T1 Loco+Payload E_v, normalized): planner / IDM encoder / IDM decoder all MLP 1.000 (IsaacSim) and 1.000
  after adaptation; all transformer 0.962 / 0.706. "replacing the attention-based modules with MLP counterparts leaves source-domain
  performance nearly unchanged but removes most of the post-adaptation benefit."  => our MLP FADA is the paper's weakest variant.
- K (Table 4): K = 1 -> 6 reduces error ~18 %; K 10 / 15 no further gain. H = 30 fixed (no history ablation). Deployment is receding-
  horizon: one action per step (the chunk is internal; the IDM's decoder attends over the K future tokens, first action supervised).

## FULL-FIDELITY FADA: figure-2 adaptation protocol (2026-11-07 ~02:15, written before any result)
fada_real_car_tf.py (paper Algorithm 1 T1-T3, Eq. 4.4-4.6, B.3): source = fada_car_tf.py's final DAgger iterate fadaT_s<s>/it8.pkl.
T1 ONE round of target rollouts by FADA-zs only: 384 episodes per car (fig2_fadaP.sh's episodes: 1.05 laps of the panel's practised plan,
random arc-length start, start perturbation U(-0.1, 0.1)); budgets K = 8 / 24 / 96 / 384 = the first K episodes (figure 2's x-axis;
real laps recorded). T2 per budget ONE offline LoRA fit from the base IDM: windows (O^H, A^H, Y_exec = o_{t+1..t+K}) -> a_t as executed
(measured observations), r 8 / alpha 16 / dropout 0.05 on every IDM Linear, planner + base frozen, 800 Adam steps, lr 3e-4, batch 4096
drawn with replacement, first-action MSE. T3 the merged IDM deployed receding-horizon -> runs/hw/fig2fadaT_<panel>_s<s>_ep<K>.
Scoring = fig2_fadaP.sh (ladder_eval, held-out peanut, 5 laps, seed 701). Launcher fig2_fadaT.sh.
Checks before launch (PASS): smoke adaptation (4 episodes): the held-out executed-action MSE of the adapted IDM 0.00061 vs base 0.0088;
the adapted pickle loads through reeval/policy_runtime (mode fada_tf) and flies. Speed: FADA_JIT=1 = the same forward XLA-compiled per
process (fada_tf_tests 9: == numpy to 3e-7 over 60 steps incl. the padded start; 3.5 vs 19.7 ms per control step); rollout workers on
CPU, the LoRA fit on a GPU (XLA command buffers off: the CUDA-graph instantiation ran out of memory at batch 4096).
Declared: Adam (optimizer unspecified in the paper); the source trained on the JAX planar DR sim, the targets are the multi-body cars.

## QUAD FULL FIDELITY (2026-11-07 ~03:40)
fada_quad_tf.py (source: Algorithm 1 S1-S5 as fada_car_tf.py; oracle rmat_s<s>/it300, suboptimal its it25-it275; o_k = [state error, time,
previous executed action] (11), c_k = the goal (2); oracle K-sample shadows from the same MJX state, estimator and command buffers;
futures capped at the takeoff sample; swing legs' unused action entries masked) and fada_adapt_tf.py (T1-T3: one round of FADA-zs jumps,
fada_adapt.py's goals / seeds / evaluation, budgets 6 / 12 / 24 / 36 jumps, LoRA r 8 / alpha 16 / dropout 0.05, 800 x 4096, lr 3e-4).
Robot runtime: dilc_execute.NumpyActor runs fada_tf.FadaPolicy when policy.npz carries fada_tf (weights as plain arrays FP__ / FI__:
the container's numpy 1.21 cannot read numpy-2 pickles). Checks (PASS): smoke source training (2 iterations, oracle + student + subopt
rollouts with shadows); robot-sim flight of the export; replay of the recorded jumps: the stateful runtime == the batch windows of
training / adaptation (0.0), == the executed actions to 4e-8, the observation's previous action == the executed one (0.0).
The MLP-era quad FADA (fada_train.py collections fada_s<s>_ck<k>, ~20 % after 3.5 h) is superseded: 5 of 6 stopped (by PID) to free
matcha's GPUs; fada_s0_ck300 kept running = the fallback for figure 6's RT_fadaP retargeting only.

## WHY FADA-zs BEATS RMA zs ON THE CAR (2026-11-07 ~10:45, user: "something is wrong here") -- checks, MEASURED
1 scoring: rma_s0 / rmaN_s0 re-scored on matcha by the same fig1_eval.py = ube's numbers exactly (0.855 / 0.608; 0.886 / 0.580).
2 inputs: FADA reads the measured obs[:13] + its own executed actions + the plan preview -- nothing privileged at test time.
3 history: FADA with its history cut to 1 step (inside its training distribution) still beats RMA on plastic (n04_real_nom 0.24 vs 0.35,
  n04_harsh_steer 0.27 vs 0.63; flat per-lap e_y) but not on rubber (ax:mu=1.667: 0.82 vs RMA 0.62; full history 0.46).
4 the recipe vs the architecture (running; rule written before results): fadaM_train_eval.sh -- the same source recipe with MLP modules.
  MLP-FADA it 8 below RMA zs on plastic / rubber -> the recipe (DAgger distillation of the true-z teacher) explains the edge; at / above
  RMA -> the transformer. Either way figure 1 charges FADA its teacher's simulated laps.

## 2026-11-07 ~11:00 LABEL LEAK in the full-fidelity FADA (car fadaT, quad fadaT) -- ALL fadaT / fadaM RESULTS INVALID

Found while diagnosing why FADA's adaptation gave ~0 gain (user: "the whole point of fada is few shot adaptation"). Our o_t appended the
previous executed action (car o = [obs[:13], a_{t-1}] (15); quad o = [err, time, prev_a] (11)). The IDM's future chunk is
Y = o_{t+1..t+K}, so Y[0]'s previous-action slot IS a_t, the IDM's target. Measured (scratchpad fada_idm_diag.py, fadaT_s0 it8, 4 target
episodes each): base-IDM action MSE on target data 2.86e-5 (n04_harsh_steer) / 2.99e-5 (n04_real_nom) vs 7.6e-3 / 8.1e-3 for copying
a_{t-1}; zeroing the whole action history moves the action by 0.0025 RMS, +1 std on Y by 0.17. The IDM copied a_t out of Y and learned no
dynamics -> LoRA adaptation had nothing to correct (fits 1e-4 -> 2e-7; fig2 gains ~0 on every car incl. pure actuation gaps); the planner,
predicting that slot, was itself the policy (a DAgger student of the privileged oracle) -> zero-shot at / above RMA.
The paper's IDM is I(O^H, A^H, Y): actions enter as their own tokens; appending a_{t-1} to o was our choice and is what leaked.
FIX: o_t = state only (car obs[:13] = 13, quad [err, time] = 7); actions only as A^H. fada_tf.d_obs(I) infers d_o from the weights so the
old models still load (legacy flag --o-prev 1). Code backups: src/paper/backup/fada_leakfix_1107/.
Invalidated: runs/fig1/fadaT_s*, fig1_eval_fadaT*, fadaM (MLP test, killed), runs/hw/fig2fadaT_*, runs/fig2base/fig2fadaT_*,
quad fig4/fadaT_s*, fig4/eval/fadaT_*, stage_fl/fadaT_s* (killed). Kept on disk for the record, not for figures.
Reruns (identical recipe, only o_t changed): car fadaL_all.sh (matcha: train s0-2 -> fig1_eval_fadaL.json -> fig2 plastic + rubber),
quad log/dilc/plane/fadaLq_all.sh (train -> fig4 eval -> fig5 adaptation). Tier fig2 panel: after the car training.
### fadaL (leak fixed) first results (MEASURED ~11:45)
Training: car IDM loss at it 2 2e-3 (leaky 9e-5); quad it 0 end 0.017 (leaky 2.4e-4) -> the IDM now has to model dynamics.
Fig 1 zero-shot, fadaL_s0 (it 1 / 2 / 4 / 8): plastic 0.622 / 0.699 / 0.640 / 0.665, rubber 0.622 / 0.595 / 0.482 / 0.428 -- about the
leaky fadaT_s0's (it 8 0.63 / 0.51) and still below RMA zero-shot (rma_s0 0.855 / 0.608). So the leak did NOT cause the zero-shot edge
(s0 only so far; s1 / s2 scoring). Whether it caused the ~0 adaptation gain: fig2L (running).
### fadaL zero-shot edge over RMA: bug hunt (MEASURED ~12:20; user: "there must be something still wrong with fada")
- eval path (real_car.run): FADA and RMA read the same measured (noisy, delayed, held) obs, same action noise / delay buffers, no clipping.
- training family: both sim_jax dr_params scale 1, F1T_DR_BOX / F1T_DR_MU unset, rma_s0's 6 plans.
- in the TRAINING sim (scratchpad fada_vs_teacher_simjax.py, 192 DR cars, no noise): teacher + exact z rms e_y 0.0209, fadaL_s0 0.0215,
  teacher with z = 0 0.0456, 0 fails -> FADA is a faithful student, no privileged leak.
- in the DEPLOYMENT sim (mb_car, fada_vs_rma.py / fada_noise.py, 3-5 laps, seed 701): FADA's edge is lateral only, same pace / speed:
  tiera_kt_hi e_y 0.031 vs RMA 0.040 vs oracle (teacher + mapped true z) 0.044; tiera_doff_hi 0.031 / 0.047 / 0.044; n04_real_nom 0.014 /
  0.035 / 0.012 (oracle pace 0.928); n04_harsh_steer 0.038 / 0.063; ax:mu=1.667 0.039 / 0.062. Sensing noise off: unchanged for all.
  The oracle's pace is low on every car (0.93-0.97) -> its mb_car -> z mapping is suspect (an oracle issue, not FADA's).
=> no bug found; the edge appears only across the planar -> multi-body gap (FADA = teacher in the training sim). Cause NOT established
   (inferred candidates: DAgger state coverage incl. shadows + suboptimal data; history conditioning vs RMA's 8-d latent bottleneck).
### DAgger-student diagnostic (launched ~11:58, Lambda; RULE WRITTEN BEFORE RESULTS)
dagger_car.py: FADA's exact source recipe (teacher rma_s<s> + exact z, DR 1, 6 plans, 8 its, 512 x 160, 2x suboptimal, 2000 x 512 steps)
with ONE memoryless MLP student pi(obs) (RMA actor size, no z, no history, no planner / IDM). Scored by fig1_eval.py (it 1 2 4 8).
Rule: dagger it 8 at / below RMA zs (rma_s<s>) on plastic AND rubber -> the edge comes from DAgger distillation of the privileged teacher
(generic; RMA's phase 2 lacks it) -> the apples-to-apples fix is on the training-recipe side (both students get the same distillation data),
not FADA's structure. dagger above RMA zs -> FADA's planner-IDM / history explains it.
## 2026-11-07 ~12:10 DECISION (user): FADA in the results = FADA trained on RMA's distillation data ("only this version in the results, it is fairer")
RMA's phase 2 (ppo_dr.py): on-policy (its deployed loop; it 0 the teacher), 150 its x 512 lanes x 160, every 4th period, buffer of the last 8
iterations, loss latent MSE + 10 x action matching to the teacher's action (p2_am 10). FADA's paper recipe adds 2x suboptimal-checkpoint
rollouts and oracle K-step shadows; those are dropped. fada_car_tf.py --data rma (planner + IDM unchanged; IDM on executed trajectories, planner
to the teacher's action at visited states through the frozen IDM; 200 Adam steps / iteration at batch 512 -- RMA's batch 8192 declared
not matched). Pipeline: fadaR_all.sh (Lambda). fadaL (paper recipe, leak fixed) = diagnostic only, not in the results.
### RESULTS ~12:45 (fig1 zero-shot, MEASURED) + the INPUT mismatch
plastic / rubber: RMA rma_s0 0.855 / 0.608; DAgger MLP (memoryless) s0-2 0.816 1.004 0.803 / 0.522 0.582 0.515; fadaR (RMA-matched data) it150
0.689 0.697 0.663 / 0.204 0.211 0.302; fadaL (paper recipe) 0.665 0.763 0.655 / 0.428 0.222 0.224; per-car PPO ceiling (v2b, best saved
iterate; memoryless) 0.542 / 0.319 -- zero-shot FADA beats the per-car ceiling on rubber.
=> the data recipe is not the edge (fadaR = fadaL); plain distillation is not (DAgger ~ RMA). What differs: FADA conditions on H = 30 steps of
history; EVERY other arm (PPO+DR, RMA's actor, DAgger, the ceiling, our stage's networks) acts on the current observation (RMA: history only
through its 8-d z_hat). The earlier FADA port was memoryless "like every arm" (deviation table above); the full-fidelity port gave FADA an
input no other arm has.
TEST (rule written before results): fadaR1 = fadaR with --hist 1 (o_t + the previous action token only), seeds 0-2, fig1 it150.
fadaR1 plastic and rubber at / above RMA zs -> the 30-step history explains FADA's zero-shot edge -> the apples-to-apples FADA is memoryless
like every arm (user decides). fadaR1 still clearly below RMA (beyond the seed spread) -> history is not it; keep looking.
- 12:52 killed fadaL fig2 plastic s2 (PID 1679370, diagnostic-only): matcha GPU 3 reached 88 C
### Does FADA's adaptation benefit from history? (launched ~13:10, Lambda, hist_adapt_rubber.sh; written before results)
fadaR (H 30) vs fadaR1 (H 1), both RMA-matched data, fig2 rubber protocol, gains it 0 -> it 8 / 24 / 96 / 384 episodes per seed. The paper fixes
H = 30 (no history ablation). Reported as measured, both directions.
### Open-loop sim-vs-sim check (MEASURED ~13:25, scratchpad openloop.py): RMA's closed-loop commands on real_nom replayed open loop in mb_car and
in sim_jax (nominal params), same start: steering angle and motor current IDENTICAL (rms diff 0.0000) -> actuator / delay / servo paths match;
chassis differs: yaw rate mean -1.35 (mb) vs -1.86 (planar) rad/s (rms diff 0.71), vy 0.18 vs 0.48, vx 2.97 vs 2.68 -> the multi-body car
understeers ~30 % vs the planar model (load transfer / suspension: the intended "real car" gap, not a bug). Consistent with: e_y bias +2.5 cm
on every policy, RMA's d_off estimate -0.7..-1 on every car, teacher + true z worse than z = 0 on tier A. FADA's history models absorb it.
- 13:45 killed fadaL car fig2 plastic s0/s1 (diagnostic only; rubber done) to free matcha GPUs 1-2 for the quad FADA adaptation
### fadaR1 (H 1) RESULT (MEASURED ~13:40; fig1 it150, s0 / s1 / s2): plastic 0.769 / 0.808 / 0.767, rubber 0.366 / 0.260 / 0.326
(fadaR H 30: 0.689 / 0.697 / 0.663, 0.204 / 0.211 / 0.302; RMA rma_s0 0.855 / 0.608; DAgger MLP 0.80-1.00 / 0.52-0.58).
Rule -> history explains PART (~0.1 on both panels) but fadaR1 stays clearly below RMA on rubber and below on plastic -> history is not
the whole edge; the remainder sits in the planner -> IDM structure (+ the a_{t-1} token). fadaR1 rubber ~ the per-car PPO "ceiling" (0.319).
### CORRECTION + controlled test (~13:55; user: "planner to inverse model is supposed to be for adaptation, not zero shot")
The fadaR1-vs-DAgger comparison was NOT controlled: DAgger used the paper-recipe data (8 its + suboptimal) and obs only; fadaR1 RMA's data
(150 on-policy its) + the a_{t-1} token. Attributing the residual edge to the planner -> IDM structure was wrong. Test: dagger_car.py
--data rma (fadaR1's exact data) with obs only (daggerR) and with [obs, a_{t-1}] (daggerRp), seeds 0-2, fig1 it150. RULE (before results):
daggerRp within the seed spread of fadaR1 -> the structure adds nothing zero-shot (edge = data + previous action); daggerRp clearly worse ->
the structure contributes. daggerR vs daggerRp isolates the previous-action input.
### QUAD in our regime (~14:05): fadaRq_all.sh (matcha): fada_quad_tf.py --data rma --hist 1 (the quad RMA student's data: 25 its x 1024
jumps on-policy, it 0 teacher, replay 1, async DR, est window 0.025; teacher-action labels; no suboptimal / shadows; memoryless) -> fig 4
scoring + fig 5 adaptation in parallel. Stopped (PID) the paper-recipe quad FADA (fadaLq_all.sh / fadaLq_adapt_par.sh): not in the results.
### daggerR RESULT (MEASURED ~14:40, fig1 it150, s0-2): daggerR (obs) plastic 0.754 0.878 0.783 / rubber 0.474 0.432 0.389; daggerRp (+a_{t-1})
0.765 0.793 0.793 / 0.559 0.424 0.307; fadaR1 0.769 0.808 0.767 / 0.366 0.260 0.326. Rule: plastic daggerRp within fadaR1's spread -> the
planner -> IDM structure adds nothing zero-shot; rubber fadaR1 better on average (0.32 vs 0.43) with overlapping seeds -> not clear-cut.
a_{t-1}: no effect (0.805 vs 0.784; 0.432 vs 0.430) -> fadaR1 KEEPS the token (FADA's own IDM input). fadaR1 FINAL = the FADA in the results.
Plain distillation on RMA's own data beats RMA zs (0.80 / 0.43 vs 0.855 / 0.608): RMA's 8-d latent bottleneck, not its data, is weak here.
Stopped (PID) fadaR (H 30) fig2 plastic / tier on Lambda (not in the results); launched fadaR1_fig2.sh (plastic + tier + all-panel scoring).
### Rubber adaptation, all FADA versions WORSE (MEASURED ~14:15, held-out peanut it 0 -> 384): fadaL (paper recipe) 0.291 -> 0.386 (+34 %),
fadaR 0.239 -> 0.258 (+9 %), fadaR1 0.317 -> 0.445 (+39 %); fadaR1 s0 on its own practice track (fig8) 0.591 -> 0.570 / 0.575 -> 0.629.
Checks: paper Eq. 4.3 trains the planner only through the IDM (no future-observation loss; Sec. 5.5 by design), T1 = plain FADA-zs rollouts
(no noise), T2 LoRA on executed futures (Eq. 4.5) -> ours faithful; windows aligned (base IDM + planner Y reproduces executed actions exactly);
fit loss 6.7e-3 -> 3-5e-4 and the merged IDM keeps it (merge correct); planner Y vs executed futures rms 3.41 (fadaR1) / 0.66 (fadaL, with
shadows) of a spread 1.0. Inferred, not tested: rubber has no headroom (FADA zs ~ the per-car ceiling 0.32); plastic (room 0.78 vs 0.54) decides.
### CORRECTION (~17:00; user: "it makes no sense that fada would be able to beat its teacher at deployment"): FADA does NOT beat its teacher
Checks: no F1T_DR_MU / F1T_DR_BOX leak on any machine; the teacher rma_s0 was trained at F1T_MU 0.6 / plans_nom06 (fig1_queue.sh), as FADA.
Teacher with z = 0 (fig1, s0-2): plastic 1.05-1.13, rubber 0.65-0.67 (FADA fadaR1 0.78 / 0.32); friction-only latent sweep: rubber best 0.64.
FULL 8-d CEM over the teacher's constant latent (scratchpad teacher_cem*.py, 24 x gens, 2 starts x 3 laps): n04_harsh_steer teacher 0.390 vs
FADA 0.492 (5 gens); ax:mu=1.667 teacher 0.303 (12 gens, still falling) vs FADA 0.269, best z UNPHYSICAL ([-1.63, 0.88, 1.16, 1.2, 3.62,
3.4, 0.65, 2.58]: servo tau / d_off / i_delay far outside the box) vs the car's mapped parameters (score 1.80). => the teacher at its best
latent ~ or better than FADA, as distillation implies; "FADA beats the teacher" earlier = vs the teacher on WRONG latents. RMA / the oracle
feed PHYSICAL parameters, which are the wrong input on the multi-body cars; the latent-free student lands near the teacher's best.

## No-teacher FADA (fadaNP / fadaNR, 2026-11-07 ~18:00-18:50): zero-shot and WHY the student differs from its teacher (MEASURED)
fig1_eval (held-out peanut), it150, mean of 3 seeds | base networks scored with the same script (runs/fig1/fig1_eval_bases.json):
  plastic: freePPON 1.146 -> fadaNP 1.375 (+20 %);  rmaN 0.932 -> fadaNR 0.862 (-8 %)
  rubber:  freePPON 0.763 -> fadaNP 0.485 (-36 %);  rmaN 0.647 -> fadaNR 0.427 (-34 %)
  tiers A / B (inside DR): student = teacher within 0.02; tier C: students better (0.870 vs 1.035, 0.639 vs 0.775).
Trained action match: rms 0.012-0.014 (ppodr), 0.043-0.053 (rma). Inspection (scratchpad np_rubber.py / np_token.py / np_noise.py / np_where.py):
student-teacher steering difference at the TEACHER's visited states, fadaNP_s0 vs freePPON_s0:
  sim_jax training plans DR1 0.012, nominal 0.008 | sim_jax HELD-OUT peanut nominal 0.153, DR1 0.162
  mb_car n04_real_nom: training plan square2fast 0.020 | held-out peanut 0.184 | rubber peanut 0.175-0.178
  zeroing the previous-action token: 0.19 (no change); sensing noise off / true observations: 0.142-0.175 (no change); low-pass part 0.13.
CAUSE (measured): the held-out TRACK's plan preview is outside the distillation data; on every trained track the student IS the teacher
(either simulator), on the held-out track the two are different extrapolations of the same labels -- not a dynamics effect, not a leak.
Direction is chance (rubber / tier C better, PPO+DR-taught plastic 20 % worse). Consequence for figures 2 / 5: each method's change is
measured from its OWN zero-shot on the held-out goals (the paper's claim, feedback 2026-10-31); FADA's zero-shot point is plotted as is.
