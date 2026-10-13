# How reasonable are the continuity assumptions? Pogo stick, quadruped, car

Companion to `distilc_theory.tex` (Assumption A, Assumption B, and (T1)–(T3) of the transfer proposition), checked
against what each implementation actually computes. Written 2026-10-13 from the code:
- pogo: `src/ilc_pogo/common.py`, `pogo_jax.py`, `hw_stage.py`;
- quadruped: `src/ilc_mjx/ilc_mjx/jump.py`, `scripts/deploy.py`, `src/ilc_quad/scripts/dilc_execute.py`;
- car: `src/ilc_f1tenth/ilc2real/sim_jax.py`, `ilc_core.py`, `hw_stage.py`.

Ratings:
- **OK**: holds as stated.
- **≈**: holds except on a measure-zero / rare set; needs the piecewise-C¹ (semismooth) restatement.
- **✗**: violated by construction.

## 1. Summary

| assumption (theory) | pogo stick | quadruped | car |
|---|---|---|---|
| A(b) fixed, transversal mode sequence near the iterates | **≈** event-driven touchdowns; the hop count inside the 3-s trial changes with the period | **OK/≈** the contact schedule is *prescribed* in time (front lift-off at sample Ndc, rear at Nc); soft MuJoCo contact; unplanned slips / early lift-off break it | **OK** no contact events; wheels always on the ground |
| vector field smoothness (A(c)) | piecewise smooth (spring force ≥ 0 clip, stops) | MuJoCo soft contact: continuous, stiff; friction-cone projection of the forces | Fiala tire C¹ but not C² at the sliding limit; wheel-speed clamp `max(ω, 0)` |
| error map continuous in θ | **✗** hard flight mask `err × (1 − stance)`: a one-period touchdown shift toggles whole rows | **OK** landing error = trunk state at a *fixed* sample N (`logX[N] − target`), terminal, 3 rows | **≈** errors against the plan at the car's own arc-length projection (argmin over a grid + tangent interpolation: jumps O(ds²κ)); hard truncation after a crash / \|e_y\| > max |
| interior actions (A(a)) | **✗** zero-thrust command during compression sits exactly on the clip | **≈** forces projected into friction cones / force limits; torque saturation handled explicitly (`sat_step`: an active set) | **≈** steering ±0.34 rad and motor-current limits saturate in drift transitions |
| Ĝ at the *measured* states (Definition 2) | **OK** autodiff of the nominal model at the measured states, the net's own feedback | **✗** finite-difference Jacobians from the nominal GPU sim's *own rollout* of the current net (nominal or measured start); only the feedback K is at the real states → G − Ĝ also contains a state-mismatch term | **OK** nominal Fiala-ST Jacobians at the measured states (`ilc_core.make_gn`) |
| full row rank, dim E ≤ dim V (B(b)) | OK: 2 rows / 2 inputs per period, half the rows masked in stance | **OK, comfortably**: 3 landing rows vs Nc × 4 inputs | OK: e_y and \|V\| rows vs steering and current |
| (T1) error systematic and smooth in the goal | OK within the gait family (damping, mass, thrust act robot-wide); breaks near v* = 0 (foot placement flips) | OK on flat goals; **breaks across flat / box** (a different contact mode at landing) | goals are **discrete plans**; continuous only along the sideslip homotopy of one track; none across tracks |
| (T2) sensitivity continuous in the goal | **✗ as stated**: time-indexed G; nearby goals' hop phases drift apart (1 % period × 6 hops ≈ 3 periods) | **OK** in time: every goal shares the knot schedule (Nc, N), so G's indices align; **✗** across flat / box | **OK** in arc length (rows aligned by the car's own s); **✗** across tracks |
| (T3) kernel coupling | plausible, unverifiable; supported by the one-goal grid (error ratio 0.11 → 0.9 with distance) | plausible; unflown goals ≤ 6 cm get 50–80 % of the flown gain | plausible within a track's homotopy (the gain falls with Δβ: +0.8 / +0.3 / +0.1 cm at 0–5° / 5–12° / 12–30°) |
| positivity (C) measured | 97 % of trial pairs, median cos 0.88; flip control 0 % | 79 % of 57 trial pairs, median +0.65; flip control fails | 64–80 % (flip 25 %) |

## 2. Pogo stick

- **Mode sequence (≈).** Touchdown happens while descending (vz < 0) and liftoff while the leg extends (L̇ > 0), so
  both crossings are transversal; grazing only at hops that barely touch.
  - Within a fixed hop sequence, the sampled states are C¹ in θ (implicit-function theorem).
  - The sequence itself changes whenever the last touchdown or liftoff crosses the trial's end (or a control-sample
    boundary). That is a kink, not a jump, in the state: a measure-zero set where only one-sided derivatives exist.
- **Error map (✗).** The error rows are multiplied by the hard flight flag (`common.errors`: `w = 1 − stance`).
  - A one-period touchdown shift switches a row between "apex error" and 0, so the trial map has jumps of the size
    of the current error.
  - The jumps shrink as the error converges, which is why the iteration still converges in practice, but Assumption A
    is violated as written.
  - Fix: a smooth flight weight (a sigmoid of foot height, as in the first design), or per-hop event errors
    (apex height and speed at each apex), which are continuous while the hop count is fixed.
- **Clip (✗, harmless).** The Raibert teacher commands zero thrust during compression (normalized action −1, the
  clip's boundary), and the network imitates it. Those inputs sit on a kink; the GN step gets zero or one-sided
  sensitivity there and does not use them.
- **Ĝ (OK).** The pogo implementation matches the theory's Definition 2 exactly: autodiff of the nominal model at
  the measured states, with the network's own feedback.
- **Saltation.** Only the touchdown's terms are added, as straight-through corrections. Liftoff needs none: the leg
  force is zero there, so the vector field is continuous.
- **Goal continuity (T2 ✗ as stated).** Rows are time-indexed, and goals with different hop periods de-phase over
  the trial. A state- or phase-indexed statement is needed (the network is a function of the state, not of time).
  The one-goal grid experiment shows broad transfer anyway.

## 3. Quadruped (Go1 jumps)

- **Mode sequence (OK/≈): the strongest of the three.**
  - The policy acts only during a contact phase whose schedule is fixed in time: front-foot lift-off at sample Ndc,
    rear at Nc.
  - Flight follows the TO joint profile; landing follows a joint PD.
  - The controlled part of the trial therefore has a prescribed mode sequence, and MuJoCo's soft contact keeps the
    dynamics continuous (stiff).
  - The forces are projected into the friction cones, so the model never slips. The real robot can (unplanned slip,
    early lift-off), and then A(b) fails for that trial.
- **Error map (OK).** The landing error is the trunk state at a fixed sample N (`dilc_execute.make`:
  `e3 = logX[N] − target`).
  - It is terminal (3 rows: x, z, pitch) and continuous in θ.
  - The model's ballistic map `PHI = Af^(N−Nc)` assumes flight until N. If the real touchdown happens before N,
    `logX[N]` contains impact and landing-PD dynamics: a kink, and a modeling error inside Ĝ, not a discontinuity.
  - Falls are discrete (fall flag), handled by rollback and the stall guard, outside the theory.
- **Saturation (≈).** Friction-cone and force-limit projections are piecewise smooth. Motor-torque saturation is
  handled explicitly by an active set (`sat_step` holds the saturated columns), which is the right treatment of
  A(a)'s failure.
- **Ĝ (✗ against Definition 2).**
  - The nominal Jacobians A_n, B_n are finite differences (step ε) taken along the nominal GPU sim's own rollout of
    the current network, from the nominal start (or the measured start with `--sim-start measured`).
  - Only the feedback K is evaluated at the real states.
  - So G − Ĝ contains a state-mismatch term besides the derivative mismatch, and positivity must absorb both. The
    proof's remark that "G − Ĝ is derivative mismatch only" does not hold for the quadruped. The theorems themselves
    still apply (they only use positivity), but the sufficient conditions of Lemma 4 have to include that term.
  - Finite differences across soft contact act as a smoothed (averaged) derivative, arguably helpful at kinks.
- **Rank (OK).** 3 terminal rows against Nc × 4 inputs: full row rank is easy; zero landing error is reachable to
  first order unless the goal is infeasible (the heavy robot's far box goals: feasibility fails, and the stall guard
  freezes them).
- **Goal continuity.** All goals share the knot schedule, so time-indexed sensitivities align (unlike the pogo
  stick).
  - (T1)–(T3) are reasonable on flat goals.
  - The flat / box boundary is a mode boundary at landing (box top vs floor, front-face clearance), with no
    first-order guarantee across it.
  - Matches the data: unflown goals ≤ 6 cm from a flown one get 50–80 % of the flown gain; box / flat interference
    is the documented failure.

## 4. Car (F1TENTH drift)

- **Mode sequence (OK): the smoothest of the three.**
  - No contact events; the trial map is C¹ almost everywhere.
  - Fiala tire force is C¹ but not C² at the sliding limit.
  - The wheel-speed clamp `max(ω, 0)` and the open-differential / multi-body structure of the real cars add kinks
    only at rare states.
- **Drift is the delicate regime for positivity, not for continuity.**
  - The plans run at tire saturation, where the sensitivity of path error to steering is small and its sign depends
    on the regime (grip vs slide, countersteer).
  - That is where positivity is weakest: measured 64–80 %; the low-friction car needed the reference ladder.
- **Error map (≈).**
  - Rows are time-indexed but measured against the plan at the car's own arc-length projection
    (`sim_jax.path_coords`, `project`), so they are self-aligned in phase.
  - The projection index is an argmin over a grid window with a tangent-offset interpolation: tiny jumps
    O(ds² κ) at index switches.
  - Hard truncation after a crash (and \|e_y\| > max_ey in the sim stage, by cumprod) is a discontinuity, but only
    next to a failure.
- **Ĝ (OK).** Nominal Fiala-ST Jacobians at the measured states, with the lifted time and the network's feedback
  (`ilc_core.make_gn`), consistent with Definition 2.
- **Delays and actuators.**
  - The real cars' steering and current delays and actuator lag make G's lag structure differ from Ĝ's. That is a
    positivity question, not a continuity one.
  - Evidence: a 2-period steering delay halves success.
- **Goal continuity.** Goals are discrete plans: 2 tracks × sideslip β.
  - (T1)–(T3) can only hold along a track's β-homotopy; the gain falls with Δβ (+0.8 / +0.3 / +0.1 cm at 0–5° /
    5–12° / 12–30°).
  - There is no continuity across tracks; cross-track transfer, if any, is generalization, not the proposition.
  - The reference ladder (`_mu80`, `_mu60` rungs) is a discrete outer loop entirely outside the theory.

## 5. What to change in the proof (all three domains)

1. **Restate Assumption A as piecewise C¹ / semismooth.**
   - Finitely many C¹ pieces (one per mode sequence and saturation pattern), with Clarke generalized Jacobians.
   - Theorem 2's argument goes through for a semismooth trial map with a uniformly positive generalized sensitivity
     (semismooth Newton-type analysis), on a neighbourhood that excludes nothing.
   - This covers: pogo hop-count kinks, quadruped late touchdowns and cone projections, car Fiala / clamp kinks,
     and all three domains' clips.
2. **Make every error map continuous**, so the semismooth restatement applies.
   - Pogo: smooth flight weight, or per-hop event errors.
   - Car: replace hard truncation by a smooth down-weighting near the envelope.
   - Quadruped: already continuous.
3. **Generalize Definition 2.** Allow Ĝ evaluated along a nominal-model trajectory (the quadruped) and add the
   state-mismatch term to Lemma 4.
4. **Restate (T2)–(T3) in state or phase coordinates**, or label the transfer proposition heuristic.
   - The quadruped (shared knot schedule) and the car (arc-length rows) already align; the pogo stick does not.
5. **Treat mode boundaries in the goal space as explicit exclusions:** quadruped flat / box, car across tracks, pogo
   near v* = 0. Report transfer only within a mode cell.
