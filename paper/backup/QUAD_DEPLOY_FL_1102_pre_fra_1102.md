# Quad hardware stage on free-landing plans (2026-11-02)

Supersedes the network and plans in `QUAD_DEPLOY_FR_1031.md`. The stage is unchanged: FR, the same flags, the same arms. Recipe and evidence for the plans: `FREE_LANDING_1101.md`.

Nothing here has flown on real hardware. Everything below is simulation, on the CPU robots (async, delays, mocap, motor curve).

## 1. Measure damping first. It picks the network.

Fly the three-jump test from `QUAD_DEPLOY_FR_1031.md`, section "First session: measure the joint damping", before any network flight. It reads high: in noisy sim truth 2 read 3.0, truth 0.1 read 0.3-0.5.

| Reading | Fly | Plans | Evidence (2026-11-02) |
|---|---|---|---|
| >= 1.5 | `fl2s/ppo_dr/it50` | `plans_fl2s` | stage -0.34 [-0.50, -0.17]; robots at 1.5: no falls, jumps 7-13 cm high |
| 0.6 to 1.4 | damping-1.0 network (`fl1s` / `fl1z`, training; section 5) | `plans_fl1s` | neither other network works here: d=2 net falls 23-78 % at 1.0, d=0.1 net lands 17-19 cm low |
| <= 0.5 | `fl01z/ppo_dr/it75` | `plans_fl01s` | stage at 0.1: flat + < 10 -0.09 [-0.16, -0.04]; zero-shot on robots at 0.5 usable (nerr 1.74 / 2.95, falls 0 / 6 %), its stage there: section 5 |

Networks trained AT damping 0.5 (`fl05z`) failed and are not to be flown: PPO never beat its start; ours + stage got worse on boxes
>= 10 cm (+0.45 [+0.25, +0.70], falls 3 -> 14 %).

Why this is a hard gate: the d=2 network on robots at other dampings, zero-shot, held-out goals, 4 robots:

| Robot damping | Falls flat / < 10 / >= 10 | nerr boxes >= 10 |
|---|---|---|
| 2 | 0 / 0 / 0 % | 1.33 |
| 1.5 | 0 / 0 / 3 % | 3.15 (jumps 7-13 cm high, stays up) |
| 1.0 (3 robots) | 17 / 42 / 83 % | 5.04 |
| 0.5 | 50 / 59 / 84 % | 5.50 |
| 0.1 | 78 / 100 / 100 % | 5.00 |

If the reading is between the bands, or the first jumps look wrong: re-solve at the reading and retrain, it takes one night on matcha:

    OUT=plans_flX DAMP=<reading> NPAR=12 bash make_plans_fl.sh    # ~3 min; then check every plan's stance and modes
    python3 make_bank_fl.py plans_flX bank_flX.json
    INIT=$PWD/init_zero_out.npz LR=3e-4 ITERS=30 bash fl_train2.sh <reading> flX <gpu>    # ~3.5 h, ours then PPO; use ppo_dr/it50

Start from `init_zero_out.npz` below damping 2. The damping-2 initial network drives low-damping jumps into the box face (6 of 7 held-out goals hit at d=0.1; zero output: 1 of 7).

## 2. Results (12 runs, 3 seeds x 4 robots, held-out goals, paired against the run's own zero-shot)

nerr = max(|ex| / 5 cm, |ez| / 3 cm), a fall = 5. Scored on plain held-out robots (`eval_planeval_start/final`). `stage_fl/st.py NET plain` reproduces it, run from stage_fl/.

### Damping 2

| Arm | Goals | Zero-shot | + stage | Change | Falls |
|---|---|---|---|---|---|
| **PPO+DR** | flat + < 10 | 1.20 | 0.88 | **-0.32 [-0.49, -0.14]** | 0 -> 0 % |
| **PPO+DR** | boxes >= 10 | 1.33 | 0.97 | **-0.36 [-0.51, -0.20]** | 0 -> 0 % |
| **PPO+DR** | all | 1.26 | 0.92 | **-0.34 [-0.50, -0.17]** | 0 -> 0 % |
| ours | flat + < 10 | 0.92 | 0.93 | +0.01 [-0.14, +0.15] | 0 -> 0 % |
| ours | boxes >= 10 | 1.17 | 0.96 | **-0.21 [-0.36, -0.07]** | 2 -> 1 % |
| Old plans, PPO+DR | flat + <= 10 | 1.35 | 1.05 | -0.30 [-0.55, -0.09] | |
| Old plans, PPO+DR | boxes 10-20 | 1.37 | 1.56 | +0.19 [-0.06, +0.46] | 10 % |

PPO+DR, boxes >= 10 per robot: r1 0.97 -> 0.64, r4 2.01 -> 1.27, s1 1.04 -> 0.73, r5 1.30 -> 1.25. Positivity median +0.48, 87 % > 0.

18-20 cm (eval set plane20, 4 jumps per robot and goal, paired with the network's own zero-shot): arm A, which never practises above
15 cm, +0.20 [-0.13, +0.53], falls 5 -> 7 %. Tested 2026-11-02: arm A + practising (0.55, 0.20) (`fl2s20_FR`, 12 runs): unpractised
18-20 cm **-0.22 [-0.39, -0.05]** but falls 6 -> 10 % (r5 12 -> 25 %), and below 16 cm weaker than arm A (flat + < 10 cm -0.18
[-0.43, +0.11], final 1.04 vs 0.88; boxes >= 10 -0.36, same). Not adopted (rule fixed before the result: adopt only if better at
20 cm without losing on the other goals). Arm A's hardware goal set stays at <= 15 cm practised, <= 16 cm held out.

Arm A scored from perturbed starts and timing (final701), PPO+DR, final 12 of 12 runs: +0.01 [-0.14, +0.19], no improvement, falls 0 -> 2 % (flat 0 -> 3 %). Ours: +0.46 [+0.23, +0.78], worse. Fly PPO+DR.

**Arm B (PPO+DR + perturbed starts + stage; each goal <= 10 cm from its own fixed stance + the stance anchor), 12 runs: no
simulation support.** From the softened perturbed set (soft701: front / rear blocks 1.5 / 2 cm, crouch, nose-down, half tall, half
nose-up) flat + < 10 cm -0.11 [-0.27, +0.07]; from the NOMINAL stance worse than its zero-shot, +0.37 [+0.01, +0.76]. Per stance:
half nose-up -1.14 [-1.62, -0.65], front blocks 2 cm -0.42 [-0.82, -0.03]; half tall +0.47, crouch +0.13, rear blocks +0.2 (n.s.).
Variants tested with a rule fixed before their results (stage_fl/ARMB_VARIANTS_1102.md): linearized at the jump's own stance -0.09 /
nominal +0.24; stance anchor x3 -0.14 / nominal +0.29: neither adopted. Control (arm B's goals and flags, nominal starts): nominal
-0.38 [-0.52, -0.24]: the perturbed-stance practice causes the nominal loss. At damping 0.1 arm B: +0.01 [-0.06, +0.08].

### Damping 0.1 (only if the damping test reads <= 0.3)

Network: `fl01z/ppo_dr/it75` (PPO+DR from the zero-output start; it75 = the lowest GPU DR error of it25-it100: 4.12, falls 7.0 %; the
same selection rule that picked it50 at damping 2). Plans `plans_fl01s`.

| Arm | Goals | Zero-shot | + stage | Change | Falls |
|---|---|---|---|---|---|
| PPO+DR it75 | flat + < 10 | 1.16 | 1.07 | **-0.09 [-0.16, -0.04]** | 0 -> 1 % |
| PPO+DR it75 | boxes >= 10 | 1.41 | 1.36 | -0.05 [-0.21, +0.13] | 5 -> 2 % |
| PPO+DR it75 | all | 1.28 | 1.21 | -0.07 [-0.17, +0.03] | 2 -> 1 % |
| ours (zero start, it30) | flat + < 10 | 1.36 | 1.38 | +0.02 [-0.05, +0.10] | 0 -> 1 % |
| ours (zero start, it30) | boxes >= 10 | 2.64 | 2.08 | **-0.56 [-0.88, -0.29]** | 3 -> 2 % |
| PPO+DR it50 (not deployed) | all | 1.40 | 1.25 | -0.15 [-0.30, +0.00] | 5 -> 3 % |

Perturbed starts and timing, PPO+DR it75 (arm A's network): -0.01 [-0.19, +0.17]. At damping 0.1 the stage's gain is small and
significant only below 10 cm; the damping-2 result is the strong one. Fly PPO+DR it75 (the hardware arms are PPO+DR); r1's boxes
>= 10 got worse (0.95 -> 1.24) while r4 / r5 improved.

18-20 cm (zero-shot, eval set plane20 = (0.50|0.55|0.60, 0.20), (0.525, 0.20), (0.575, 0.18), 4 jumps per robot): damping 2 nerr
1.96, falls 5 %; damping 0.1 it75 2.43, falls 16 % (r5 falls every jump on (0.575, 0.18) and (0.60, 0.20)). The damping-2 stage,
which never practises above 15 cm, does not help there: +0.20 [-0.13, +0.53], falls 5 -> 7 % (12 runs). r5's 20 cm jumps are
feasible in the model (r5feas/README.md: the TO converges at r5's mass, motors and motor curve with 10 % torque margin).

## 3. Copy to the robot workstation (Nothing here is on it yet)

All default off without these plans; plans carry their own settings. Everything is in one archive with checksums:
`log/dilc/plane/deploy_bundle_1102/` (`quad_deploy_1102.tgz`, 96 files: the ilc_quad package without experiments/, incl. the new
`quad_eval_session.py`; the overlay; `plans_fl2s` + `plans_fl01s` (ref_*.npz); the policies `fl2s/ppo_dr/it50/policy`,
`fl01z/ppo_dr/it75/policy`).

    # on the robot workstation, in the workspace root (the folder holding src/ and log/; bank.json holds /ilc_ws paths: --path-map if yours differ)
    tar xzf quad_deploy_1102.tgz && bash verify.sh .        # must print "ALL 96 FILES OK"; a changed or missing file fails
    colcon build --packages-select ilc_quad && source install/setup.bash
    export PYTHONPATH=$PWD/log/dilc/plane/fixbox/overlay:$PYTHONPATH     # the overlay first

Tested 2026-11-02: unpacked into an empty folder, verify.sh OK; one edited file -> FAILED, exit 1.

## 4. Commands

GPU machine (`cd ~/ilc_ws/src/ilc_mjx`). D = the bank's damping, 2 or 0.1. These env lines are required: without them the target height is wrong by 13 cm.

```bash
export ILC_SIM_DAMPING=D ILC_FREE_LANDING=2 ILC_LAND_ZMIN=0.14 ILC_FTH_MAX=1.6 ILC_LAND_FEET=0.30,0.25
export ILC_QUAD_OVERLAY=/ilc_ws/log/dilc/plane/fixbox/overlay
PYQ=~/miniconda3/envs/ilcmjx/bin/python
NET=/home/henry/ilc_ws/log/dilc/plane/fl2s/ppo_dr/it50/policy     # or fl01z
COMMON="--backend manual --robots go1 --iters 6 --policy $NET --member policy --update gn --est-window 0.025 --target-base own --gn-beta 0.5 --gn-delta 0.02 --step-rms 0.1 \
  --clip-mode weighted --nbr-radius 0.125 --clear 0 --rollback 1.5 0.5 --gate 1000 1000 1000 --seed 9001 --eval none"
$PYQ scripts/deploy.py $COMMON --out <RUN_A> --goals 0.45 0.60 0.50,0.15 0.60,0.10 0.50,0.05 0.575,0.15
# arm B (no sim support, see section 2; if flown: boxes <= 10 cm, each goal its own stance + the stance anchor)
$PYQ scripts/deploy.py $COMMON --out <RUN_B> --goals 0.45 0.60 0.60,0.10 0.50,0.05 --train-starts 1.0 --starts-fixed --sim-start nominal --anchor-pert
```
Arm B's hardware path was tested on a sim robot (stage_fl/hwsm4: REQUEST.json carries each goal's stance; the robot stood in it,
front thigh 0.836 vs 0.886 rad nominal; 2 iterations, 4 jumps, both exit 0).

Robot workstation, as before; `hw_session.py` now reads `flight_spin_change_deg` and `ilc_landing_state` from each plan:

```bash
python3 hw_session.py <RUN_A>/go1 --pose-topic <topic> --pose-type pose --pose-latency <measured> --mocap-offset x y z
```

Box placement: `hw_session.py` prints it per goal; the front face is x / 2 ahead of the standing CoM.

Tested end to end with a sim robot (`stage_fl/hwsm2`, `deploy.py --backend manual` + `hw_session.py --backend sim`, two iterations, 4 jumps, both exit 0). Not tested on the bridge, the real mocap or the IMU.

## 5. Damping between the networks (2026-11-02)

**Reading <= 0.5: `fl01z/ppo_dr/it75` + stage.** On robots at damping 0.5 (the network and its GPU sim at 0.1; 12 runs, held-out):

| Goals | Zero-shot | + stage | Change | Falls |
|---|---|---|---|---|
| flat + < 10 | 1.74 | 1.58 | **-0.16 [-0.23, -0.09]** | 0 -> 0 % |
| boxes >= 10 | 2.95 | 2.57 | **-0.38 [-0.58, -0.17]** | 6 -> 4 % |
| all | 2.34 | 2.08 | **-0.27 [-0.40, -0.14]** | 3 -> 2 % |

Commands as section 4 with the d=0.1 network and plans (`ILC_SIM_DAMPING=0.1`); the robot's damping is whatever it is.

**Reading 0.6 to 1.4: `fl1s/ours_s2` member `g0_plane_it15` (ours, trained at damping 1.0; plans `plans_fl1s`) + stage.** On robots at 1.0
(12 runs, held-out): flat + < 10 -0.14 [-0.25, -0.04], boxes >= 10 -0.28 [-0.41, -0.17], all **-0.21 [-0.31, -0.14]**, falls 1 -> 1 %.
From perturbed starts (final701) arm A got WORSE: +0.78 [+0.48, +1.06] (full nose-up stance +3.17: the stage's jumps from it land
23.7 cm short). FIX ADOPTED (rule fixed before the result, stage_fl/ARMB_VARIANTS_1102.md): arm A + the stance anchor `--anchor-pert`
(12 runs): nominal -0.18 [-0.26, -0.09] (within 0.05 of arm A's -0.21), perturbed -0.04 [-0.21, +0.13], nose-up -0.02. In this band
fly arm A WITH `--anchor-pert` (section 4 command + `--anchor-pert`). PPO+DR at 1.0 (and 0.5) failed in the
deployed recipe; it is being retrained with the baseline audit's cost cap (`ppo_cc.sh`, `<tag>/ppo_dr_cc`). The d=0.1 network +
stage on robots at 1.0 improves -1.00 [-1.23, -0.76] but ends ~14 cm off: not flyable.

**Not to fly:** networks trained at damping 0.5 (`fl05z`): PPO never beat its start (DR error 9.03 at it0; 10.6 to 12.8 after);
ours + stage worse on boxes >= 10 (+0.45 [+0.25, +0.70], falls 3 -> 14 %).

## 6. Hardware evaluation: `quad_eval_session.py` (new 2026-11-02; tested on a sim robot)

The arms are scored on held-out goals with no learning: `src/ilc_quad/scripts/quad_eval_session.py` (copy it with the other scripts)
writes one REQUEST.json at a time for the unchanged `hw_session.py`, alternating the policies block by block (order reversed every
round), and ends the session with summary.json. One folder per condition. On the GPU machine:

```bash
E=~/quad_eval/$(date +%m%d)           # one folder per condition below
ZS=/home/henry/ilc_ws/log/dilc/plane/fl2s/ppo_dr/it50/policy           # (d = 0.1: fl01z/ppo_dr/it75/policy)
HG="0.4375 0.5625 0.4625,0.06 0.6125,0.07 0.53,0.11 0.56,0.12 0.51,0.13 0.52,0.16"   # held-out planeval goals, never practised
# arm A vs zero-shot, nominal stance, 2 rounds x 2 jumps = 4 jumps per goal per policy
python3 src/ilc_quad/scripts/quad_eval_session.py $E/nominal --policy zs=$ZS A=<RUN_A>/go1/it6/policy --goals $HG --reps 2 --rounds 2
# arm B vs zero-shot from perturbed starts (boxes <= 10 cm, as arm B practises): one folder per stance
for ST in tallh noseuph crouch; do
python3 src/ilc_quad/scripts/quad_eval_session.py $E/$ST --policy zs=$ZS B=<RUN_B>/go1/it6/policy --goals 0.4375 0.5625 0.4625,0.06 0.6125,0.07 --reps 2 --rounds 2 --start $ST
done
python3 src/ilc_quad/scripts/quad_eval_session.py $E/blk15 --policy zs=$ZS B=<RUN_B>/go1/it6/policy --goals 0.4375 0.5625 0.4625,0.06 0.6125,0.07 --reps 2 --rounds 2 --note "front-feet block 1.5 cm"
# score (any time; per goal and policy, and the change vs zs)
python3 src/ilc_quad/scripts/quad_eval_session.py $E/nominal --score
```
Robot workstation, for each folder in turn (exits by itself when the evaluation is done):
`python3 hw_session.py $E/<condition> --pose-topic <topic> --pose-type pose --pose-latency <measured> --mocap-offset x y z`
(`--path-map` if the folder is mounted elsewhere). The stance comes from each request, no `--start` needed.
Tested 2026-11-02 with `hw_session.py --backend sim` (stage_fl/evsm: 2 policies x 2 goals, half-tall stance, 4 blocks, 8 jumps,
stance applied: front thigh 0.838 vs nominal 0.886 rad; scored).

## 7. What to watch on the first real jumps

- Landing: the legs arrive folded, CoM 14 cm above the surface, calf about -2.5 rad, 0.3 rad from the stop. At 2 m/s. Stops and foot compliance are a known sim gap. Foam or the catch rig first.
- Take-off pitch is up to 44 deg (was 33), rear calf at the torque margin on d=2 plans (32 N m).
- The front thigh swings further in flight at low damping (up to 1.6 rad).
- Landing pitch at N runs +3 to +8 deg on these plans in per-goal ILC, not the paper's 1 deg.
- r5 (mass x1.14, motor x0.94) is the weak case: per-goal it reaches <= 2 cm on most boxes but not (55,20) or (60,20) at d=0.1.
