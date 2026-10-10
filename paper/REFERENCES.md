# DistILC: related work

Every reference collected for the paper (2026-10-07 to 2026-10-10), grouped by how it relates to DistILC, closest
prior art first. Each entry appears once; detailed notes and the contrast with our method follow the entries we read.

**Status of each entry**
- **[READ]**: full text (or the stated part) read.
- **[abstract]**: abstract / metadata only.
- **[web]**: a web search confirmed it exists; not read in full.
- **[DBLP]**: bibliographic record only.
- **[memory]**: cited from memory, NOT verified -- check authors / venue / year before citing.

## Positioning (the one-paragraph answer to "how is this different?")

Neural-network ILC (Section 1) runs ILC per reference to (near) convergence, distills the converged FEEDFORWARD into a
network that generalizes across references, keeps a fixed hand-designed feedback loop, and works on smooth tracking of
stable plants. Guided policy search (Section 2) distills trajectory optimization, on hardware through dynamics fitted
from robot samples. DistILC trains a goal-conditioned FEEDBACK policy by regressing onto ONE approximate-model ILC
step per trial (no goal is ever converged), and reuses that same update on the real system -- Jacobians from the
nominal simulator, the measured outcome -- so every real trial updates a network shared across goals: a few trials
adapt it, including goals never flown. Hybrid, contact-rich agile motion (quadruped jumps onto boxes, drifting car)
against sim-to-real RL baselines (PPO+DR, RMA, FADA), also trained on the real systems with privileged data.

What NOT to claim: that a network generalizing ILC across references is new (Arif 2002; Zhang, Wang & Tomizuka 2021;
Chen & Wen 2021), or that fast goal-conditioned iterative correction under a reality gap is new (Chi et al. 2022).
Main exposure: Chen & Wen (same pipeline shape), Ma et al. (generalization + their feedforward-only stability
argument, which our structure probe partly supports -- address it head-on), guided policy search (the conceptual
parent: say why a cheap approximate-Jacobian ILC teacher is preferable to a full trajectory optimizer with fitted
dynamics).

## 1. Neural networks that carry ILC across references (closest prior art)

- [DBLP] **Arif, Ishihara & Inooka**, "Generalization of Iterative Learning Control for Multiple Desired Trajectories in
  Robotic Systems", PRICAI 2002, pp. 295-304; also "Experience-Based Iterative Learning Controllers for Robotic
  Systems", J. Intelligent & Robotic Systems 35(4):381-396, 2002. UNREAD (DBLP record only).
  - Per the reviewer: a network stores ILC experience and initializes ILC on new trajectories. Cite as the earliest
    "a network carries ILC across references" precedent.
- [web] "Realization of a neural network controller by using iterative learning control", Korean conference, 1992 --
  ILC generates the training data for a neural controller.
  https://koreascience.or.kr/article/CFKO199211919699902.page?lang=en
- [READ via HTML] **Chen & Wen**, "Industrial Robot Trajectory Tracking (Control) Using Multi-Layer Neural Networks
  Trained by Iterative Learning Control", Robotics (MDPI) 10(1):50, 2021; arXiv 1903.00082. https://arxiv.org/abs/1903.00082
  - Network: one MLP per joint (2 x 100 ReLU), an approximate inverse of the robot's inner loop, FEEDFORWARD: a
    non-causal window of the desired trajectory (50 samples, 0.1 s at 4 ms) -> 25 command samples. ABB IRB6640-180.
  - Data: gradient-descent ILC (line search) in the RobotStudio simulator on many smooth trajectories -> (desired,
    refined command) pairs; offline training. Sim-to-real: ILC on the physical robot on 20 trajectories, only the output
    layer fine-tuned. No learned feedback ("no closed-loop action, the system is always stable"). Joint 1, 8 rad/s
    sinusoid: l2 error 58.7 deg -> 14.0 (sim-trained) -> 13.7 (after transfer). Needs a high-fidelity simulator.
  - CONTRAST (closest pipeline shape: sim ILC -> network -> real ILC fine-tune): feedforward inverse vs goal-conditioned
    feedback policy; imitation of per-trajectory ILC solutions vs one-step online regression; real fine-tuning from ILC
    on 20 trajectories vs our 24 jumps (or 2-10 laps) per robot with the same Gauss-Newton update as the sim stage,
    every trial updating all goals; high-fidelity vs deliberately nominal simulator; smooth joint tracking vs hybrid
    contact; no RL / DR baselines.
- [abstract] **Zhang, Wang & Tomizuka**, "Neural-Network-Based Iterative Learning Control for Multiple Tasks", IEEE
  TNNLS, 2021, doi 10.1109/TNNLS.2020.3017158. https://ieeexplore.ieee.org/document/9186361/
  - Position-based ILC compensates each task; the ILC outputs over many tasks are written as a function of the
    reference (position, velocity, acceleration): a linear part + a nonlinear part from complementary networks (general
    + switching). New references get a compensation without new ILC. Robot arm experiments.
  - CONTRAST: distills converged per-task FEEDFORWARD compensation; ours is a feedback policy trained by one ILC step
    per trial, the same update reused on a new robot.
- [READ] **Ma, Büchler, Schölkopf & Muehlebach**, "A Learning-based Iterative Control Framework for Controlling a Robot
  Arm with Pneumatic Artificial Muscles", RSS 2022. https://www.roboticsproceedings.org/rss18/p029.pdf
  - Network: per-DoF CNN (6 conv + 4 FC, tanh) = FEEDFORWARD; input a non-causal window (+-1 s) of the reference + a
    linear-regression channel; output u_ff[k].
  - Data: ILC run to convergence (40 iterations) ON THE REAL ARM for each of 44 references (lifted model from
    frequency-domain system ID + Kalman disturbance estimate); trained on 31, validated on 13; ~1,760 real trials.
  - Feedback: a FIXED H-inf PID, deliberately not learned (feedforward cannot destabilize an ISS system). CNN + PID
    beats ILC on unseen references (< 0.04 m). No simulation stage.
  - CONTRAST: we learn the closed-loop FEEDBACK policy (they argue against it; our structure probe partly agrees: the
    learned feedback amplified late-push disturbances -- say so); converged per-reference ILC distilled offline vs one
    step per trial online; ~1,760 real trials vs 24 per robot after a nominal-sim stage; smooth tracking vs contact-rich
    jumps; no sim-to-real, no RL / DR comparison.
- [abstract] **Lakshmidevinivas et al.**, "Neural Network Augmented Intelligent ILC for a Nonlinear System", IJCNN 2020,
  doi 10.1109/IJCNN48605.2020.9207260.
  - A network trained offline on logged ILC histories approximates the CONVERGED ILC; its output is the first iterate
    of a new ILC run. CONTRAST: a warm start for per-condition ILC; ours is the policy, no per-goal ILC run follows.
- [abstract] **Patan & Patan**: "Design and convergence of ILC based on neural networks" (ECC 2018); "Neural-network-based
  high-order ILC" (ACC 2019); "Robustness of neural-network-based nonlinear ILC" (ICARCV 2020).
  - The network IS the ILC learning operator (and a model) for ONE repetitive process (pneumatic servo, maglev).
    CONTRAST: the network implements the per-task ILC iteration; ours is the deployed multi-goal policy, ILC its
    training rule. (Taken acronym: "GILC" = generalized ILC, ISNN 2006.)
- [title only, UNREAD] **Erens**, "Interpolation and neural network based ILC for coping with new references without
  relearning", MSc thesis, TU Eindhoven, 2021 (PDF 0997909_Erens.pdf on research.tue.nl).
- [memory] **Li, Zhou & Schoellig**, "Deep neural networks for improved, impromptu trajectory tracking of quadrotors",
  ICRA 2017.
- [memory] **Pereida, Helwa & Schoellig**, "Data-efficient multirobot, multitask transfer learning for trajectory
  tracking", RA-L 2018.
- [memory] **Schoellig et al.**, optimization-based iterative learning for quadrocopter trajectory tracking (ILC
  background).

## 2. Trajectory optimization distilled into one network

- [web] **Levine & Koltun**, "Guided Policy Search", ICML 2013. https://graphics.stanford.edu/projects/gpspaper/gps_full.pdf
  - The conceptual parent: alternate trajectory-level optimization and supervised learning of a neural feedback
    policy. Hardware variants fit local dynamics from robot trials. CONTRAST: our teacher is one damped Gauss-Newton
    step on the measured outcome through the NOMINAL simulator's Jacobians -- no fitted dynamics, no system ID, no
    solve to optimality; one policy shared across goals, starting from a sim stage. TODO: verify the real-robot sample
    counts of the hardware GPS papers before arguing "fewer trials" against them.
- [memory] **Levine, Finn, Darrell & Abbeel**, "End-to-End Training of Deep Visuomotor Policies", JMLR 2016 (GPS on a
  real robot).
- [memory] **Mordatch & Todorov**, "Combining the benefits of function approximation and trajectory optimization", RSS 2014.
- [READ via HTML] **Goikoetxea & Palacián**, "GCImOpt: Learning efficient goal-conditioned policies by imitating optimal
  trajectories", L4DC 2026, PMLR 331:574-588; arXiv 2604.22724. https://proceedings.mlr.press/v331/goikoetxea26a.html
  - Behaviour cloning of a goal-conditioned MLP (state, goal -> control) on 20,000 FATROP-optimal trajectories per task;
    intermediate states relabelled as goals (~11-15 M samples). Cart-pole, planar / 3D quadrotor, Panda reaching.
  - ALL SIMULATED; they assume an accurate model and full state; sim-to-real and adaptation are future work.
  - CONTRAST: offline optimal demonstrations under an exact model vs error-driven closed-loop ILC targets under a wrong
    model, plus a hardware stage. SAME VENUE as our target: cite and position directly.
- [web, search summary] **Panichi et al.**, 2025 journal article: dual-layer coarse-to-refine trajectory optimization +
  variable-impedance landing, distilled into a network by behaviour cloning (quadruped jumping). Title / venue to find.
- [web] "Online Trajectory Planning Through Combined Trajectory Optimization and Function Approximation: Application to
  the Exoskeleton Atalante", arXiv 1910.00514. https://arxiv.org/pdf/1910.00514
- Closed-loop pitfalls of distillation (support our structure-probe finding):
  - [web] "Closed-loop optimisation of neural networks for the design of feedback policies under uncertainty",
    ScienceDirect 2023 -- a network fit to MPC solutions can have near-zero training error yet poor closed-loop
    performance; train in closed loop. https://www.sciencedirect.com/science/article/pii/S0959152423002329
  - [web] "Closed-loop training of static output feedback neural network controllers for large systems: A distillation
    case study", arXiv 2402.19309. https://arxiv.org/html/2402.19309v1
  - [web] "Learning Lipschitz Feedback Policies from Expert Demonstrations: Closed-Loop Guarantees, Generalization and
    Robustness", arXiv 2103.16629. https://arxiv.org/pdf/2103.16629

## 3. Few real trials with an approximate model (precedent for the hardware stage)

- [READ] **Abbeel, Quigley & Ng**, "Using Inaccurate Models in Reinforcement Learning", ICML 2006, doi
  10.1145/1143844.1143845. https://ai.stanford.edu/~ang/papers/icml06-usinginaccuratemodelsinrl.pdf
  - Start from the model's DDP-optimal policy; fly it; add a time-dependent bias so the corrected model reproduces the
    real trajectory (derivatives along the TRUE trajectory); direction = DDP run to completion in the corrected model
    minus the current policy; step by a LINE SEARCH ON THE REAL SYSTEM. Theorem: converges to a region
    ||grad U||^2 <= 2 K^2 eps^2 scaled by the derivative error. Flight sim +76 % in ~5 real trials; real RC car
    +97 / 88 / 63 % after 5-10 iterations. One maneuver per controller. They call it "an instance of ILC".
  - CONTRAST: they optimize a utility (approximate gradient -> biased stationary region); we root-find a measured
    residual (approximate Jacobian -> exact fixed point under positivity). One damped step, a goal-conditioned network,
    a sim stage, noisy robots. Tested as an arm (hw_abbeel: 36 % vs our 33-35 %, a tie; their derivative point alone,
    --jac-at real, 25 %).
- [web] **Kolter**, "Learning and Control with Inaccurate Models", PhD thesis, Stanford, 2010.
  https://zicokolter.com/publications/kolter2010thesis.pdf
- [READ via HTML] **Chi, Burchfiel, Cousineau, Feng & Song**, "Iterative Residual Policy for Goal-Conditioned Dynamic
  Manipulation of Deformable Objects", RSS 2022 (best paper); arXiv 2203.00663. https://irp.cs.columbia.edu/
  - Learns DELTA DYNAMICS: (observed trajectory image, delta action) -> new trajectory image (DeepLabV3+), from 54 M
    simulated rope trajectories; at test time samples 128 delta actions per trial and executes the best; <= 10 real
    trials; actions are 3 (rope) / 4 (cloth) parameters. The network is NOT updated on hardware.
  - CONTRAST: trial-to-trial search over a few open-loop parameters with a frozen learned model, per goal and repeated
    every time; ours updates a closed-loop policy's weights (persists, transfers to unflown goals), uses the nominal
    simulator's Jacobians instead of a learned delta model, ~10^3-10^4 sim episodes instead of 5x10^7.
- [abstract] **Suresh & Atkeson**, "Learning Dynamic Rope Manipulation Using Task-Level Iterative Learning Control",
  arXiv 2602.21302, 2026. https://arxiv.org/abs/2602.21302
  - Task-level ILC inverting a simplified robot + rope model (QP); one demonstration; learns directly on hardware;
    100 % within 10 trials on 7 ropes. CONTRAST: supports our thesis (approximate model + real trials); per task, no
    policy, no generalization across goals, no sim stage.
- [READ pp. 1-6] **Pereida, Kooijman, Duivenvoorden & Schoellig**, "Transfer Learning for High-Precision Trajectory
  Tracking Through L1 Adaptive Feedback and Iterative Learning", Int. J. Adaptive Control & Signal Processing, 2018;
  arXiv 1807.05289.
  - An L1 adaptive inner loop forces a reference model; ILC learns the reference input per trajectory; that input
    transfers between quadrotors. Assumes SISO, minimum-phase, PI-stabilizable plants. CONTRAST: transfer by MAKING the
    dynamics equal vs CORRECTING a policy with grounded updates; per trajectory vs a goal-conditioned network; their
    assumptions fail through contact / flight phases.
- [web] **Poot, Portegies & Oomen**, "On the Role of Models in Learning Control: Actor-Critic Iterative Learning Control",
  IFAC World Congress 2020; arXiv 2007.00430 -- model-free actor-critic over feedforward basis functions; relevant to our
  value-gradient / residual-critic arms. https://arxiv.org/abs/2007.00430v2
- [web] "Bridging Reinforcement Learning and Iterative Learning Control: Autonomous Motion Learning for Unknown,
  Nonlinear Dynamics" -- a GP dynamics model, feedforward optimized per trial (balancing robot).
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9315427/
- [web] **Chatzilygeroudis et al.**, "A survey on policy search algorithms for learning robot controllers in a handful of
  trials", arXiv 1807.02303.
- [web] "What Matters for Sim-to-Online Reinforcement Learning on Real Robots", arXiv 2602.20220, 2026 -- reusing data
  from a few preceding real trials helps.
- [memory] **Deisenroth & Rasmussen**, "PILCO", ICML 2011 -- sample-efficient model-based policy search.

## 4. ILC on legged robots, and quadruped jumping

- [READ: method + experiments] **Gori, Degiacomo, Pierallini, Angelini & Garabini**, "Contact-Implicit Optimal Planning
  and Iterative Learning Control for Quadrupedal Robots", IEEE Trans. Industrial Electronics, 2026, doi
  10.1109/TIE.2025.3645471.
  - 2D single-rigid-body contact-implicit trajectory optimization -> IK -> joint references; PD + a model-free P/D-type
    ILC feedforward torque, tau_ff_k = tau_ff_{k-1} + Kp' e_{k-1} + Kd' de_{k-1}. Solo-12, periodic pronking / bounding
    learned "while doing" over 15-20 iterations; +25 % payload, mixed terrain, grass. Metric: joint RMSE. No network.
  - CONTRAST: per-gait joint-tracking feedforward vs a task-level (landing / path) Gauss-Newton step through model
    Jacobians into one goal-conditioned network; single aperiodic jumps; held-out goals.
- [memory: see reference_nguyen_ilc_jumping] **Nguyen et al.**, arXiv 2408.02619 -- the per-goal JumpILC our baseline
  follows (PD landing).
- [web] **Cheng, Alqaham, Gan & Sanyal**, "Iteratively Learning Muscle Memory for Legged Robots to Master Adaptive and
  High Precision Locomotion", arXiv 2507.13662, 2025 -- ILC + a torque library; joint tracking errors down 85 %. CHECK
  how the library generalizes across tasks. https://arxiv.org/abs/2507.13662v1
- [web] Same group, "Practice Makes Perfect: an iterative approach to achieve precise tracking for legged robots",
  arXiv 2211.11922.
- [web] **Ding et al.**, "Robust Jumping with an Articulated Soft Quadruped via Trajectory Optimization and Iterative
  Learning" (TO + ILC pronking, per-task reference).
  https://research.tudelft.nl/en/publications/robust-jumping-with-an-articulated-soft-quadruped-via-trajectory-
- [web] **Atanassov, Ding et al.**, "Curriculum-Based Reinforcement Learning for Quadrupedal Jumping: A Reference-free
  Design", arXiv 2401.16337, 2024 -- goal-conditioned DRL jumping (landing target, obstacle).
- [web] **Apostolides et al.**, "Explosive Jumping with Rigid and Articulated Soft Quadrupeds via Example Guided
  Reinforcement Learning", arXiv 2503.16197, 2025.
- [web] "Continuous Versatile Jumping Using Learned Action Residuals", arXiv 2304.08663.
- [web] "Impedance Matching: Enabling an RL-Based Running Jump in a Quadruped Robot", arXiv 2404.15096.
- [web] "Learning Task-Specific Dynamics to Improve Whole-Body Control", arXiv 1803.01978 -- iterative learning of
  task-space accelerations; cross-task transfer left as future work.
- [web] "Learning a Structured Neural Network Policy for a Hopping Task", arXiv 1710.00022.

## 5. Sim-to-real baselines and adaptation

- [web] **Peng et al.**, "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization", arXiv 1710.06537 (DR).
- [web] **Zhao, Queralta & Westerlund**, "Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey",
  arXiv 2009.13303.
- [memory] **Kumar et al.**, "RMA: Rapid Motor Adaptation for Legged Robots", RSS 2021 -- our RMA baseline (and the
  cross-trial variant).
- [web] "FADA: Few-Shot Domain Adaptation via Dynamics Alignment for Humanoid Control", arXiv 2606.28476 -- likely the
  FADA our planner + inverse-dynamics + LoRA baseline follows; CHECK.
- [web] "Robot trains robot: Automatic real-world policy adaptation and learning for humanoids", arXiv 2508.12252.
- [web] "Simulator Adaptation for Sim-to-Real Learning of Legged Locomotion via Proprioceptive Distribution Matching",
  arXiv 2604.11090 -- < 5 min of hardware data (Go2) to adapt the simulator (contrast: we estimate no parameters).
- [web] "Learning Deployable Locomotion Control via Differentiable Simulation", arXiv 2404.02887 (SHAC, quadruped).
- [memory] **Ross, Gordon & Bagnell**, "DAgger", AISTATS 2011 -- the cross-trial RMA student's data collection.

## 6. Methods and tools we use

- [memory] **Czarnecki et al.**, "Sobolev Training for Neural Networks", NeurIPS 2017 -- the --k-match Jacobian matching.
- [memory] **Heess et al.**, "Learning Continuous Control Policies by Stochastic Value Gradients", NeurIPS 2015.
- [memory] **Gurumurthy et al.** -- VG-SAC / value-gradient regularization (the critic we audited; find the exact paper).
- [memory] Broyden's method (secant update) -- --sens-adapt.

## 7. The car platform

- [memory] F1TENTH platform (O'Kelly et al., F1TENTH: an open-source autonomous cyber-physical platform; check venue).
- [web] LLA-MPC-onboard (the car's onboard stack: OptiTrack node, EKF, VESC current control; our hardware layer ports
  it). https://github.com/LLA-Control/LLA-MPC-onboard
- [memory] Fiala brush tire model (Fiala 1954; as used in Pacejka, "Tire and Vehicle Dynamics") -- the car's nominal sim.

## Also surfaced (less relevant)

- [web] **Xie et al.**, "Iterative Reinforcement Learning Based Design of Dynamic Locomotion Skills for Cassie", arXiv
  1903.09537 -- expert policies distilled into one.
- [web] Hypernetwork distillation for universal morphology control, arXiv 2402.06570.
- [web] Hierarchical learning control inspired by the CNS (teacher-student + DAgger), arXiv 2408.03525.

## Open items

- Verify every [memory] entry (authors / venue / year), first the GPS papers' real-robot sample counts.
- Read Arif et al. 2002 (only the DBLP record so far) and Cheng et al. 2025 (how the torque library generalizes).
- Searches still to run: "iterative learning control" + "policy gradient" / "neural network" in IEEE TCST, Automatica,
  CDC / ACC 2022-2026; Schoellig-lab multi-task / transfer ILC; GPS on legged robots / agile motions.
- Names: "ILC2Real" (the old name) -- no prior use found (2026-10-08); "DistILC" -- no prior use found (web search,
  2026-10-10).
