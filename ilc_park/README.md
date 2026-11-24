# ilc_park — DistILC drift parking (F1TENTH)

The car circles on a steady drift; on "go" it leaves the circle and must stop in a parking slot (position + heading,
at rest). One goal-conditioned network (PPO in the nominal JAX car), then the DistILC stage on real parks
(drone soccer's structure). Separate from `src/ilc_f1tenth/ilc2real` (continuous track tracking), which is untouched;
the nominal car model is vendored (`ilc_park/car_model.py`, header holds the source md5).

| file | what |
|---|---|
| `ilc_park/trajopt.py` | time-domain Hermite-Simpson TO: drift equilibrium -> slot at rest (feasibility / plans) |
| `scripts/feasibility.py` | slot sweep -> `results/feasibility_R1.5_b15.json`; `results/slots_R1.5_b15.json` = feasible, T < 3.5 s |
| `ilc_park/sim.py` | batched JAX car (sim_jax's integration: 25 ms control, 0.25 ms RK4), drift release table, mirror symmetry |
| `ilc_park/ppo.py` | PPO (nominal) / `--dr` PPO+DR (friction grid 0.36-0.76, each lane released from its own steady drift) |
| `ilc_park/stage.py` | rows (nominal predicted stop pose per moving tick + measured final pose, optional sideslip floor), lifted G, GN step |
| `scripts/stage_run.py` | stage vs a stand-in real car (`--real-mu`, default 0.44); `--positivity N` step-0 check |

Environment: `~/miniconda3/envs/f1t`, `F1T_MU=0.6`, run from this directory.

    python -m ilc_park.ppo --slots results/slots_R1.5_b15.json --out runs/ppo_nominal_s0 --seed 0
    python scripts/stage_run.py --policy runs/ppo_nominal_s0/policy_it0600.npz --slots results/slots_R1.5_b15.json \
        --out runs/stage_nom_s0 --parks 96 --eval-every 12          # --positivity 24 for the step-0 check

The stand-in real car's friction (0.44) is inferred from hardware: the real car held its 13-18 deg drift at 2.4-2.6 m/s
where the nominal model's equilibrium is 2.9 m/s (not identified on hardware; the stage never uses it).
