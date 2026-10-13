"""
scheduler_axes.py: the coverage suite -- every method on single-axis robots (real_nom + one axis moved), the DR range
inside the sweep, plus the hardware stage (and its positivity log) on each.
    POGO_STATE=scheduler_axes POGO_CPU_JOBS=6 nohup python scheduler.py --jobs scheduler_axes > logs/scheduler_axes.out 2>&1 &
One job per (arm, axis): the axis' 7 robots in one hw_stage call. Outputs runs/hw/axes/<arm>_<axis>/summary.json.
"""
AXES = {
    'k': [0.5, 0.65, 0.8, 1.0, 1.2, 1.35, 1.5],        # spring stiffness x (DR +-20 %)
    'm': [0.7, 0.8, 0.9, 1.0, 1.2, 1.35, 1.5],         # mass x (DR +-25 %)
    'gth': [0.5, 0.6, 0.75, 0.9, 1.0, 1.15, 1.3],      # thrust gain x (DR +-20 %)
    'phi_off': [-0.1, -0.06, -0.03, 0.0, 0.03, 0.06, 0.1],   # leg-angle offset rad (DR +-0.03)
    'kp': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],         # progressive spring (DR: none, kp = 0 only)
    'b': [0.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0],    # leg damping N s/m (DR 0-30)
}
DR_RANGE = {'k': (0.8, 1.2), 'm': (0.75, 1.25), 'gth': (0.8, 1.2), 'phi_off': (-0.03, 0.03), 'kp': (0.0, 0.0),
            'b': (0.0, 30.0)}
ARMS = {
    'expert_zs': ('expert', '--iters 0'),
    'nom_s0_zs': ('runs/nom_s0/policy_pick.npz', '--iters 0'),
    'nom_s0_hw24': ('runs/nom_s0/policy_pick.npz', '--iters 4'),
    'drB_s0_zs': ('runs/drB_s0/policy_pick.npz', '--iters 0'),
    'drB_s0_hw24': ('runs/drB_s0/policy_pick.npz', '--iters 4'),
    'ppo_dr_hw24': ('runs/ppo_dr_s0/policy_final.npz', '--iters 4'),
    'ppo_dr_zs': ('runs/ppo_dr_s0/policy_final.npz', '--iters 0'),
    'rma_zs': ('runs/rma_s0/policy_final.npz', '--iters 0'),
}


def robots(axis):
    return [f'ax:{axis}={v:g}' for v in AXES[axis]]


def add_jobs(S):
    for arm, (pol, args) in ARMS.items():
        for axis in AXES:
            rob = ' '.join(f"'{r}'" for r in robots(axis))
            S.arm(f'axes/{arm}_{axis}', pol, f'{args} --robots {rob}')
