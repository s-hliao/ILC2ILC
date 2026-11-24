"""Export a walker network to the numpy npz hw/walker_core.Policy runs (2026-11-21), and check it against the JAX policy.

    ~/venvs/pg0/bin/python hw/export_policy.py --params runs/walker_nom_s0/params.pkl --out hw/policies/nominal.npz
    ~/venvs/pg0/bin/python hw/export_policy.py --params BASE.pkl --stage runs/goalnv_pay4R_s0/policy_it9.pkl --out ...
--params: brax params (normalizer, policy, value). --stage: a stage pickle {norm, pol} (stage_goal_nv / stage_goal_hw), which
replaces the normalizer and policy (the normalizer is the base's, unchanged by the stage).
"""
import argparse, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def to_npz(norm, pol, out):
    st = norm.mean['state'], norm.std['state']
    P = pol['params']
    d = dict(mean=np.asarray(st[0], np.float32), std=np.asarray(st[1], np.float32))
    for i in range(4):
        d[f'W{i}'] = np.asarray(P[f'hidden_{i}']['kernel'], np.float32)
        d[f'b{i}'] = np.asarray(P[f'hidden_{i}']['bias'], np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, **d)
    return d


def check(norm, pol, out, net, n=2000, seed=0):
    """max |numpy - jax| over random observations around the normalizer's statistics (and a few far-out ones)."""
    import jax.numpy as jnp
    from ilc_go1lla import walkenv as WE
    from walker_core import Policy
    rng = np.random.default_rng(seed)
    mu, sd = np.asarray(norm.mean['state']), np.asarray(norm.std['state'])
    obs = mu + sd * rng.normal(size=(n, 48)) * np.r_[np.ones(n - 50), 5 * np.ones(50)][:, None]
    aj = np.asarray(WE.policy_apply(net, (norm, pol), {'state': jnp.asarray(obs, jnp.float32), 'privileged_state': jnp.zeros((n, 123))}))
    p = Policy(out)
    an = np.stack([p(o) for o in obs])
    return float(np.abs(an - aj).max())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', required=True)
    ap.add_argument('--stage', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-check', action='store_true')
    a = ap.parse_args()
    from ilc_go1lla import walkenv as WE
    env = WE.make_env()
    params, net = WE.load_policy(env, a.params)
    norm, pol = params[0], params[1]
    if a.stage:
        pk = pickle.load(open(a.stage, 'rb'))
        norm, pol = pk['norm'], pk['pol']
    to_npz(norm, pol, a.out)
    if not a.no_check:
        err = check(norm, pol, a.out, net)
        print(f'wrote {a.out}; max |numpy - jax| action = {err:.2e}')
        if err > 1e-4:
            raise SystemExit('!!! the numpy policy does not match the JAX one')
    else:
        print(f'wrote {a.out}')
