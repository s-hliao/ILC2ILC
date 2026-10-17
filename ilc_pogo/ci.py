"""
ci.py: confidence intervals for the pogo-stick success rates. The unit of resampling is the (robot, goal) cluster:
the 4 perturbed drops of a goal on a robot are correlated, so resampling single runs would understate the
uncertainty. Seeds of an arm are pooled inside each cluster (the same robots and goals in every seed).
    cluster_ci([summary, ...])            -> (mean, lo, hi) in [0, 1]
    paired_ci([sum_a, ...], [sum_b, ...]) -> (mean diff a - b, lo, hi), the same clusters resampled for both arms
    wilson(k, n)                          -> (p, lo, hi), the run-level interval (a check: it ignores the clustering)
"""
import numpy as np

B_DEFAULT = 10000


def clusters(summaries):
    """{(robot, goal): [success, ...]} pooled over the summaries (seeds)."""
    out = {}
    for s in summaries:
        runs = s['runs'] if isinstance(s.get('runs'), dict) else s['per_run']
        for r, ms in runs.items():
            for m in ms:
                out.setdefault((r, tuple(np.round(m['goal'], 6))), []).append(float(m['success']))
    return out


def _boot(means, weights, B, seed):
    rs = np.random.default_rng(seed)
    idx = rs.integers(0, len(means), (B, len(means)))
    return (means[idx] * weights[idx]).sum(1) / weights[idx].sum(1)


def cluster_ci(summaries, B=B_DEFAULT, alpha=0.05, seed=0):
    cl = clusters(summaries)
    m = np.array([np.mean(v) for v in cl.values()])
    w = np.array([len(v) for v in cl.values()], float)
    bs = _boot(m, w, B, seed)
    return float((m * w).sum() / w.sum()), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def paired_ci(sa, sb, B=B_DEFAULT, alpha=0.05, seed=0):
    ca, cb = clusters(sa), clusters(sb)
    keys = sorted(set(ca) & set(cb))
    d = np.array([np.mean(ca[k]) - np.mean(cb[k]) for k in keys])
    w = np.ones(len(d))
    bs = _boot(d, w, B, seed)
    return float(d.mean()), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, c - h, c + h
