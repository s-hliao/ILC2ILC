#!/usr/bin/env python3
"""
reeval.py ARM_DIR [...]: re-evaluate frozen arms on the current evaluation set (POGO_EVAL_SET; 'test' for the single
final pass) -> ARM_DIR/summary_<set>.json in the summary format (eval, runs, real_trials and positivity copied from
the arm's summary.json). The per-robot networks ARM_DIR/<robot>_policy.npz are flown if present (adapted arms, FADA);
otherwise the arm's starting policy (args.policy of a zero-shot hw_stage arm). A checkpoint dir ARM/it<k> of a curve
run (hw_stage --eval-at) reads its record from ARM/summary_it<k>.json.
    POGO_EVAL_SET=test python reeval.py runs/v3/nom_s0_curve ...
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                               # noqa: E402
import policies                                  # noqa: E402
import real_pogo as rp                           # noqa: E402


def policies_of(d, robots):
    if all(os.path.exists(os.path.join(d, f'{r}_policy.npz')) for r in robots):
        return {r: policies.load_any(os.path.join(d, f'{r}_policy.npz')) for r in robots}
    args = json.load(open(os.path.join(d, 'args.json')))
    base = policies.load_any(args['policy'])
    return {r: base for r in robots}


def main():
    for d in sys.argv[1:]:
        t0 = time.time()
        d = d.rstrip('/')
        src = os.path.join(d, 'summary.json')
        if os.path.basename(d).startswith('it') and not os.path.exists(src):      # a saved checkpoint of a curve run
            src = os.path.join(os.path.dirname(d), f'summary_{os.path.basename(d)}.json')
        s = json.load(open(src))
        robots = list(s['eval'])
        pols = policies_of(d, robots)
        goals = C.eval_goals(len(s.get('goals_eval', [None] * 8)))
        runs = rp.evaluate(pols, goals, robots, runs=4, seconds=10.0)
        out = dict(eval_set=C.EVAL_SET, eval_seed=C.EVAL_SEED, goals_eval=goals.tolist(),
                   eval={r: rp.summarize(runs[r]) for r in robots}, runs=runs,
                   real_trials=s.get('real_trials', {}), positivity=s.get('positivity', {}),
                   source=src, seconds=time.time() - t0,
                   time=time.strftime('%Y-%m-%d %H:%M:%S'))
        tag = os.environ.get('POGO_EVAL_TAG', '')                        # e.g. 2301: summary_test2301.json
        json.dump(out, open(os.path.join(d, f'summary_{C.EVAL_SET}{tag}.json'), 'w'), indent=1)
        print(d, C.EVAL_SET, json.dumps({r: round(v['success'], 3) for r, v in out['eval'].items()}), flush=True)


if __name__ == '__main__':
    main()
