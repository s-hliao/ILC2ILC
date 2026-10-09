"""Computes and caches the periodic LQR gains (plans/<name>_K.npy) of every plan the runs use, once, before any
parallel job loads the bank (no concurrent writes of the same cache file)."""
import bank
names = [f'mocap_{t}_b{b}' for t in ('square2fast', 'figfast') for b in (0, 10, 14, 18, 21, 25)]
names += [f'mocap_{t}_b{b}_mu80' for t in ('square2fast', 'figfast') for b in (0, 10, 18, 25)]
for n in names:
    bank.load_plan(n)
    print('cached', n, flush=True)
open(bank.PLAN_DIR + '/lqr_cache.done', 'w').write('ok\n')
