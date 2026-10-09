"""A machine-wide cap on concurrent robot flights (each flies 8 CPU robot sims): fly() holds one of N lock files for
its duration. Active only when TOKENS_FILE exists (it holds N) and ILC_QUAD_OVERLAY is set -- the box-fix reruns, which
share a 32-CPU budget with everything else; every other run is unaffected."""
import fcntl, os, time
from contextlib import contextmanager

TOKENS_FILE = "/home/henry/ilc_ws/log/dilc/plane/fixbox/fly_tokens"


@contextmanager
def flight_token():
    n = 0
    if os.environ.get("ILC_QUAD_OVERLAY") and os.path.exists(TOKENS_FILE):
        try:
            n = int(open(TOKENS_FILE).read().strip() or 0)
        except ValueError:
            n = 0
    if n <= 0:
        yield
        return
    d = os.path.dirname(TOKENS_FILE)
    while True:
        for i in range(n):
            fh = open(os.path.join(d, f".fly_token_{i}"), "w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                fh.close()
                continue
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
                fh.close()
            return
        time.sleep(2)
