"""solve_plan.py LOCKSTEP_ARGS...: ilc_jump_lockstep.py with the TO's IPOPT iteration cap
raised to $TO_MAX_ITER (default 3000) -- the node has no parameter for it, and the harder
box plans need 600-1300 iterations. Everything else is the node's own parameter flow, so
the saved plan carries exactly the config later runs check it against."""
import os, runpy, sys
from ilc_quad import ilc_gen

_init = ilc_gen.QuadILCStageSolver.init_trajopt


def init_trajopt(self, *a, **kw):
    kw["ipopt_options"] = {**(kw.get("ipopt_options") or {}),
                           "max_iter": int(os.environ.get("TO_MAX_ITER", 3000))}
    return _init(self, *a, **kw)


ilc_gen.QuadILCStageSolver.init_trajopt = init_trajopt
lockstep = os.path.join(os.environ["WS"], "install/ilc_quad/lib/ilc_quad/ilc_jump_lockstep.py")
sys.argv = [lockstep] + sys.argv[1:]
runpy.run_path(lockstep, run_name="__main__")
