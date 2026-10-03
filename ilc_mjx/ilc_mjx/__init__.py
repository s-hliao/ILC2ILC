"""ilc_mjx: the jump on the GPU -- many Go1 jumps in parallel in MuJoCo MJX (JAX), for training at scale
(domain randomization baselines, finite-difference Jacobians as batched copies). Synchronous and nominal:
no asynchronous reality model, the landing a joint PD (as in Nguyen et al., arXiv 2408.02619). The robot,
the plans and the controller are ilc_quad's; evaluation stays on ilc_quad's CPU async simulator."""

import os
import sys

ILC_QUAD = os.environ.get("ILC_QUAD_SRC", "/home/henry/ilc_ws/src/ilc_quad")
MENAGERIE = os.environ.get("MUJOCO_MENAGERIE_PATH", "/home/henry/mujoco_menagerie")
WS_HOST, WS_CONTAINER = "/home/henry/ilc_ws", "/ilc_ws"     # bank.json holds container paths
for p in (ILC_QUAD, os.path.join(ILC_QUAD, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)


def host_path(p: str) -> str:
    return WS_HOST + p[len(WS_CONTAINER):] if p.startswith(WS_CONTAINER + "/") else p
