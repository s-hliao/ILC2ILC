"""
GPU rollouts for dilc_train.py, over stdin/stdout -- dilc_execute.py serve's protocol, a batch at a time:

  request   header dict(cmd="batch", jobs=[dict(id, goal, seed), ...], stochastic, fd=dict(eps_a, eps_s)
            or absent), payload: the actor's weights (npz bytes)
  reply     one message per job, as serve's: header dict(ok, id, cond, wall, fd_ok), payload: the episode
            (dilc_execute.EpisodeMaker.make, the FD Jacobians in place of the SRB model's)

The batch flies with one actor (the newest when it was asked for) while the learner goes on: asynchronous
actor and learner, as with the CPU workers. Every jump is nominal (the GPU sim has no reality model).

    python -m ilc_mjx.serve --bank OUT/bank.json [--gpu 0]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys
import time


def _read_msg(f):
    h = f.read(4)
    if len(h) < 4:
        return None, None
    (n,) = struct.unpack("<I", h)
    header = json.loads(f.read(n))
    (m,) = struct.unpack("<Q", f.read(8))
    return header, (f.read(m) if m else b"")


def _write_msg(f, header, payload=None):
    import numpy as np
    hb = json.dumps(header).encode()
    pb = b""
    if payload:
        buf = io.BytesIO()
        np.savez(buf, **payload)
        pb = buf.getvalue()
    f.write(struct.pack("<I", len(hb)) + hb + struct.pack("<Q", len(pb)) + pb)
    f.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--land-time", type=float, default=1.0)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    proto = os.fdopen(os.dup(1), "wb")              # the protocol owns stdout
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    inp = sys.stdin.buffer

    import jax
    import jax.numpy as jnp
    import numpy as np

    from . import host_path
    from .jump import Config, JumpEnv
    from dilc_execute import EpisodeMaker, GoalBank

    bank = json.load(open(host_path(a.bank)))
    bank_h = dict(bank, plans=[dict(p, path=host_path(p["path"])) for p in bank["plans"]])
    env = JumpEnv(bank_h, Config(land_time=a.land_time))
    gb = GoalBank(bank_h)
    maker = EpisodeMaker(gb, os.environ.get("MUJOCO_MENAGERIE_PATH", "/home/henry/mujoco_menagerie"))
    # ILC_MJX_DYN="mass_scale,com_x,motor_scale,curve,speed_scale,friction,joint_friction,payload" (JumpEnv.DYN_KEYS):
    # every jump in this one fixed model instead of the nominal one (the wrong-simulator test; not randomization)
    fixed = os.environ.get("ILC_MJX_DYN")
    P_fixed = np.array([float(x) for x in fixed.split(",")]) if fixed else None
    print(f"ilc_mjx serve: GPU {a.gpu} ({jax.devices()[0]}), bank {a.bank}"
          + (f", fixed model {dict(zip(JumpEnv.DYN_KEYS, P_fixed))}" if fixed else ""), flush=True)
    while True:
        header, payload = _read_msg(inp)
        if header is None or header.get("cmd") == "quit":
            break
        jobs = header["jobs"]
        t0 = time.time()
        try:
            with np.load(io.BytesIO(payload)) as d:
                w = {k: jnp.asarray(d[k], jnp.float32) for k in d.files if k != "n_hidden"}
            goals = np.array([j["goal"] for j in jobs], float)
            fd = header.get("fd")
            if fd:
                env.cfg.eps_a, env.cfg.eps_s = float(fd["eps_a"]), float(fd["eps_s"])
            ref = env.references(goals)
            q_off = a_off = None
            if header.get("explore"):                 # exploration: own stances, smooth action offsets
                q_off, a_off = env.explore_noise(len(jobs), np.random.default_rng(int(jobs[0]["seed"])),
                                                 *map(float, header["explore"]))
            out = env.rollout(w, ref, jax.random.PRNGKey(int(jobs[0]["seed"]) % (2 ** 31)),
                              stochastic=bool(header.get("stochastic", True)), fd=bool(fd),
                              a_offset=a_off, q_offset=q_off,
                              dyn=None if P_fixed is None else env.dyn_arrays(np.tile(P_fixed, (len(jobs), 1))))
            wall = (time.time() - t0) / len(jobs)
            for b, j in enumerate(jobs):
                rf = gb.reference(goals[b])[0]
                jac = (out["A"][b], out["B"][b]) if fd else None
                ep = maker.make(goals[b], rf, out["X"][b], out["U"][b], out["X"][b], bool(out["fell"][b]),
                                jac=jac)
                ep["mu_b"] = out["mu"][b].astype(np.float32)        # the behaviour policy's mean action
                ok = float(np.isfinite(jac[0]).all((1, 2)).mean()) if jac is not None else 0.0
                _write_msg(proto, dict(ok=True, id=j["id"], cond="gpu_nominal" if P_fixed is None else "gpu_fixed", cond_args="", wall=wall,
                                       fd_ok=ok), ep)
        except Exception as err:          # report every job of the batch; the server carries on
            import traceback
            traceback.print_exc()
            for j in jobs:
                _write_msg(proto, dict(ok=False, id=j["id"], error=repr(err)))


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    main()
