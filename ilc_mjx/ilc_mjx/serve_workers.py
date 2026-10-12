"""The executor's serve workers (dilc_execute.py serve in the ilc_quad container): rollouts on the CPU robots -- the very
robots (condition strings, controller, estimator, overlay) every evaluation flies. Shared by quad_real_ppo.py (on-robot
RL) and fada_collect_cpu.py (FADA's data on the robot), so an oracle trained through them sees EXACTLY the transferred
dynamics; check_cond() asserts it per rollout from the condition string the worker echoes (header cond_args)."""
import io
import json
import os
import queue
import struct
import subprocess
import threading

import numpy as np

WS, CWS = "/home/henry/ilc_ws", "/ilc_ws"


def cpath(p):
    """A host path under the workspace -> the container's (symlinks resolved first: their absolute host targets do not exist in
    the container)."""
    p = os.path.realpath(p)
    assert p.startswith(WS), f"{p} is outside the workspace the container mounts"
    return CWS + p[len(WS):]


def xc(overlay):
    """The container command prefix that runs the executor with the box-fix overlay."""
    return (f"cd /ilc_ws && source install/setup.bash && export ROS_LOCALHOST_ONLY=1 OMP_NUM_THREADS=1 && "
            f"export PYTHONPATH={overlay}:$PYTHONPATH && python3 install/ilc_quad/lib/ilc_quad/dilc_execute.py")


def _msg(header, payload=None):
    hb = json.dumps(header).encode()
    pb = b""
    if payload is not None:
        buf = io.BytesIO()
        np.savez(buf, **payload)
        pb = buf.getvalue()
    return struct.pack("<I", len(hb)) + hb + struct.pack("<Q", len(pb)) + pb


def _read(f):
    h = f.read(4)
    if len(h) < 4:
        return None, None
    (n,) = struct.unpack("<I", h)
    header = json.loads(f.read(n))
    (m,) = struct.unpack("<Q", f.read(8))
    pb = f.read(m) if m else b""
    if not pb:
        return header, None
    with np.load(io.BytesIO(pb), allow_pickle=False) as d:
        return header, {k: d[k] for k in d.files}


def check_cond(header, expected, prefix=False):
    """Raises unless the rollout ran the expected condition string (prefix: an appended option, e.g. a stance offset)."""
    got = header.get("cond_args")
    ok = got is not None and (got.startswith(expected) if prefix else got == expected)
    if not ok:
        raise RuntimeError(f"rollout {header.get('id')} ran condition {got!r}, expected {expected!r}"
                           f"{' (prefix)' if prefix else ''}: not the transferred dynamics")


class Workers:
    def __init__(self, n, bank_path, overlay):
        self.cmd = f"{xc(overlay)} serve --bank {cpath(bank_path)}"
        self.procs = [self._spawn() for _ in range(n)]

    def _spawn(self):
        return subprocess.Popen(["sg", "docker", "-c", f"docker exec -i ilc_quad bash -lc '{self.cmd}'"],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def run(self, jobs, retries=3):
        """jobs: [(header, payload)] -> [(header, episode)] in order, the workers in parallel. A worker that dies (broken pipe,
        closed stream) is respawned and its job re-queued, up to `retries` times per job; a job that keeps failing returns
        (None, None)."""
        q, out = queue.Queue(), [(None, None)] * len(jobs)
        tries = [0] * len(jobs)
        for i, j in enumerate(jobs):
            q.put((i, j))

        def loop(slot):
            while True:
                try:
                    i, (h, pl) = q.get_nowait()
                except queue.Empty:
                    return
                p = self.procs[slot]
                try:
                    p.stdin.write(_msg(h, pl))
                    p.stdin.flush()
                    res = _read(p.stdout)
                    if res[0] is None:
                        raise EOFError("worker closed its stream")
                    out[i] = res
                except (BrokenPipeError, EOFError, OSError, ValueError):
                    try:
                        p.kill()
                    except Exception:
                        pass
                    self.procs[slot] = self._spawn()
                    tries[i] += 1
                    if tries[i] <= retries:
                        q.put((i, (h, pl)))
        th = [threading.Thread(target=loop, args=(k,)) for k in range(len(self.procs))]
        for t in th:
            t.start()
        for t in th:
            t.join()
        return out

    def close(self):
        for p in self.procs:
            try:
                p.stdin.write(_msg(dict(cmd="quit")))
                p.stdin.close()
            except Exception:
                pass
        for p in self.procs:
            p.wait(timeout=60)
