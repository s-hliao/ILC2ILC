"""fig1_cmd.py SCRIPT SRC_ARGS_JSON OUT GPU OVERRIDES_JSON (2026-11-06, paper/FIGURE_PLAN_1106.md figure 1): run SCRIPT with a source
run's saved arguments (its args.json), --out OUT, --gpu GPU and the overrides (a JSON dict; lists stay lists, null drops a key).
DRY=1 prints the command only."""
import json
import os
import subprocess
import sys

script, src, out, gpu, ov = sys.argv[1:6]
a = json.load(open(src))
a.update(json.loads(ov))
a.update(out=out, gpu=gpu)
cmd = [sys.executable, "-u", script]
for k, v in a.items():
    if v is None or v is False:
        continue
    f = f"--{k.replace('_', '-')}"
    cmd += [f] if v is True else ([f] + [str(x) for x in v] if isinstance(v, list) else [f, str(v)])
print(" ".join(cmd), flush=True)
if not os.environ.get("DRY"):
    os.makedirs(os.path.dirname(out.rstrip("/")) or ".", exist_ok=True)
    sys.exit(subprocess.call(cmd))
