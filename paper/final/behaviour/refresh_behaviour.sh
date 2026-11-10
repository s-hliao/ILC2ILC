#!/bin/bash
# refresh_behaviour.sh: re-collect every machine's PRACB_* runs (collect_beh.py) and re-render car_behaviour_sim.{pdf,png} + _table.txt.
cd "$(dirname "$0")"; PY=~/miniconda3/envs/f1t/bin/python; T=$(mktemp -d)
(cd ~/ilc_ws/src/ilc_f1tenth/ilc2real && $PY $OLDPWD/collect_beh.py) > $T/m.json
ssh henry@172.26.86.229 'cd ~/ilc_ws/src/ilc_f1tenth/ilc2real; ~/miniconda3/envs/f1t/bin/python -' < collect_beh.py > $T/l.json
ssh henry@nsh1609server5.lan.local.cmu.edu 'cd ~/ilc_ws/src/ilc_f1tenth/ilc2real_fix; ~/miniconda3/envs/f1t/bin/python -' < collect_beh.py > $T/u.json
python3 -c "import json,sys; a=[]; [a.extend(json.load(open(f))) for f in sys.argv[1:]]; json.dump(a, open('car_behaviour_runs.json','w')); print(len(a), 'runs')" $T/m.json $T/l.json $T/u.json
python3 fig_behaviour.py car_behaviour_runs.json car_behaviour_sim 2>&1 | grep -v -i findfont | tail -1
