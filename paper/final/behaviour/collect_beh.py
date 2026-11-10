import json, glob, re, sys
out = []
for f in sorted(glob.glob('runs/screen/PRACB_*/ladder_launch.json')):
    m = re.match(r'runs/screen/PRACB_(\w+?)_(\w+?)_(freePPON|rmaN|oursdrWSL|oursdr|ours)_s(\d+)_ss(\d+)/', f)
    if not m: continue
    d = json.load(open(f)); p = list(d)[0]
    if '0' not in d[p] or '3' not in d[p]: continue
    a, b = d[p]['0'], d[p]['3']
    cars = [c for c in a if c in b]
    out.append(dict(prot=m[1], track=m[2], net=m[3], seed=int(m[4]), ss=int(m[5]), plan=p,
                    cars={c: dict(s0=a[c]['score'], s3=b[c]['score'], r0=a[c].get('rear3'), r3=b[c].get('rear3'),
                                  b0=a[c].get('sb_p50'), b3=b[c].get('sb_p50'), ey0=a[c]['rms_ey'], ey3=b[c]['rms_ey'],
                                  pace0=a[c]['pace'], pace3=b[c]['pace']) for c in cars}))
json.dump(out, sys.stdout)
