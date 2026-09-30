"""mk.py NAME VARIANTS...: jobs file sweep/NAME.jobs, the landing cells x variants 'name=args'
on the default candidate (ilc_gain 2, ilc_mdc go1)."""
import sys
D = "--param ilc_gain:=2.0 --param ilc_mdc:=go1"
REAL = ("--act-delay 0.004 --pose-rate 240 --pose-delay 0.006 --pose-noise 0.0005 0.003 "
        "--param pose_latency:=0.006 --joint-noise 0.002 0.3 --mass-scale 1.05 --motor-curve "
        "--joint-friction 0.2 --foot-sensor 40 0.3 5 --seed 1")
cells = [("S1", "b50_20_m90", ""), ("S2", "b50_20_m90", "--mass-scale 1.15"),
         ("S4", "b50_20_m90", "--motor-curve --motor-speed-scale 0.85"), ("S5", "b50_20_m90", REAL),
         ("S6", "b50_20_m90", "--com-offset 0.02 0"), ("S7", "b50_15_m90", "--mass-scale 1.15")]
name, V = sys.argv[1], [v.split("=", 1) for v in sys.argv[2:]]
with open(f"sweep/{name}.jobs", "w") as f:
    for v, a in V:
        for s, t, c in cells:
            f.write(f"{s}__{v}|{t}|scratch|{c} {D} {a}\n")
