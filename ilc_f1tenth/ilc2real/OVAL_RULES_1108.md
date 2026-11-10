# Plastic second track = room oval `oval165` (2026-11-08 ~08:15, written from matcha BEFORE any oval result of tonight)

Why (measured, 1108_plastic 00013/00014/00025/00026): on fig8R115 a36 the real car needs curvature 0.87 in the lobes, about 93 % of its
kinematic maximum (~0.94 1/m, servo ~0.30 rad); it sits at full lock 41-100 % of the lobe sections and cannot flip curvature at the
crossover (achieved 0.45-0.75 vs 0.87, e_y 0.22-0.31 m right after it). The stage then had no steering authority to use (actuation
precondition, paper): pace stayed 0.79-0.81, nerr +1..+3 %. The oval's turns are R 1.65 m (curvature 0.61, ~65 % of the maximum) and its
straights give the speed term room; the 11-07 sim check on oval a26 (old mask) improved 4/4 runs by 28-33 % through pace.

Method unchanged (the same stage_track / prac_one SB command, 3 trials x 2 laps on hardware, 4 iterations in sim, --skip-below 0.5).
Anchors = the practised oval plan + the oval nominal plan (the rule for every new cell).

## Rules
1. Sim plan (zero-shot sweep, 12 plastic cars, both networks s0, launched scorer, 5 laps; plans a26 a28 a30 a34 a40): the fastest plan
   with no zero-shot crash and mean rms e_y <= 0.17 m for both networks (fig8's plastic rule).
2. Hardware plan: fly the zero-shots on a30, then a34 (both networks, 5 laps). Stage on the FASTER of the two that has (i) no wall / crash,
   (ii) full lock under half of the corner time, (iii) zero-shot nerr >= 0.5. If neither: a26.
3. Sim cell = PRACB_plastic_oval<A> on the rule-1 plan; if the hardware plan differs, also a cell on the hardware plan (as rubber a30).
4. Report: % change it0->it3 + absolute change, rear-out (signed sideslip < -3 deg in corners) before/after, every run.
