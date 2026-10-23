### Quad (Go1 jump to flat/box goals, 4 robots incl. r5: friction 0.59, mass x1.14) - start drB_s2, 36 paired (seed, robot) cells

| arm | real jumps / robot | success % | score (raw) | score (capped) | d success vs control (pts) | d capped vs control |
|---|---|---|---|---|---|---|
| zero-shot (start network) | 0 | 43.0 | 21.30 | 12.06 | - | - |
| DistILC stage (recipe) | 24 | 43.2 | 20.19 | 10.56 | +1.6 [-0.9, +3.9] (n=36) | -1.05 [-1.53, -0.61] |
| stage, pooled-data variant | 24 | 45.3 | 19.30 | 10.80 | +4.9 [+2.3, +7.3] (n=12) | -0.79 [-1.60, -0.08] |
| random-direction control | 24 | 41.6 | 20.04 | 11.61 | (control) | (control) |

**Replication: start network drCft_s2 (24 paired cells)**

| arm | real jumps / robot | success % | score (raw) | score (capped) | d success vs control (pts) | d capped vs control |
|---|---|---|---|---|---|---|
| zero-shot (start network) | 0 | 39.8 | 22.00 | 12.26 | - | - |
| DistILC stage (recipe) | 24 | 44.5 | 20.82 | 11.89 | +7.2 [+2.5, +12.0] (n=24) | -0.97 [-1.44, -0.49] |
| random-direction control | 24 | 37.4 | 23.10 | 12.86 | (control) | (control) |

Validation goals (planeval); success = pre-registered (no fall, |ex| <= 5 cm, |ez| <= 3 cm); score = 0.01 landing cost + 20 fall, lower = better; capped = each jump's cost term capped at 20 (POST HOC: the raw mean is dominated by a few tumbles). r5 alone, recipe - control success: drB +2.8 [+0.0, +5.6] pts (9 cells), drCft -3.6 [-6.8, +0.0] pts (6 cells). Paired 95 % bootstrap CIs.
