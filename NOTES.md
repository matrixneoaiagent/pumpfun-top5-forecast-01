# pumpfun-top5-forecast-01 — Monte Carlo variant

The working deterministic entry is preserved as `main_det.py` (sha256
`79766d57…f4a8e824`). It is the last committed working file (`af3827c`,
round-1 plus the 1.25× size-spread experiment). The exact round-1 scored file
is `7dbee2d`; that is the -35.14683 practice entry (size -16.879945).

`main.py` is now a Monte Carlo variant. It is deterministic: fixed seed
`20261112`, 4,000 paths, standard library only, no network. Runtime on the
practice input is about 12 s.

## Monte Carlo model

For each of the 4,000 paths, the seven answer horizons in `task.json` are
stepped in order:

- **Drift / mean reversion.** Each known coin's log10 cap starts at its last
  mark value. Its target is the round-1 damped-momentum `mu(h)` (same features:
  10/20/40-minute slopes, rank-bucket shrinkage, age decay, crash penalty).
  The path is an OU step
  `c += theta * (mu - c) * dt + sigma * sqrt(dt) * N(0,1)`, with
  `theta = 0.40/h`.
- **Volatility.** `sigma` is the round-1 base sigma times `2.50`, then scaled
  by venue (pool 1.0, curve 1.15, migrating 0.85), age (1.15 under 30 min,
  1.08 under 2 h, 0.90 over 6 h), and rank (1.10 over rank 50, 1.05 over 20).
- **Jumps.** A 3 %/h chance of a `N(0, 0.40)` jump.
- **Death.** A per-hour hazard by venue/rank/age: pool 0.02, curve 0.05,
  migrating 0.03, times 0.5 for the top 5, 0.8 for 6–20, 1.2 for 21–50, 1.5
  beyond; +0.02 for age >3 h and +0.02 for age <15 min. A dead coin collapses
  to 1.45 log10 SOL and stays there.
- **Level alignment.** At each read the common path shift is set so the median
  of the top five `mu` equals the round-1 deterministic place-3 size level.
  This preserves the round-1 centre while the MC supplies the dispersion.
- **Births.** `birth_rate` is read from the history's `counts.eligible`.
  `4 %` of births are pumped contenders. The top 20 contender caps are drawn
  from the order statistics of a normal centred at the deterministic place-3
  level `- 0.80` with sd `0.45`; each is active with probability calibrated so
  the expected newborn top-five share equals the history's unseen share `W(h)`.
  Newborns are ranked with the known coins but are **not named**; they take
  the unnamed probability mass and enter the size bins.

From each path/read the top five caps are taken. Place counts accumulate for
known coins; size-bin counts accumulate for the top five caps (known or
newborn). After all paths/reads:

- Each place's named mass is scaled down to `1 - mean W(h)` when the MC known
  share is larger, so the history-calibrated newcomer share stays unnamed.
- Up to 100 known coins are named per place; `top5` is the five coins with the
  highest summed scaled place probabilities.
- Size bins are the MC counts with a 1 % uniform floor, normalized to 1.

## Parameters

```
MC_PATHS      4000          MC_SEED       20261112
MC_VOL_MULT   2.50          MC_THETA      0.40 /h
MC_P_JUMP     0.03 /h       MC_JUMP_SD    0.40
MC_NEW_K      20            MC_NEW_SHIFT -0.80
MC_NEW_SD     0.45          MC_NEW_NOISE  0.10
MC_P_PUMP     0.04          MC_DEATH_CAP  1.45
```

## Shape and determinism

Two runs are byte-identical:

```
46925a970f985de8109097f52382ef9732e8c94ef75a119c4c5556c47f0e64e2  run1/forecast.json
46925a970f985de8109097f52382ef9732e8c94ef75a119c4c5556c47f0e64e2  run2/forecast.json
```

- 65,472 bytes
- 500 probability rows, exactly 100/place; place sums
  `0.5379, 0.5330, 0.5282, 0.5249, 0.5202`
- 410 size rows, exactly 82 bins/place, every place sum `1.0`
- `top5` five distinct history mints
- independent shape check: **PASS** (`SHAPE_CHECK.txt`)

## Local pseudo-backtests

Full pipeline, one truncated history → next mark, rules order + size score.
`main_det.py` is the preserved working deterministic file; `round1` is the
exact `7dbee2d` scored entry.

Against `main_det.py`:

```
origin->target        MC order  MC size  MC total | DET order DET size DET total | delta
20:30->20:40           -14.55   -12.38    -26.93 |   -14.06   -15.00    -29.06 | +2.13
20:30->20:50           -17.86   -12.21    -30.07 |   -16.88   -15.40    -32.27 | +2.20
20:40->20:50           -14.47   -11.96    -26.43 |   -13.76   -14.49    -28.25 | +1.82
20:20->20:40           -22.83   -18.95    -41.78 |   -21.31   -19.26    -40.57 | -1.21
20:20->20:50           -23.66   -15.34    -38.99 |   -21.88   -19.63    -41.50 | +2.51
```

Against exact round 1:

```
origin->target        MC total | round-1 total | delta
20:30->20:40           -26.93 |   -27.98      | +1.05
20:30->20:50           -30.07 |   -31.25      | +1.18
20:40->20:50           -26.43 |   -27.18      | +0.76
20:20->20:40           -41.78 |   -41.71      | -0.07
20:20->20:50           -38.99 |   -42.69      | +3.70
```

The MC loses 0.4–1.0 order points but gains 1.0–2.0 size points. On the
mature 4–5-mark cases (the practice cut has 5 non-empty marks) the total gain
is **+0.76 to +1.18**; the cold-start cases are mixed. Expected practice total
is about **-34.0 to -34.2** (from -35.15), with the size score moving toward
-15.0 to -15.5.

Against the known last-read practice top five, the MC last-read order estimate
is **-20.7** versus **-19.9** for round 1 (the two nameable coins MIHAO VI and
Google are at p≈0.054/0.051, versus 0.109/0.063). The size gain is expected to
more than offset that order loss on the mean over the seven reads.

A 25 % blend with the deterministic order matrix was also tested; it recovers
0.05–0.2 of the order loss but is not applied because the requested variant
uses the simulated place matrix.

## Status

`main.py` is the MC variant; `main_det.py` is the untouched deterministic
fallback. No commit, no bundle, no submission.
