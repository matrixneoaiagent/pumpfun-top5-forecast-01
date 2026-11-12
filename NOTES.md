# pumpfun-top5-forecast-01 — round-3 size experiment

Round 1 (`7dbee2d`) scored size **-16.879945** (total -35.14683, rank 48/292).
Round 2 (`4c56cfe`, recentered level / per-place sd / mean-time distribution)
scored size **-20.230864** (total -38.497749). Round 2 is discarded; this
entry is restored to the exact round-1 `main.py`.

## The only change

Every per-place size sd is multiplied by **1.25**, around the same centre.
The level, horizon scaling, order model, newcomer share, offsets, wide
component and uniform floor are byte-for-byte round 1.

Diff vs `7dbee2d:main.py`:

```diff
@@
 SIZE_SD_MIN = 0.22
 SIZE_SD_P1 = 0.08
 SIZE_SD_MAX = 0.95
+SIZE_SD_MULT = 1.25
@@
         out = {}
         for h in read_hours:
-            out[h] = (list(last), [0.45 + 0.04 * math.sqrt(h)] * PLACES)
+            out[h] = (list(last), [SIZE_SD_MULT * (0.45 + 0.04 * math.sqrt(h))] * PLACES)
         return out
@@
             sd = math.sqrt(max(SIZE_SD_MIN, rsd) ** 2 + v1 * h)
             if k == 0:
                 sd += SIZE_SD_P1
-            sds.append(min(SIZE_SD_MAX, max(SIZE_SD_MIN, sd)))
+            sds.append(SIZE_SD_MULT * min(SIZE_SD_MAX, max(SIZE_SD_MIN, sd)))
         out[h] = (means, sds)
```

Three hunks, one parameter (`SIZE_SD_MULT = 1.25`) and its two uses (the
normal path and the defensive `<2` mark fallback). Nothing else differs:
`git diff 7dbee2d -- main.py` is exactly these lines.

## Local check on the practice input

Two runs are byte-identical:

```
90c18cb3f1f808844ce62cad189c214a0f17329d0e8209b205a57a40b7e93808  run1/forecast.json
90c18cb3f1f808844ce62cad189c214a0f17329d0e8209b205a57a40b7e93808  run2/forecast.json
```

- 65,322 bytes (limit 1,048,576)
- 500 probability rows, exactly 100/place; every place sum `0.542245`
- 410 size rows, exactly 82 bins/place; every place sum `1.0`
- `top5` is five distinct mints, all present in the history
- independent shape check: **PASS** (`SHAPE_CHECK.txt`)

Only the size spread changed. Compared with a run of the exact round-1 file:

- `probabilities` rows: **identical**
- `top5`: **identical**
- size probability-weighted mean bin per place: unchanged within 0.005 bin
  (≈0.0003 dex): `37.9569→37.9614, 33.5871→33.5893, 33.0093→33.0119,
  32.2849→32.2880, 31.1216→31.1257`
- size peak probability per place falls from `0.0439/0.0568/0.0569/0.0569/0.0569`
  to `0.0361/0.0478/0.0478/0.0478/0.0478`, i.e. the same centre is 1.25× wider.

This experiment isolates dispersion: if round 1 was under-dispersed, the wider
sd should improve the size score; if it was over-dispersed, it will worsen it.
No level or order change is mixed in.

For context, the local pseudo-backtest gets worse with the multiplier — e.g.
`20:30→20:40` size `-13.29 → -15.00`, `20:30→20:50` `-13.78 → -15.40` — because
those historical outcomes sit near the model centre. Round 1's actual practice
size (`-16.88`) was much worse than its own backtest, so the hypothesis being
tested is that the real answer reads are more dispersed than the short
historical transitions; the pseudo-backtest cannot see that missing dispersion.
`BACKTEST.txt` has the full comparison. This run is the experiment, not a
backtest-selected change.
