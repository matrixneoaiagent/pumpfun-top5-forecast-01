# pumpfun-top5-forecast-01 entry — notes

`main.py` reads `$MAKEFUN_INPUT_DIR` (default `/input`), writes
`$MAKEFUN_OUTPUT_DIR/forecast.json` (default `/output/`). Python 3.13, standard
library only, no network, deterministic. Typical runtime on a 66-mark history
is under 1 s.

## What it does

1. **Parse the marks.** Only `ok` marks with rows are used. The last mark is the
   cut; every read in `task.json` is converted to hours after that cut. Nothing
   is hard-coded to the practice or final dates.
2. **Per-coin cap path.** For every coin in the last mark: current log10 cap in
   SOL, age, venue (`curve` / `migrating` / `pool`), complete / mayhem flags,
   quote reserves, rank, and slopes over the last 2, 3 and 5 observations
   (~10/20/40 min). A damped momentum forecast is made for each horizon h:

       g   = 0.50*g2 + 0.30*g4 + 0.20*g1          (per hour, clipped by venue)
       mu  = c0 + g * mult * tau * (1 - exp(-h/tau))

   with `(tau, mult, gmax)`: pool `(1.6 h, 0.60, 1.2)`, curve
   `(1.1 h, 0.50, 2.5)`, migrating `(0.7 h, 0.25, 1.0)`. Growth is capped at
   +2.0/+2.5/+1.0 dex and decline at -2.5/-2.5/-2.0 dex. The level is shrunk
   20 % toward the median of the coin's rank bucket (1–3, 4–8, …, 101–200) as
   the horizon grows. A crash/reliability penalty subtracts up to 0.6 dex for
   coins with fewer than four reads, age under 15 min, a vertical last hour
   (`g2 > 0.9`/h), or pool SOL depth below 1 % of cap. `sigma = 0.30 + 0.10*sqrt(h)` (+0.08 curve, +0.05 thin
   history), capped at 0.92; it is used for documentation/uncertainty, not for
   the place matrix.

3. **Unseen share W(h).** For every historical origin/target pair, the code
   asks what fraction of the target's top five was outside the origin's top
   `chart_size` (or not born yet). Pairs are binned by horizon (15 min / 30 min
   / 1 h), the first two or three marks are excluded as a cold start, and a saturating
   curve `W(h) = Winf * (1 - exp(-h/tau))` is fitted to the bin means with a
   prior toward 0.50. The result is blended 75/25 with that prior and clamped to
   [0.15, 0.85]. This is the share of each place left unnamed for coins the
   history cannot name, and it adapts to the actual churn in the history (a
   stable top five yields W ≈ 0.17; a completely fresh top five yields ≈ 0.82).

4. **Place probabilities.** The named set is the 100 last-mark mints with the
   best average predicted cap (the 100-row limit is per place). For each read:
   a sharp softmax over `mu` (`tau_w = 0.12 + 0.02h`) gives the strength of each
   coin; a logistic soft rank over the leading 20 gives a rank centre; a
   Gaussian place kernel (`tau_place = 1.50 + 0.06h`) spreads each coin over
   places 1–5. The place column is normalized to `1 - W(h)`. The seven reads are
   averaged, so the final unnamed mass is `1 - mean W(h)`. This is deliberately
   broad: over 7–13 h the exact place is much less certain than the set of
   coins that can hold it.

5. **Sizes.** The five largest caps of each mark form a top-5 series. The level
   is the median of that series; place offsets are the median gaps to the level,
   shrunk toward the mature pattern `[+0.25, +0.03, 0, -0.03, -0.06]`. The
   level trend is the Theil–Sen slope over the last 6 h: the first hour follows
   it up to 0.80 dex/h, later hours at up to 0.12 dex/h, with total growth
   capped at ±1.10 dex. **One distribution per place is written, at the mean
   answer time** (the mean of the seven read hours), not an average of seven
   per-read distributions. The log-cap sd is per place:

       sd_k(h) = sqrt(SD_BASE[k]^2 + (SD_VK[k] + 0.30*v1)*h)

   with `SD_BASE = [0.18, 0.20, 0.24, 0.28, 0.32]`,
   `SD_VK = [0.008, 0.015, 0.020, 0.025, 0.030]`, and `v1` the robust
   variance-per-hour from historical top-5 differences (0.018–0.050). Place 1
   is the most predictable and place 5 the least. A 10 % wide component
   (1.8× sd) and a 1.5 % uniform floor keep the tails above the scorer's
   1e-4 floor.

6. **top5.** The five named coins with the highest expected place value
   `sum_k p(place k) * (6-k)`. All are last-mark mints, hence in the history.

## Size diagnosis and variants (practice feedback)

The submitted practice score was **-35.14683 = order -18.266885 + size
-16.879945** (rank 46/292). Order is competitive; size is the gap. The
practice history has only six marks (five non-empty, 50 min), so the 1-hour
answer reads are an extrapolation. The transitions that are available:

- 10-min mean Δlog10 cap, mature origins 20:30/20:40: `[0.15, 0.07, 0.07,
  0.09, 0.09]`; sd `[0.06, 0.10, 0.02, 0.03, 0.07]`.
- 10-min mean, early surge origins 20:10/20:20: `[0.13, 0.30, 0.33, 0.33,
  0.36]`; sd `[0.05, 0.30, 0.27, 0.27, 0.35]`.
- 20-min mean, mature: `[0.30, 0.13, 0.13, 0.18, 0.19]`; early:
  `[0.29, 0.51, 0.64, 0.71, 0.80]`.

Two things stood out. First, the per-place dispersion is stable and the
submitted model had it backwards: place 1 is the most predictable
(sd ≈ 0.05–0.15) and places 4–5 the least (≈ 0.3–0.4), while the submitted
model gave place 1 the widest sd (0.43 at h = 1.17). Second, the level was
low. Combining the two reported aggregate scores (ours -16.879945 and the
baseline's -19.027956) with a smooth last-mark-to-read-7 ramp puts the
practice answer level at about 4.1–4.2 at read 7; the submitted mean was
about 3.82, i.e. **0.3–0.4 dex low**.

Three variants were scored with the rules' size log score:

- **A — submitted:** seven per-read distributions averaged; short growth cap
  0.50, total cap 0.80; common v1 sd with +0.08 on place 1; 15 % wide.
- **B — recentered:** one distribution at the mean answer time; short growth
  cap 0.80, total cap 1.10; per-place sd above; 10 % wide.
- **C — surge mixture:** B plus a newcomer-surge component
  (`0.50*W` weight, `+0.40` dex, 1.25× sd).

```
variant             hist_mean  short<=20m  long40m  1h_diag_mean  1h_diag_min
A-submitted           -20.51      -18.36   -28.65       -16.40       -18.24
B-recentered          -19.37      -17.23   -29.30       -15.61       -16.69
C-surge-mixture       -19.37      -17.23   -29.30       -16.16       -17.16
```

`hist_mean` is the mean size score over every origin/target pair in the
practice; `short<=20m` is the 10/20-minute subset (mature and early-surge origins); `long40m` is the
single 40-minute cold-start pair; `1h_diag` scores the practice-cut
distribution on 15 smooth 1-hour ramps whose read-7 level is 3.9–4.3 (the
range consistent with the baseline score).

**B is the choice.** It beats A on the 10/20-minute transitions (+1.1) and
on every 1-hour diagnostic with read-7 level ≥ 4.0; only the single 40-minute
cold-start pair is slightly worse (B -29.30 vs A -28.65), where the trend is
undefined. C's extra surge tail is too wide (−0.55 versus B on the 1h
diagnostic), so it is not used.

Expected gain on the practice: A's 1h diagnostic mean -16.40 → B -15.61
(**+0.79**); at the baseline-implied level ≈4.1, A -16.27 → B -15.35
(**+0.92**); at 4.2, +1.22. So the expected total is about **-34.0 to
-34.2**, up from -35.15.

## Validation done

- Two runs on the practice input are byte-identical
  (`sha256 05a91c01…1aac08`), 65,418 bytes, 500 probability rows (100/place),
  410 size rows (82/place), every place sum ≤ 1.
- Pseudo-backtests on the practice (truncate history at a mark, forecast the
  next mark with the full pipeline, score with the rules' order + size formulas):
  `20:30→20:40` order −14.06 / size −13.29; `20:30→20:50` −16.88 / −13.78;
  `20:40→20:50` −13.76 / −13.19. The cold-start sizes also improved sharply
  (`20:20→20:30` −18.16, `20:20→20:40` −16.69, `20:20→20:50` −16.21).
- Against the known last-read practice top five (three of which were born after
  the history ended), the last-read order estimate is about −19.9, with the two
  nameable coins (MIHAO VI #2, Google #3) at the right places and the three
  newcomers paid from the unnamed share.
- Synthetic 66-mark stable and steadily-growing histories: the size level
  follows the trend, W stays low for a stable top five and saturates near 0.82
  for a completely fresh one, and the output shape stays valid.

## Main risks / choices

- The size forecast is the largest lever. The recentered variant bets on a
  stronger first-hour trend and a narrower, per-place distribution. If the
  final's top-5 level plateaus instead of continuing, the mean is high; the
  wide component and the trend cap are the hedge.
- The 1h diagnostic is calibrated from the two reported aggregate scores, not
  from the hidden practice caps; it brackets the level (3.9–4.3) rather than
  pinning one path. The historical 10/20-minute backtest independently favors B
  over A, which is the stronger evidence.
- The single 40-minute cold-start pair is slightly worse under B; it has only
  one mark of history and an undefined trend, so it is not representative of
  the final's 66-mark input.
- The unnamed share is data-driven; if the final's top five is more stable than
  the history's transitions suggest, the forecast overpays newcomers.
- No randomness, no model calls, no network; all ordering is by `(−score, mint)`
  so two runs match byte for byte.
