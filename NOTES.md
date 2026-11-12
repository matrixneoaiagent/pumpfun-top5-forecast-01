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
   it up to 0.50 dex/h, later hours at up to 0.12 dex/h, with total growth
   capped at ±0.80 dex. The log-cap sd is `sqrt(max(0.22, robust residual)^2 +
   v1*h)`, where `v1` is a robust variance-per-hour from historical top-5
   differences, clamped to [0.018, 0.050]; place 1 gets +0.08. A 15 % wide
   component (1.8× sd) and a 1.5 % uniform floor keep the tails above the
   scorer's 1e-4 floor. The seven reads are averaged per bin and normalized.

6. **top5.** The five named coins with the highest expected place value
   `sum_k p(place k) * (6-k)`. All are last-mark mints, hence in the history.

## Validation done

- Two runs on the practice input are byte-identical
  (`sha256 96374de5…636256d8`), 65,378 bytes, 500 probability rows (100/place),
  410 size rows (82/place), every place sum ≤ 1.
- Pseudo-backtests on the practice (truncate history at a mark, forecast the
  next mark with the full pipeline, score with the rules' order + size formulas):
  `20:30→20:40` order −14.06 / size −13.92; `20:30→20:50` −16.88 / −14.37;
  `20:40→20:50` −13.76 / −13.42. The cold-start cases (`20:20` origins) are
  worse, as expected with only one or two marks of context.
- Against the known last-read practice top five (three of which were born after
  the history ended), the last-read order estimate is about −19.9, with the two
  nameable coins (MIHAO VI #2, Google #3) at the right places and the three
  newcomers paid from the unnamed share.
- Synthetic 66-mark stable and steadily-growing histories: the size level
  follows the trend, W stays low for a stable top five and saturates near 0.82
  for a completely fresh one, and the output shape stays valid.

## Main risks / choices

- The size forecast is the largest lever. It bets on some continuation of the
  recent top-5 level trend, but caps the extrapolation and keeps a wide tail.
- The unnamed share is data-driven; if the final's top five is more stable than
  the history's transitions suggest, the forecast overpays newcomers.
- No randomness, no model calls, no network; all ordering is by `(−score, mint)`
  so two runs match byte for byte.
