#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry.

Reads the house's 10-minute marks of pump.fun coins, builds a damped-momentum
model of each known coin's log10 market cap, estimates the share of each place
that will be taken by coins the history cannot name (newcomers), and writes one
forecast scored at all seven answer reads.

Standard library only, deterministic, no network.  The output is
$MAKEFUN_OUTPUT_DIR/forecast.json (or /output/forecast.json).
"""

import json
import math
import os
from datetime import datetime, timezone

# --------------------------------------------------------------------------- #
# Tunables.  Kept in one block so NOTES.md can cite them without hunting.
# --------------------------------------------------------------------------- #
PLACES = 5
MAX_ROWS_PER_PLACE = 100
MAX_BYTES = 1_048_576
P_FLOOR = 1e-4

# per-coin cap-path model
TAU = {"pool": 1.6, "curve": 1.1, "migrating": 0.7}
MULT = {"pool": 0.60, "curve": 0.50, "migrating": 0.25}
GMAX = {"pool": 1.2, "curve": 2.5, "migrating": 1.0}
GMIN = {"pool": -0.8, "curve": -1.0, "migrating": -0.5}
MAX_GAIN = {"pool": 2.0, "curve": 2.5, "migrating": 1.0}
MAX_LOSS = {"pool": 2.5, "curve": 2.5, "migrating": 2.0}
SIG_BASE = 0.30
SIG_H = 0.10
SIG_CURVE = 0.08
SIG_MAX = 0.92
AGE_DECAY_START = 3.0
AGE_DECAY = 0.05
CURVE_AGE_DECAY = 0.07
NOBS_PENALTY = 0.06
YOUNG_PENALTY = 0.08
FLASH_PENALTY = 0.18
SHRINK = 0.20
SHRINK_H = 4.0
CAP_HI = 6.5
CAP_LO = 1.0

# place-level size model
LEVEL_HIST_H = 6.0
LEVEL_SHRINK = 0.15
LEVEL_SLOPE_SHORT = 0.50
LEVEL_SLOPE_LONG = 0.12
LEVEL_DAMP = 1.00
LEVEL_H_CAP = 13.0
LEVEL_GROWTH_CAP = 0.80
OFFSET_PRIOR = [0.25, 0.03, 0.0, -0.03, -0.06]
OFFSET_PRIOR_W = 0.55
OFFSET_PRIOR_N = 5.0
SIZE_SD_MIN = 0.22
SIZE_SD_P1 = 0.08
SIZE_SD_MAX = 0.95
V1_DEFAULT = 0.035
V1_MIN = 0.018
V1_MAX = 0.050
V1_SCALE = 0.60
SIZE_MIX_WIDE = 0.15
SIZE_WIDE_FACTOR = 1.8
UNIFORM_MIX = 0.015

# order model
TAU_W_BASE = 0.12
TAU_W_H = 0.020
TAU_R_BASE = 0.30
TAU_R_H = 0.020
TAU_PLACE_BASE = 1.50
TAU_PLACE_H = 0.060
RANK_FIELD = 20          # soft ranks are measured inside this leading field
W_PRIOR = 0.50
W_MIN = 0.15
W_MAX = 0.85


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def parse_time(value):
    """ISO 8601 UTC string -> seconds since the epoch (float)."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def logcap(row):
    """log10 of a row's market cap in SOL, exactly as the scorer takes it."""
    return math.log10(max(1, int(row["market_cap_lamports"]))) - 9.0


def quantile(values, q):
    """Linear-interpolated quantile; values need not be sorted."""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def median(values):
    return quantile(values, 0.5)


def mad_sd(values):
    """A robust standard deviation from the median absolute deviation."""
    if not values:
        return 0.0
    m = median(values)
    return 1.4826 * median([abs(v - m) for v in values])


def weighted_slope(points, weights):
    """Least-squares slope of y against x, with weights."""
    n = len(points)
    sw = sum(weights)
    if n < 2 or sw <= 0.0:
        return 0.0
    mx = sum(w * p[0] for w, p in zip(weights, points)) / sw
    my = sum(w * p[1] for w, p in zip(weights, points)) / sw
    den = sum(w * (p[0] - mx) ** 2 for w, p in zip(weights, points))
    if den <= 1e-15:
        return 0.0
    num = sum(w * (p[0] - mx) * (p[1] - my) for w, p in zip(weights, points))
    return num / den


def theil_sen(points):
    """Median of all pairwise slopes: a robust trend."""
    slopes = []
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            if abs(dx) > 1e-9:
                slopes.append((yj - yi) / dx)
    return median(slopes) if slopes else 0.0


def normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def bin_probabilities(mean, sd, edges):
    """Probabilities of the 82 log10-cap bins for a normal(mean, sd)."""
    if not (sd > 0.0) or not math.isfinite(sd):
        sd = 0.30
    out = []
    prev = 0.0
    # bin 0 is below edges[0]
    cur = normal_cdf((edges[0] - mean) / sd)
    out.append(max(0.0, cur - prev))
    prev = cur
    for i in range(1, len(edges)):
        cur = normal_cdf((edges[i] - mean) / sd)
        out.append(max(0.0, cur - prev))
        prev = cur
    out.append(max(0.0, 1.0 - prev))
    return out


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
def read_inputs(input_dir):
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as fh:
        task = json.load(fh)
    marks = []
    path = os.path.join(input_dir, "history.ndjson")
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                mark = json.loads(line)
            except json.JSONDecodeError:
                continue
            if mark.get("ok") is True and isinstance(mark.get("rows"), list) and mark["rows"]:
                marks.append(mark)
    marks.sort(key=lambda m: parse_time(m["at"]))
    return task, marks


# --------------------------------------------------------------------------- #
# Coin state and trajectory forecast
# --------------------------------------------------------------------------- #
def build_coin_states(marks, cut_time, chart_size):
    """State of every coin in the last mark, plus its recent trajectory."""
    last = marks[-1]
    observations = {}
    for mark in marks:
        t = parse_time(mark["at"])
        for row in mark["rows"]:
            observations.setdefault(row["mint"], []).append((t, row))

    states = {}
    for row in last["rows"]:
        mint = row["mint"]
        obs = sorted(observations.get(mint, []), key=lambda x: x[0])
        recent = [x for x in obs if cut_time - x[0] <= 2.0 * 3600.0][-6:]
        if not recent:
            recent = obs[-2:]
        points = [((t - cut_time) / 3600.0, logcap(r)) for t, r in recent]
        g1 = g2 = g4 = 0.0
        if len(points) >= 2:
            g1 = weighted_slope(points[-2:], [1.0, 1.0])
        if len(points) >= 3:
            g2 = weighted_slope(points[-3:], [1.0, 2.0, 3.0])
        if len(points) >= 4:
            g4 = weighted_slope(points[-4:], [1.0, 2.0, 3.0, 4.0])
        elif len(points) >= 2:
            g4 = g2
        created = row.get("created_at") or last.get("at")
        age = max(0.0, (cut_time - parse_time(created)) / 3600.0)
        quote = int(row.get("quote_reserves_lamports") or 0)
        states[mint] = {
            "mint": mint,
            "c0": logcap(row),
            "age": age,
            "venue": row.get("venue", "curve"),
            "complete": bool(row.get("complete")),
            "mayhem": bool(row.get("mayhem")),
            "rank": int(row.get("rank") or 999),
            "g1": g1,
            "g2": g2,
            "g4": g4,
            "nobs": len(points),
            "peak": max((p[1] for p in points), default=logcap(row)),
            "quote": math.log10(max(1, quote)) - 9.0,
            "symbol": row.get("symbol") or "",
            "on_chart": int(row.get("rank") or 999) <= chart_size,
        }
    return states


def predict_coin(state, h, bucket_prior):
    """Predicted log10 cap and its sd at horizon h hours."""
    venue = state["venue"] if state["venue"] in TAU else "curve"
    tau = TAU[venue]
    mult = MULT[venue]
    g = 0.50 * state["g2"] + 0.30 * state["g4"] + 0.20 * state["g1"]
    if state["nobs"] < 3:
        g = 0.60 * state["g1"] + 0.40 * g
    g = max(GMIN[venue], min(GMAX[venue], g))
    growth = g * mult * tau * (1.0 - math.exp(-h / tau))
    mu = state["c0"] + growth

    if venue == "pool" and state["age"] > AGE_DECAY_START:
        mu -= AGE_DECAY * (state["age"] - AGE_DECAY_START) * min(1.0, h / 4.0)
    if venue == "curve" and state["age"] > 1.0:
        mu -= CURVE_AGE_DECAY * (state["age"] - 1.0) * min(1.0, h / 3.0)

    prior = bucket_prior.get(state["mint"])
    if prior is not None:
        w = SHRINK * min(1.0, h / SHRINK_H)
        mu = (1.0 - w) * mu + w * (prior + growth)

    # Crash/reliability penalty.  A coin with only one or two reads, a very
    # young age, or a vertical last hour is the classic flash pump: its cap is
    # high now but rarely survives a multi-hour horizon.  The penalty is small
    # at short h and grows over the first couple of hours.
    penalty = 0.0
    if state["nobs"] < 4:
        penalty += NOBS_PENALTY * (4 - state["nobs"]) * min(1.0, h / 2.0)
    if state["age"] < 0.25:
        penalty += YOUNG_PENALTY * min(1.0, h / 2.0)
    flash = max(0.0, state["g2"] - 0.9)
    penalty += FLASH_PENALTY * flash * min(1.0, h / 2.0)
    if venue == "pool":
        # A pool whose SOL depth is tiny relative to its cap is fragile: a
        # single sell can move it a long way.  Mild, and zero for the normal
        # 5–10 % depth seen in the rehearsal.
        thin = max(0.0, -2.0 - (state["quote"] - state["c0"]))
        penalty += 0.08 * thin * min(1.0, h / 3.0)
    mu -= min(0.60, penalty)

    mu = min(mu, state["c0"] + MAX_GAIN[venue], CAP_HI)
    mu = max(mu, state["c0"] - MAX_LOSS[venue], CAP_LO)

    sigma = SIG_BASE + SIG_H * math.sqrt(max(0.0, h))
    if venue != "pool":
        sigma += SIG_CURVE
    if state["nobs"] < 3:
        sigma += 0.05
    if state["c0"] - state["peak"] < -0.50:
        sigma += 0.04
    if state["age"] < 0.5:
        sigma += 0.03
    return mu, min(SIG_MAX, max(0.15, sigma))


def bucket_priors(states):
    """Median current log cap of each rank bucket, for shrinkage."""
    buckets = [(1, 3), (4, 8), (9, 20), (21, 50), (51, 100), (101, 200), (201, 10 ** 9)]
    out = {}
    for lo, hi in buckets:
        xs = [s["c0"] for s in states.values() if lo <= s["rank"] <= hi]
        if xs:
            m = median(xs)
            for s in states.values():
                if lo <= s["rank"] <= hi:
                    out[s["mint"]] = m
    return out


# --------------------------------------------------------------------------- #
# Unseen-coin share, from the history's own top-5 source ranks
# --------------------------------------------------------------------------- #
def top_mints(mark, chart_size):
    rows = sorted(mark["rows"], key=lambda r: int(r.get("rank") or 10 ** 9))
    return [r["mint"] for r in rows[:PLACES]]


def estimate_unseen_share(marks, read_hours, chart_size):
    """P(a top-5 place at h is held by a coin outside the top chart_size now).

    Measured at every historical origin/target pair, then a saturating curve
    W(h) = Winf * (1 - exp(-h / tau)) is fitted.  The first three marks are
    excluded: the rehearsal's first half hour is a cold start whose churn
    overstates a mature market's.
    """
    n = len(marks)
    if n < 5:
        return {h: W_PRIOR for h in read_hours}

    start = 3 if n >= 8 else 2
    samples = []  # (delta_hours, unseen_fraction, weight)
    rank_at = []
    for mark in marks:
        rank_at.append({r["mint"]: int(r.get("rank") or 10 ** 9) for r in mark["rows"]})
    tops = [top_mints(m, chart_size) for m in marks]

    for o in range(start, n):
        for t in range(o + 1, n):
            dh = (parse_time(marks[t]["at"]) - parse_time(marks[o]["at"])) / 3600.0
            if dh <= 0.02:
                continue
            unseen = sum(1 for mint in tops[t] if rank_at[o].get(mint, 10 ** 9) > chart_size)
            samples.append((dh, unseen / float(PLACES)))
    if not samples:
        return {h: W_PRIOR for h in read_hours}

    # Bin the pairs by horizon and fit the bin means, not the raw pairs: the
    # 10-minute pairs are far more numerous than the 8-hour ones, and fitting
    # raw pairs would let the short end drown the long end.
    bins = {}
    for dh, y in samples:
        if dh < 1.0:
            key = round(dh * 4.0) / 4.0
        elif dh < 3.0:
            key = round(dh * 2.0) / 2.0
        else:
            key = float(round(dh))
        if key <= 0.0:
            continue
        bins.setdefault(key, []).append(y)
    points = [(key, sum(vals) / len(vals), len(vals)) for key, vals in sorted(bins.items())]
    if not points:
        return {h: W_PRIOR for h in read_hours}

    # A saturating curve through the binned means.  The fit is coarse and
    # heavily smoothed: Winf in [0.20, 0.92], tau in [0.5h, 20h].
    best = None
    for i in range(37):
        winf = 0.20 + 0.02 * i
        for j in range(25):
            tau = 0.5 * (1.18 ** j)
            if tau > 20.0:
                break
            sse = 0.0
            for dh, y, _count in points:
                pred = winf * (1.0 - math.exp(-dh / tau))
                sse += (pred - y) ** 2
            sse /= len(points)
            # keep the curve near the prior when the sample is thin
            sse += 0.02 * (winf - W_PRIOR) ** 2
            if best is None or sse < best[0]:
                best = (sse, winf, tau)
    _, winf, tau = best
    out = {}
    for h in read_hours:
        w = winf * (1.0 - math.exp(-h / tau))
        w = 0.75 * w + 0.25 * W_PRIOR
        out[h] = max(W_MIN, min(W_MAX, w))
    return out


# --------------------------------------------------------------------------- #
# Size model: the top-5 order statistics of log10 cap
# --------------------------------------------------------------------------- #
def top5_series(marks):
    ys = []
    for mark in marks:
        caps = sorted((logcap(r) for r in mark["rows"]), reverse=True)[:PLACES]
        if len(caps) == PLACES:
            ys.append((parse_time(mark["at"]), caps))
    return ys


def estimate_v1(ys, cfg_v1=V1_DEFAULT):
    """Variance per hour of a place's log cap, from historical differences."""
    if len(ys) < 3:
        return cfg_v1
    vals = []
    for a in range(len(ys)):
        ta = ys[a][0]
        for b in range(a + 1, len(ys)):
            dh = (ys[b][0] - ta) / 3600.0
            if dh < 0.15:
                continue
            if dh > 8.0:
                continue
            for k in range(PLACES):
                dy = ys[b][1][k] - ys[a][1][k]
                vals.append((dy * dy) / dh)
    if len(vals) < 5:
        return cfg_v1
    v = median(vals) * V1_SCALE
    return max(V1_MIN, min(V1_MAX, v))


def size_forecast(marks, cut_time, read_hours):
    """Mean and sd of each place's log10 cap at each read hour."""
    ys = top5_series(marks)
    if len(ys) < 2:
        # Fallback: the last top five, with a wide horizon-growing sd.
        last = sorted((logcap(r) for r in marks[-1]["rows"]), reverse=True)[:PLACES]
        while len(last) < PLACES:
            last.append(last[-1] if last else 2.0)
        out = {}
        for h in read_hours:
            out[h] = (list(last), [0.45 + 0.04 * math.sqrt(h)] * PLACES)
        return out

    # Window: last LEVEL_HIST_H hours, but always at least the last three marks.
    cutoff = cut_time - LEVEL_HIST_H * 3600.0
    window = [(t, caps) for t, caps in ys if t >= cutoff]
    if len(window) < 3:
        window = ys[-3:]

    # The level is the median of the five places; offsets describe the spread.
    levels = [median(caps) for _, caps in window]
    times = [((t - cut_time) / 3600.0) for t, _ in window]
    weights = [math.exp(max(-4.0, x / 1.5)) + 0.15 for x in times]
    points = list(zip(times, levels))
    slope = theil_sen(points) if len(points) >= 3 else weighted_slope(points, weights)

    last_level = levels[-1]
    level_ref = median(levels[-3:]) if len(levels) >= 3 else median(levels)
    base_level = (1.0 - LEVEL_SHRINK) * last_level + LEVEL_SHRINK * level_ref

    offsets = []
    for k in range(PLACES):
        vals = [caps[k] - median(caps) for _, caps in window]
        emp = median(vals) if vals else OFFSET_PRIOR[k]
        w = OFFSET_PRIOR_W / (1.0 + len(window) / OFFSET_PRIOR_N)
        offsets.append((1.0 - w) * emp + w * OFFSET_PRIOR[k])

    v1 = estimate_v1(ys)

    out = {}
    for h in read_hours:
        # Short horizons can follow the local trend closely; beyond an hour the
        # trend is damped and the total extrapolation is capped, so a transient
        # pump at the cut cannot fly the whole size distribution away.
        short = max(-LEVEL_SLOPE_SHORT, min(LEVEL_SLOPE_SHORT, slope)) * min(h, 1.0)
        long = (
            LEVEL_DAMP
            * max(-LEVEL_SLOPE_LONG, min(LEVEL_SLOPE_LONG, slope))
            * max(0.0, min(h, LEVEL_H_CAP) - 1.0)
        )
        growth = max(-LEVEL_GROWTH_CAP, min(LEVEL_GROWTH_CAP, short + long))
        level = base_level + growth
        means = [level + offsets[k] for k in range(PLACES)]

        sds = []
        for k in range(PLACES):
            residuals = [(caps[k] - median(caps)) - offsets[k] for _, caps in window]
            rsd = mad_sd(residuals)
            sd = math.sqrt(max(SIZE_SD_MIN, rsd) ** 2 + v1 * h)
            if k == 0:
                sd += SIZE_SD_P1
            sds.append(min(SIZE_SD_MAX, max(SIZE_SD_MIN, sd)))
        out[h] = (means, sds)
    return out


# --------------------------------------------------------------------------- #
# Order model
# --------------------------------------------------------------------------- #
def order_forecast(states, preds, read_hours, unseen, top_n):
    """Place probabilities for the named coins, averaged over the reads.

    preds[h][mint] = (mu, sigma).  Each read gets a softmax strength over the
    named field and a logistic soft rank; the two are combined with a Gaussian
    place kernel and normalized to that read's named mass 1 - W(h).
    """
    mints = sorted(states)
    # Fixed named set: the coins with the best average predicted cap.  This is
    # also the set that reduces the scorer's unnamed denominator most.
    avg_mu = {m: sum(preds[h][m][0] for h in read_hours) / len(read_hours) for m in mints}
    named = sorted(mints, key=lambda m: (-avg_mu[m], m))[:top_n]
    if len(named) < top_n:
        # pad with the remaining coins, if any
        for m in sorted(mints, key=lambda m: (-avg_mu[m], m)):
            if m not in named:
                named.append(m)
            if len(named) >= top_n:
                break

    total = {m: {k: 0.0 for k in range(1, PLACES + 1)} for m in named}
    for h in read_hours:
        w_unseen = unseen[h]
        tau_w = TAU_W_BASE + TAU_W_H * h
        tau_r = TAU_R_BASE + TAU_R_H * h
        tau_place = TAU_PLACE_BASE + TAU_PLACE_H * h
        mus = {m: preds[h][m][0] for m in named}
        max_mu = max(mus.values())
        weights = {m: math.exp((mus[m] - max_mu) / tau_w) for m in named}
        sw = sum(weights.values())
        # Soft ranks are measured inside the leading field only.  Including the
        # long tail of flat 28-SOL coins would add a near-constant offset to
        # every contender and make the place kernel far too sharp.
        order = sorted(named, key=lambda m: (-mus[m], m))
        field = order[:RANK_FIELD]
        soft_rank = {}
        for m in field:
            rr = 1.0
            for other in field:
                if other == m:
                    continue
                rr += 1.0 / (1.0 + math.exp((mus[m] - mus[other]) / tau_r))
            soft_rank[m] = rr
        # The pairwise sum carries a constant offset from the size of the
        # field.  Only differences matter to the place kernel, so put the
        # leader at rank 1.
        r_min = min(soft_rank.values())
        for m in field:
            soft_rank[m] = 1.0 + (soft_rank[m] - r_min)
        for idx, m in enumerate(order[RANK_FIELD:]):
            soft_rank[m] = RANK_FIELD + 1.0 + 0.2 * idx
        for k in range(1, PLACES + 1):
            raw = {}
            for m in named:
                phi = math.exp(-0.5 * ((k - soft_rank[m]) / tau_place) ** 2)
                raw[m] = (weights[m] / sw) * phi
            raw_sum = sum(raw.values())
            if raw_sum <= 0.0:
                for m in named:
                    total[m][k] += (1.0 - w_unseen) / len(named)
            else:
                scale = (1.0 - w_unseen) / raw_sum
                for m in named:
                    total[m][k] += raw[m] * scale

    n = float(len(read_hours))
    for m in named:
        for k in range(1, PLACES + 1):
            total[m][k] /= n
    return named, total


# --------------------------------------------------------------------------- #
# Forecast assembly
# --------------------------------------------------------------------------- #
def build_forecast(task, marks):
    edges = [float(x) for x in task["size"]["edges"]]
    chart_size = int(task.get("chart_size") or 100)
    places = int(task.get("places") or PLACES)
    places = min(PLACES, places)

    cut_time = max(parse_time(m["at"]) for m in marks)
    reads = [parse_time(x) for x in task.get("reads", [])]
    read_hours = sorted({round((t - cut_time) / 3600.0, 6) for t in reads if t > cut_time})
    if not read_hours:
        answer_at = parse_time(task.get("window", {}).get("answer_at") or task.get("history_until") or marks[-1]["at"])
        read_hours = [max(0.0, (answer_at - cut_time) / 3600.0)]

    states = build_coin_states(marks, cut_time, chart_size)
    if len(states) < 5:
        # Extremely defensive: use every mint the history names, in rank order.
        seen = []
        for mark in reversed(marks):
            for row in sorted(mark["rows"], key=lambda r: int(r.get("rank") or 10 ** 9)):
                if row["mint"] not in seen:
                    seen.append(row["mint"])
                if len(seen) >= 5:
                    break
            if len(seen) >= 5:
                break
        last5 = seen[:5]
        return {
            "probabilities": [{"mint": m, "place": k + 1, "p": 0.2} for k, m in enumerate(last5)],
            "top5": last5,
            "sizes": [
                {"place": p, "bin": b, "p": 1.0 / (len(edges) + 1)}
                for p in range(1, places + 1)
                for b in range(len(edges) + 1)
            ],
        }

    priors = bucket_priors(states)
    preds = {}
    for h in read_hours:
        preds[h] = {m: predict_coin(s, h, priors) for m, s in states.items()}

    unseen = estimate_unseen_share(marks, read_hours, chart_size)
    named, place_probs = order_forecast(states, preds, read_hours, unseen, MAX_ROWS_PER_PLACE)

    # Size distributions, averaged over the reads.
    size_means_sds = size_forecast(marks, cut_time, read_hours)
    size_bins = {p: [0.0] * (len(edges) + 1) for p in range(1, places + 1)}
    for h in read_hours:
        means, sds = size_means_sds[h]
        for p in range(1, places + 1):
            mean, sd = means[p - 1], sds[p - 1]
            core = bin_probabilities(mean, sd, edges)
            wide = bin_probabilities(mean, sd * SIZE_WIDE_FACTOR, edges)
            for b in range(len(edges) + 1):
                prob = (1.0 - SIZE_MIX_WIDE) * core[b] + SIZE_MIX_WIDE * wide[b]
                # A small uniform floor keeps a tail outcome above the scorer's
                # 1e-4 floor without changing the peak materially.
                size_bins[p][b] += (1.0 - UNIFORM_MIX) * prob + UNIFORM_MIX / (len(edges) + 1)
    for p in range(1, places + 1):
        total = sum(size_bins[p])
        if total > 0.0:
            size_bins[p] = [x / total for x in size_bins[p]]

    # Probability rows: only the named field, sorted deterministically.  The
    # mass a place leaves unnamed is 1 - sum(rows), which is what pays for the
    # coins born after the cut (and any history coin outside the named 100).
    probabilities = []
    for p in range(1, places + 1):
        items = [(m, place_probs[m][p]) for m in named]
        items.sort(key=lambda x: (-x[1], x[0]))
        s = sum(v for _, v in items)
        if s > 1.0:
            items = [(m, v / s) for m, v in items]
        for m, v in items:
            probabilities.append({"mint": m, "place": p, "p": max(0.0, min(1.0, v))})

    # top5: the five coins with the highest expected place value.
    score = {}
    for m in named:
        score[m] = sum(place_probs[m][p] * (places + 1 - p) for p in range(1, places + 1))
    top5 = sorted(named, key=lambda m: (-score[m], m))[:places]
    if len(top5) < places:
        # pad from history in rank order
        for mark in reversed(marks):
            for row in sorted(mark["rows"], key=lambda r: int(r.get("rank") or 10 ** 9)):
                if row["mint"] not in top5:
                    top5.append(row["mint"])
                if len(top5) >= places:
                    break
            if len(top5) >= places:
                break

    sizes = []
    for p in range(1, places + 1):
        for b, prob in enumerate(size_bins[p]):
            sizes.append({"place": p, "bin": b, "p": max(0.0, min(1.0, prob))})

    return {"probabilities": probabilities, "top5": top5[:places], "sizes": sizes}


# --------------------------------------------------------------------------- #
# Validation and output
# --------------------------------------------------------------------------- #
def validate(forecast, history_mints, edges, places=PLACES):
    if not isinstance(forecast, dict) or set(forecast.keys()) != {"probabilities", "top5", "sizes"}:
        raise ValueError("forecast must have exactly probabilities, top5, sizes")
    bins = len(edges) + 1
    by_place = {p: {} for p in range(1, places + 1)}
    for row in forecast["probabilities"]:
        if set(row.keys()) != {"mint", "place", "p"}:
            raise ValueError("probability row keys")
        mint = row["mint"]
        if not isinstance(mint, str) or not (32 <= len(mint) <= 44):
            raise ValueError("mint length")
        p = row["place"]
        if not isinstance(p, int) or not (1 <= p <= places):
            raise ValueError("place")
        if not isinstance(row["p"], (int, float)) or not (0.0 <= row["p"] <= 1.0):
            raise ValueError("probability")
        if mint in by_place[p]:
            raise ValueError("duplicate mint/place")
        by_place[p][mint] = row["p"]
    for p in range(1, places + 1):
        if len(by_place[p]) > MAX_ROWS_PER_PLACE:
            raise ValueError("too many rows per place")
        if sum(by_place[p].values()) > 1.0 + 1e-9:
            raise ValueError("place sum > 1")
    if len(forecast["top5"]) != places or len(set(forecast["top5"])) != places:
        raise ValueError("top5 shape")
    if any(m not in history_mints for m in forecast["top5"]):
        raise ValueError("top5 mint not in history")

    by_bin = {p: {} for p in range(1, places + 1)}
    for row in forecast["sizes"]:
        if set(row.keys()) != {"place", "bin", "p"}:
            raise ValueError("size row keys")
        p, b = row["place"], row["bin"]
        if not isinstance(p, int) or not (1 <= p <= places):
            raise ValueError("size place")
        if not isinstance(b, int) or not (0 <= b < bins):
            raise ValueError("size bin")
        if not isinstance(row["p"], (int, float)) or not (0.0 <= row["p"] <= 1.0):
            raise ValueError("size probability")
        if b in by_bin[p]:
            raise ValueError("duplicate size bin")
        by_bin[p][b] = row["p"]
    for p in range(1, places + 1):
        if sum(by_bin[p].values()) > 1.0 + 1e-9:
            raise ValueError("size place sum > 1")


def write_forecast(forecast, output_dir, filename):
    payload = json.dumps(forecast, separators=(",", ":"), ensure_ascii=True) + "\n"
    data = payload.encode("utf-8")
    if len(data) > MAX_BYTES:
        raise ValueError("forecast over 1 MiB")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, marks = read_inputs(input_dir)
    if not marks:
        raise SystemExit("no ok mark in the history")
    output_name = task.get("output") or "forecast.json"
    history_mints = set()
    for mark in marks:
        for row in mark["rows"]:
            history_mints.add(row["mint"])
    edges = [float(x) for x in task["size"]["edges"]]
    try:
        forecast = build_forecast(task, marks)
        validate(forecast, history_mints, edges, int(task.get("places") or PLACES))
    except Exception:
        # Never write an invalid file.  This fallback is the house baseline's
        # shape: persistence of the last mark's order, a normal size around its
        # caps, and a 0.5 unnamed share per place.
        top = []
        for mark in reversed(marks):
            for row in sorted(mark["rows"], key=lambda r: int(r.get("rank") or 10 ** 9)):
                if row["mint"] not in top:
                    top.append(row["mint"])
                if len(top) >= PLACES:
                    break
            if len(top) >= PLACES:
                break
        top = top[:PLACES]
        probabilities = []
        for p in range(1, PLACES + 1):
            if p <= len(top):
                probabilities.append({"mint": top[p - 1], "place": p, "p": 0.25})
            if p > 1 and p - 2 < len(top):
                probabilities.append({"mint": top[p - 2], "place": p, "p": 0.125})
            if p < PLACES and p < len(top):
                probabilities.append({"mint": top[p], "place": p, "p": 0.125})
        last_rows = sorted(marks[-1]["rows"], key=lambda r: int(r.get("rank") or 10 ** 9))
        sizes = []
        for p in range(1, PLACES + 1):
            mean = logcap(last_rows[min(p - 1, len(last_rows) - 1)])
            probs = bin_probabilities(mean, 0.35, edges)
            for b, prob in enumerate(probs):
                sizes.append({"place": p, "bin": b, "p": prob})
        forecast = {"probabilities": probabilities, "top5": top, "sizes": sizes}
    path = write_forecast(forecast, output_dir, output_name)
    print(path)


if __name__ == "__main__":
    main()
