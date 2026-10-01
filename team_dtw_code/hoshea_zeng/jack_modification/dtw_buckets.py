# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: blackrock-intraday
#     language: python
#     name: blackrock-intraday
# ---

# %% [markdown]
# # Bucket-conditioned DTW
#
# Does today's opening half-hour resemble other days' opening half-hours, closely enough
# to say which way the next 30 minutes goes?
#
# The existing DTW feature pools candidates by volatility regime and ignores the clock.
# This tests the opposite hypothesis: that price paths repeat **within intraday
# buckets**. The query runs from the start of the current bucket to the decision point,
# and is matched against the same bucket on earlier days.
#
# Nothing here touches the main pipeline. If the feature clears the thresholds below it
# gets added to `feature_engineering.py`; otherwise the directional question is closed.
#
# **Protocol.** Hyperparameters are tuned on 2016-2019, confirmed once on 2020, and
# 2021-2026 is left alone until the configuration is frozen. Every sweep is scored
# against a null built by shuffling outcomes, because a search this size will find
# something in noise.

# %%
import numpy as np
import pandas as pd
from dtaidistance import dtw
from pathlib import Path

SHARED_PROCESSED = Path("../data/processed")
JACK_DATA = Path("data")

TUNE_YEARS = (2016, 2019)
CONFIRM_YEAR = 2020
BAR_MINUTES = 30

pd.set_option("display.width", 160)

bars = pd.read_parquet(JACK_DATA / "es_30min_bars.parquet")
clean = pd.read_parquet(SHARED_PROCESSED / "es_1min_clean.parquet")
et = bars.index.tz_convert("America/New_York")
bars = bars.assign(slot=et.strftime("%H:%M"))
print(f"{len(bars):,} bars, {bars.slot.nunique()} slots, "
      f"{bars.index.min().date()} to {bars.index.max().date()}")


# %% [markdown]
# ## Bucket definitions
#
# Slots are ordered by position in the session, which starts at 18:00 ET, so that a
# contiguous block of clock time is a contiguous block of the array.

# %%
def session_key(slot: str) -> int:
    minutes = int(slot[:2]) * 60 + int(slot[3:])
    return minutes - 18 * 60 if minutes >= 18 * 60 else minutes + 6 * 60


SLOT_ORDER = sorted(bars.slot.unique(), key=session_key)
log_volume = np.log(bars.groupby("slot").volume.mean().reindex(SLOT_ORDER).to_numpy())


def dp_segment(x: np.ndarray, k: int) -> list[tuple[int, int]]:
    """Optimal contiguous k-segment partition minimising within-segment SSE."""
    n = len(x)
    cost = np.full((n, n), np.inf)
    for i in range(n):
        total = sq = 0.0
        for j in range(i, n):
            total += x[j]
            sq += x[j] ** 2
            cost[i, j] = sq - total * total / (j - i + 1)

    dp = np.full((k + 1, n + 1), np.inf)
    back = np.zeros((k + 1, n + 1), dtype=int)
    dp[0, 0] = 0
    for kk in range(1, k + 1):
        for j in range(kk, n + 1):
            for i in range(kk - 1, j):
                c = dp[kk - 1, i] + cost[i, j - 1]
                if c < dp[kk, j]:
                    dp[kk, j] = c
                    back[kk, j] = i

    bounds, j = [], n
    for kk in range(k, 0, -1):
        i = back[kk, j]
        bounds.append((i, j))
        j = i
    return list(reversed(bounds))


HAND_BUCKETS = {
    "asia": ("18:00", "02:30"), "europe": ("03:00", "07:30"),
    "us_pre": ("08:00", "09:00"), "us_open": ("09:30", "10:30"),
    "us_mid": ("11:00", "14:30"), "us_close": ("15:00", "16:30"),
}


def buckets_contiguous(k: int) -> dict[str, int]:
    return {SLOT_ORDER[i]: b for b, (a, z) in enumerate(dp_segment(log_volume, k))
            for i in range(a, z)}


def buckets_kmeans(k: int) -> dict[str, int]:
    from sklearn.cluster import KMeans
    x = ((log_volume - log_volume.mean()) / log_volume.std()).reshape(-1, 1)
    labels = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(x)
    return dict(zip(SLOT_ORDER, labels))


def buckets_hand() -> dict[str, int]:
    out = {}
    for b, (name, (start, end)) in enumerate(HAND_BUCKETS.items()):
        lo, hi = session_key(start), session_key(end)
        for s in SLOT_ORDER:
            if lo <= session_key(s) <= hi:
                out[s] = b
    return {s: out.get(s, len(HAND_BUCKETS) - 1) for s in SLOT_ORDER}


BUCKET_METHODS = {f"contiguous_{k}": (lambda k=k: buckets_contiguous(k)) for k in (3, 5, 7)}
BUCKET_METHODS |= {f"kmeans_{k}": (lambda k=k: buckets_kmeans(k)) for k in (3, 5, 7)}
BUCKET_METHODS["hand"] = buckets_hand

for name, fn in BUCKET_METHODS.items():
    assign = fn()
    sizes = pd.Series(assign).value_counts().sort_index().to_list()
    print(f"{name:<14} {len(set(assign.values()))} buckets, slot counts {sizes}")

# %% [markdown]
# **Findings:** Contiguity changes the answer where it matters. Unconstrained k-means
# puts the 09:30 open and the 15:30 close in one bucket because both trade near 160k
# contracts. The constrained version at k=7 isolates the open (09:30-10:30, 128k) from
# the rest of the US session (11:00-16:00, 74k). Both produce a singleton bucket at
# 16:30, which has no internal progress for a query to run through.

# %% [markdown]
# ## Windows and matching
#
# The fine grid is gap-free, so a window can never splice two distant times together.
# For a decision at bar *t* the query runs from the start of *t*'s bucket to *t*'s close,
# capped at `MAX_QUERY_BARS`. A reference is the same bucket on an earlier day shifted by
# an offset, and the forward return is read **from where the match ends**, not from the
# clock.
#
# The cap matters. Without it the overnight bucket produces a nine-hour query, which is
# a different object from the half-hour shape this is meant to test.

# %%
RESOLUTION = 5                 # minutes per fine bar
MAX_QUERY_BARS = 18            # 90 minutes
TOLERANCE = 6                  # +/- 30 min of alignment float
OFFSET_STEP = 3
MIN_SLOT_ROWS = 150

# The sweep scores configurations on every SWEEP_STRIDE-th training session, which cuts
# the pairwise cost roughly fourfold. It only has to rank configurations; the winner is
# recomputed on every session in Stage F.
SWEEP_STRIDE = 2

TRAIN = np.asarray((bars.index.year >= TUNE_YEARS[0]) & (bars.index.year <= TUNE_YEARS[1]))
_sessions = pd.Index(sorted(bars.session_date.unique()))
_keep = set(_sessions[::SWEEP_STRIDE])
SWEEP = TRAIN & bars.session_date.isin(_keep).to_numpy()
TARGET = bars.fwd_ret_30m.to_numpy()


FEATURES = pd.read_parquet(JACK_DATA / "features.parquet")


def build_grid(res: int):
    fine = clean.close.resample(f"{res}min").last()
    grid = pd.date_range(fine.index.min(), fine.index.max(), freq=f"{res}min", tz="UTC")
    fine = fine.reindex(grid)
    steps = BAR_MINUTES // res
    fwd = np.log(fine.shift(-steps) / fine).to_numpy()
    anchor = pd.Series(np.arange(len(grid)), index=grid).reindex(
        bars.index + pd.Timedelta(minutes=BAR_MINUTES - res)).to_numpy()
    # Volatility prevailing at any fine bar, for z-scoring a neighbour's outcome.
    vol = FEATURES.realised_vol.reindex(grid, method="ffill").to_numpy()
    return fine.to_numpy(), fwd, anchor, steps, vol


def to_shape(w: np.ndarray) -> np.ndarray:
    """Cumulative log return, detrended, unit variance."""
    cum = np.log(w) - np.log(w[:, [0]])
    t = np.arange(w.shape[1], dtype=float)
    tc = t - t.mean()
    slope = (cum * tc).sum(axis=1) / (tc**2).sum() if w.shape[1] > 1 else 0.0
    detrended = cum - (slope[:, None] * tc[None, :] + cum.mean(axis=1)[:, None])
    scale = detrended.std(axis=1, keepdims=True)
    return detrended / np.where(scale > 0, scale, np.nan)


def bucket_position(assign: dict[str, int]) -> dict[str, int]:
    """How many slots into its bucket each slot sits."""
    out = {}
    for b in set(assign.values()):
        for i, s in enumerate([s for s in SLOT_ORDER if assign.get(s) == b]):
            out[s] = i
    return out


def compute_cache(assign, res=RESOLUTION, tol=TOLERANCE, ostep=OFFSET_STEP,
                  max_q=MAX_QUERY_BARS, mask=None, bucket_relative=False):
    """Per slot: rows, anchors, min-over-offset distances, and the winning offset.

    Everything downstream (lookback, k, z-scoring of the output) is a cheap
    post-processing step on this, so the expensive part runs once per
    (buckets, resolution, tolerance).
    """
    fv, fwd, anchor, steps, vol = build_grid(res)
    pos = bucket_position(assign)
    mask = TRAIN if mask is None else mask
    offsets = list(range(-tol, tol + 1, ostep))
    cache = {}

    for slot in SLOT_ORDER:
        rows = np.flatnonzero(mask & (bars.slot == slot).to_numpy())
        if len(rows) < MIN_SLOT_ROWS:
            continue
        L = min((pos[slot] + 1) * steps, max_q)
        a = anchor[rows]
        ok = np.isfinite(a) & (a - L + 1 - tol >= 0) & (a + tol + steps < len(fv))
        rows, a = rows[ok], a[ok].astype(int)
        if len(rows) < MIN_SLOT_ROWS:
            continue

        window_idx = np.arange(-L + 1, 1)
        query = to_shape(fv[a[:, None] + window_idx])
        if bucket_relative:
            query = query - np.nanmean(query, axis=0)     # remove the bucket's mean path
        good = np.isfinite(query).all(axis=1)
        query = np.ascontiguousarray(query[good])
        rows, a = rows[good], a[good]
        n = len(query)
        if n < MIN_SLOT_ROWS:
            continue

        band = max(2, L // 6)
        best = np.full((n, n), np.inf, np.float32)
        best_off = np.zeros((n, n), np.int16)
        for o in offsets:
            ref = to_shape(fv[(a + o)[:, None] + window_idx])
            if bucket_relative:
                ref = ref - np.nanmean(ref, axis=0)
            fine_ok = np.isfinite(ref).all(axis=1)
            combined = np.ascontiguousarray(
                np.vstack([query, np.where(fine_ok[:, None], ref, 0.0)]))
            d = dtw.distance_matrix_fast(combined, window=band)[:n, n:]
            d = np.where(np.isfinite(d), d, np.inf).astype(np.float32)
            d[:, ~fine_ok] = np.inf
            take = d < best
            best = np.where(take, d, best)
            best_off = np.where(take, o, best_off)

        cache[slot] = dict(rows=rows, anchor=a, dist=best, offset=best_off)

    return cache, fwd, vol


def build_feature(cache, fwd, vol, lookback=250, k=20, halflife=125,
                  z_returns=True, similarity=True):
    """Weighted mean outcome of the k nearest earlier days, read from each match end.

    Three weightings, all on by default:
      z_returns  divide each neighbour's outcome by the volatility prevailing at its
                 match end, so a handful of high-vol neighbours cannot set the sign
      similarity weight by exp(-z) on the within-k distances
      halflife   decay by session age, in calendar sessions

    `lookback` and `halflife` are in **calendar sessions**; the stride conversion is
    applied here rather than left implicit.
    """
    feature = np.full(len(bars), np.nan)
    noise = []
    span = max(1, lookback // SWEEP_STRIDE)          # array positions, not sessions
    for slot, c in cache.items():
        rows, a, dist, off = c["rows"], c["anchor"], c["dist"], c["offset"]
        for i in range(1, len(rows)):
            pool = np.arange(max(0, i - span), i)
            pool = pool[np.isfinite(dist[i, pool])]
            if len(pool) < k:
                continue
            nearest = pool[np.argpartition(dist[i, pool], k - 1)[:k]]
            d = dist[i, nearest]
            ends = a[nearest] + off[i, nearest]
            outcomes = fwd[ends]
            if z_returns:
                v = vol[ends]
                outcomes = outcomes / np.where(v > 0, v, np.nan)
            ok = np.isfinite(outcomes)
            if ok.sum() < k // 2:
                continue

            if similarity:
                spread = d[ok].std()
                z = (d[ok] - d[ok].mean()) / spread if spread > 0 else np.zeros(ok.sum())
                w = np.exp(-z)
            else:
                w = np.ones(ok.sum())
            if halflife:
                age = (i - nearest)[ok] * SWEEP_STRIDE     # array steps -> sessions
                w = w * 0.5 ** (age / halflife)

            feature[rows[i]] = np.sum(w * outcomes[ok]) / w.sum()
            noise.append(outcomes[ok].std() / np.sqrt(ok.sum()))
    return feature, (np.mean(noise) if noise else np.nan)


def ic(feature, target=None, mask=None):
    target = TARGET if target is None else target
    mask = TRAIN if mask is None else mask
    ok = mask & np.isfinite(feature) & np.isfinite(target)
    if ok.sum() < 1000:
        return np.nan, 0
    return float(np.corrcoef(feature[ok], target[ok])[0, 1]), int(ok.sum())


print(f"training bars {TRAIN.sum():,}   sweep subset {SWEEP.sum():,}   "
      f"target defined on {np.isfinite(TARGET).sum():,}")

# %% [markdown]
# ## Stage A: bucket definition
#
# Seven candidates, with everything else at defaults: 5-minute resolution, ±30 minutes of
# alignment float, a 250-session (one year) lookback, and k = 20, with outcomes
# z-scored by the volatility at their match end, weighted by similarity, and decayed on
# a 125-session halflife. Scored on every second training session, which is enough to
# rank them.
#
# One session is one trading day; the cleaned data averages 244 a year, so 250 sessions
# is a round year. The sweep samples every second session, so a 250-session lookback is
# 125 array positions, which is where the earlier mislabelled "125" came from.
#
# Each configuration is also scored against a **null**, built by shuffling the target
# within slot so the feature keeps its structure while the outcome becomes random. The
# statistic that matters is the best IC the whole search reaches under that null, since
# that is what the real sweep has to beat.

# %%
N_SHUFFLES = 20
rng = np.random.default_rng(11)


def shuffled_targets(n: int) -> list[np.ndarray]:
    """Permute the target within each slot, preserving time-of-day structure."""
    out = []
    slot_rows = {s: np.flatnonzero(SWEEP & (bars.slot == s).to_numpy()) for s in SLOT_ORDER}
    for _ in range(n):
        t = TARGET.copy()
        for rows in slot_rows.values():
            vals = t[rows]
            t[rows] = rng.permutation(vals)
        out.append(t)
    return out


NULL_TARGETS = shuffled_targets(N_SHUFFLES)

stage_a, null_a = [], []
for name, fn in BUCKET_METHODS.items():
    cache, fwd, vol = compute_cache(fn(), mask=SWEEP)
    feat, noise = build_feature(cache, fwd, vol, lookback=250)
    real, n = ic(feat, mask=SWEEP)
    nulls = [ic(feat, t, mask=SWEEP)[0] for t in NULL_TARGETS]
    null_a.append(nulls)
    stage_a.append({"buckets": name, "n": n, "IC": real,
                    "t": real * np.sqrt(n) if np.isfinite(real) else np.nan,
                    "noise": noise, "coverage": np.isfinite(feat).mean()})

stage_a = pd.DataFrame(stage_a).set_index("buckets")
print(stage_a.round(4).to_string())

null_best = np.max(np.abs(np.array(null_a)), axis=0)     # best of the sweep, per shuffle
print(f"\nnull: best |IC| across the {len(stage_a)} configurations, over "
      f"{N_SHUFFLES} shuffles")
print(f"  mean {null_best.mean():.4f}   p90 {np.percentile(null_best, 90):.4f}   "
      f"max {null_best.max():.4f}")
print(f"real: best |IC| = {stage_a.IC.abs().max():.4f} "
      f"({stage_a.IC.abs().idxmax()})")

# %% [markdown]
# **Findings: no bucket definition beats the null.** The seven methods span |IC| 0.0015
# to 0.0071, all positive, with t-statistics from 0.21 to 1.00. The same search run
# against shuffled outcomes reaches 0.0114 on average and 0.0157 at the 90th percentile,
# so every real configuration lands below the average null. The spread between the seven
# is not information.
#
# `hand` scores highest at 0.0071 and is carried forward, which is also what the
# no-fitting tie-break would have chosen.

# %% [markdown]
# ## Stages B to E: the remaining hyperparameters
#
# Alignment tolerance, pool depth, k, resolution, and z-scoring, each swept with the
# others held at defaults. Every stage is scored against the same null.

# %%
BUCKETS = buckets_hand()
null_all = list(null_a)


def sweep(label, variants, builder):
    rows = []
    for name, kwargs in variants.items():
        cache, fwd, vol = builder(kwargs)
        feat, noise = build_feature(cache, fwd, vol,
                                    lookback=kwargs.get("lookback", 250),
                                    k=kwargs.get("k", 20),
                                    halflife=kwargs.get("halflife", 125),
                                    z_returns=kwargs.get("z_returns", True),
                                    similarity=kwargs.get("similarity", True))
        real, n = ic(feat, mask=SWEEP)
        null_all.append([ic(feat, t, mask=SWEEP)[0] for t in NULL_TARGETS])
        rows.append({label: name, "n": n, "IC": real,
                     "t": real * np.sqrt(n) if np.isfinite(real) else np.nan,
                     "noise": noise})
    out = pd.DataFrame(rows).set_index(label)
    print(f"\n{label}:")
    print(out.round(4).to_string())
    return out


# Stage B: alignment tolerance
stage_b = sweep("tolerance", {"exact": dict(tol=0, ostep=1), "+/-15min": dict(tol=3, ostep=3),
                              "+/-30min": dict(tol=6, ostep=3), "+/-60min": dict(tol=12, ostep=3)},
                lambda kw: compute_cache(BUCKETS, mask=SWEEP, tol=kw["tol"], ostep=kw["ostep"]))

# Stage C and D share one cache: lookback and k are selections on the same distances.
_cache, _fwd, _vol = compute_cache(BUCKETS, mask=SWEEP)
stage_c = sweep("lookback", {f"{n} sessions ({n / 244:.2g}y)": dict(lookback=n)
                             for n in (20, 60, 120, 250, 500, 1000)},
                lambda kw: (_cache, _fwd, _vol))
stage_d = sweep("k", {f"k={k}": dict(k=k) for k in (5, 10, 20, 50, 100)},
                lambda kw: (_cache, _fwd, _vol))
stage_w = sweep("weighting", {
    "none (plain mean)": dict(z_returns=False, similarity=False, halflife=None),
    "similarity only": dict(z_returns=False, similarity=True, halflife=None),
    "z-returns only": dict(z_returns=True, similarity=False, halflife=None),
    "z + similarity": dict(z_returns=True, similarity=True, halflife=None),
    "z + sim + 30d decay": dict(halflife=30),
    "z + sim + 60d decay": dict(halflife=60),
    "z + sim + 125d decay": dict(halflife=125),
    "z + sim + 250d decay": dict(halflife=250),
}, lambda kw: (_cache, _fwd, _vol))

# Stage D continued: resolution
stage_d2 = sweep("resolution", {f"{r}min": dict(res=r) for r in (2, 5, 10)},
                 lambda kw: compute_cache(BUCKETS, mask=SWEEP, res=kw["res"]))

# Stage E: z-scoring the path within its bucket
stage_e = sweep("path_norm", {"shape only": dict(br=False), "bucket-relative": dict(br=True)},
                lambda kw: compute_cache(BUCKETS, mask=SWEEP, bucket_relative=kw["br"]))

# %%
null_matrix = np.abs(np.array(null_all, dtype=float))
null_matrix = null_matrix[~np.isnan(null_matrix).all(axis=1)]     # drop degenerate configs
best_null = np.nanmax(null_matrix, axis=0)
all_real = pd.concat([stage_a.IC, stage_b.IC, stage_c.IC, stage_d.IC,
                      stage_d2.IC, stage_e.IC, stage_w.IC]).dropna()
best_real = all_real.abs().max()

print(f"\n{'':-<66}")
print(f"configurations evaluated : {len(all_real)}")
print(f"best real |IC|           : {best_real:.4f}  ({all_real.abs().idxmax()})")
print(f"null best-of-sweep |IC|  : mean {best_null.mean():.4f}   "
      f"median {np.median(best_null):.4f}   p90 {np.percentile(best_null, 90):.4f}   "
      f"max {best_null.max():.4f}")
print(f"percentile of the real best within the null: "
      f"{100 * (best_null < best_real).mean():.0f}th")
print(f"\nkill criteria")
print(f"  best training |IC| < 0.02          : {best_real < 0.02}  ({best_real:.4f})")
print(f"  best real inside null distribution : {best_real < np.percentile(best_null, 90)}")

# %% [markdown]
# **Findings: weighting helps, and the answer does not change.**
#
# The weighting sweep settles what the aggregation should be. A plain mean gives -0.0020;
# similarity weighting alone +0.0046; z-scoring the outcomes alone +0.0011; the two
# together **+0.0074**. Both contribute, and they contribute more together than either
# does alone.
#
# Recency decay adds a little. Across no decay and halflives of 30, 60, 125 and 250
# sessions the z-scored and similarity-weighted feature gives 0.0074, **0.0100**, 0.0076,
# 0.0071 and 0.0071. The 30-session setting is the best of the row, and it is a mild bump
# rather than a spike: the row spans 0.0029 against 0.0170 for the same sweep on a plain
# mean, which runs -0.0020, +0.0150, +0.0083, +0.0026, +0.0001. A response that moves
# within a narrow band is an estimator behaving itself; one that spikes and collapses on
# both sides is a noise surface.
#
# **The verdict is unchanged.** Across 34 configurations the best is |IC| 0.0167 (k=100)
# against a null whose median best is 0.0144 and whose 90th percentile is 0.0225. The
# real winner sits at the **65th percentile of its own null**, up from the 50th before
# weighting was fixed, and still inside it. Both kill criteria fire.
#
# Two details worth keeping. The 20-session lookback returns nothing at all, because a
# pool of twenty days cannot supply twenty neighbours; the original "past couple of days"
# framing is arithmetically impossible at any useful k. And the nominal winner, k = 100
# drawn from a 250-session pool sampled at stride 2, selects roughly 80% of everything
# available, so the distance ordering barely determines membership. The best-scoring
# configuration is the one where DTW is doing least.

# %% [markdown]
# ## Stage F: not run
#
# The plan called for a single-shot confirmation on 2020 followed by integration. Both
# kill criteria fired on the tuning period, so neither happens. **2020 and 2021-2026 were
# never touched**, which keeps them clean for whatever comes next.
#
# One thing was not tested. The literal design was a full-bucket subsequence search;
# what ran was bounded offsets out to ±60 minutes, because a full search costs 4.6 hours
# per configuration against 15 seconds for bounded offsets. Widening the tolerance from
# exact alignment through ±60 minutes moved the IC around inside the null without
# trending, so a full search is unlikely to differ, but it was not run and should not be
# described as ruled out.
#
# ### What this closes
#
# The sweep now uses the aggregation the design actually called for: outcomes z-scored
# by the volatility at their match end, weighted by similarity, with an optional recency
# decay. An earlier version of this notebook took a plain unweighted mean, which threw
# away the distance information that is the point of DTW. Fixing it moved the result from
# the 50th to the 65th percentile of the null and left the conclusion where it was.
#
# Bucket-conditioned path matching was the last untested version of the DTW idea. With
# volatility-regime pooling already flat and clock-bucket pooling now flat against a
# proper null, price-path shape does not carry directional information at this horizon
# in this sample. Taken with the seven-feature OLS, the win-rate analysis, and the
# intraday attribution, the directional question is closed for this feature set.
#
# The measurements that keep working are all about magnitude: `vol_z` at +0.40 against
# absolute return, the intraday volatility profile persisting at +0.87 out of sample, and
# event days separating 9.7 from 8.6 basis points. That is where the remaining work is.

# %% [markdown]
# ## Anchored paths: the session-landmark design
#
# A later attempt, built after the magnitude version succeeded, on the hypothesis that a
# path should be measured against a session landmark rather than floated free. If we are
# an hour past the cash open we compare against other first hours; if we are just past
# the cash close we compare against other post-close stretches.
#
# This is a different idea from the bucket pooling above, and worth separating. The
# earlier designs *detrended* the query path so DTW would stay complementary to momentum,
# which removed the drift from the anchor. That drift is exactly what this design
# conditions on, and removing it was arguably the same class of mistake as z-normalising
# amplitude away for magnitude.
#
# **Construction.** The anchor is the most recent cash open (09:30 ET) or cash close
# (16:00 ET). The path is the cumulative log return from that anchor to the decision
# point, resampled to 24 points and scaled by prevailing volatility. The pool is
# restricted to bars with the same anchor kind and a session phase within one 30-minute
# bin, so opens match opens. Candidates must be at least a day old and drawn from
# strictly earlier calendar years.
#
# **Timing.** A five-minute bar labelled `T` carries data through `T+5`, so the query ends
# at `t+25`, not `t+30`. Getting this wrong puts the first five minutes of the target
# window inside the query and produced a directional IC of +0.29 before it was caught.
#
# Two path types, anchored and a plain trailing two-hour path, against two targets: the
# forward return, and the gap between upside and downside excursion inside the window.
# The second target exists because entry and exit can fall anywhere in the window, so an
# excursion that never reaches the close is still capturable.

# %% [markdown]
# **Findings:** Nothing clears its null. Against 20 shuffled-outcome and 20
# random-neighbour controls per cell:
#
# | path / target | 2019-2020 IC | controls beating | 2021-2026 IC | controls beating |
# |---|---|---|---|---|
# | anchored / forward return | +0.0051 | 21/40 | +0.0101 | 2/40 |
# | anchored / excursion gap | -0.0026 | 30/40 | +0.0028 | 23/40 |
# | trailing / forward return | -0.0021 | 34/40 | -0.0002 | 40/40 |
# | trailing / excursion gap | -0.0014 | 34/40 | +0.0005 | 37/40 |
#
# Comparing best-of-four against best-of-four controls, which is the right correction for
# having tried four variants, 32 of 40 controls beat the real result on the confirmation
# window. The one cell that looks strong, anchored against the forward return on
# 2021-2026, does not replicate on 2019-2020.
#
# Anchoring does beat the trailing path in every cell, so the reference-point intuition
# points the right way. It is not enough. The scalar version of the same idea, the drift
# from the anchor on its own, scores -0.0156 and +0.0001, so there is nothing for DTW to
# add to.

# %% [markdown]
# ## One open lead: the 16:00 ET window
#
# Splitting the anchored result by session phase to see where any signal concentrates
# produced one cell that behaves differently from the other nine.
#
# Bars labelled 16:00 ET, predicting the 16:30-17:00 ET window, score an IC of **+0.1557**
# with none of 40 controls beating it, z = +7.0. It survives a split-half, +0.117 on
# 2022-2023 and +0.160 on 2024-2026, and it is not a drift artefact: the window closes up
# 46.2% of the time and averages -0.03 basis points. Sign-based, it is +1.24 basis points
# per trade gross and +1.11 net of a quarter-tick, t = +4.29, on roughly 244 trades a
# year, or about 2.7% of notional annually per contract.
#
# **Four reasons to hold it at arm's length.**
#
# It was the best of ten post-hoc subgroup tests, which demands a discount on its own.
#
# It cannot be checked on the confirmation window, because the 16:00 ET bar does not exist
# before 2021. The bar spans 16:00-16:30 ET, the old daily halt ran 16:15-16:30, and a bar
# with 15 minutes of data fails the 23-minute minimum in `cleaning.py` and is dropped. So
# the split-half above is the only validation available and both halves come from the
# period that was searched.
#
# The sample is 1,099 bars in a single clock slot at the thin end of the day, sitting
# directly on a market-structure change.
#
# Wider spreads in that window have not been modelled, and 2.7% of notional does not
# survive much of that.
#
# The honest test is prospective: freeze the configuration and check it on data as it
# arrives. It is recorded here as a lead, not a result.

# %% [markdown]
# ## Does the normalisation matter?
#
# The magnitude feature succeeded partly because it stopped z-normalising the path, so
# the obvious question is whether the directional failures are a normalisation problem
# too. Six schemes were applied to the same anchored cumulative price path, with the same
# phase-restricted pooling and the same 40 controls each:
#
# - `volscale`, divide by prevailing volatility, preserving relative amplitude
# - `zscore`, unit mean and variance within the window, the classic that failed before
# - `minmax`, rescaled to [-1, 1] by the window's own range
# - `rank`, each point replaced by its rank within the path, fully scale-free
# - `l2`, unit norm
# - `detrend_z`, linear trend removed then z-scored, the original construction

# %% [markdown]
# **Findings:** The normalisation does not matter, because there is nothing for it to
# reveal.
#
# | scheme | 2019-2020 IC | beaten by | 2021-2026 IC | beaten by |
# |---|---|---|---|---|
# | `volscale` | +0.0028 | 29/40 | +0.0077 | 3/40 |
# | `zscore` | +0.0028 | 24/40 | +0.0024 | 22/40 |
# | `minmax` | -0.0009 | 35/40 | -0.0023 | 22/40 |
# | `rank` | -0.0031 | 25/40 | -0.0024 | 22/40 |
# | `l2` | +0.0053 | 13/40 | -0.0074 | 4/40 |
# | `detrend_z` | -0.0026 | 27/40 | +0.0071 | 4/40 |
#
# Best-of-six against best-of-six controls, the right correction for having tried six,
# leaves 24 of 40 controls ahead on the confirmation window and 8 of 40 on the main one.
#
# The sign flips settle it. `l2` scores +0.0053 on one window and -0.0074 on the other;
# `detrend_z` goes -0.0026 to +0.0071. A representation that carried directional
# information would not reverse sign between adjacent periods. Every scheme lands in the
# same place, which is the place all six directional designs have landed.
#
# This is worth stating plainly because the magnitude result invited the opposite
# conclusion. There, dropping the z-normalisation mattered enormously, because amplitude
# was the signal being discarded. For direction there is no equivalent: the information is
# absent from the path in every representation tried, so no transform recovers it.
