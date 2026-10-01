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
# # Feature engineering — `dtw_dir` and `dtw_disp`
#
# One row per (asset, event family, release timestamp, anchor). Two features:
#
# * `dtw_dir` — the direction signal: the weighted mean forward return of the k most
#   similar past reactions.
# * `dtw_disp` — the weighted dispersion of those neighbours' forward returns, which sizes
#   the trade.
#
# Plus the universe screen (`admitted`), which decides which release families are traded.
#
# The DTW setting comes from `config.py`: the original ES book's (k = 15, 3-year lookback).
# Here it is scored on the training years (2013-2017) and the validation
# years (2018-2020) by how well each feature ranks what came next. The test years are never
# scored here.
#
# **Nothing uses data at or after the entry instant.** The DTW path ends exactly at entry,
# neighbours are strictly earlier releases, and the universe screen for a year uses only
# prior years.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dtaidistance import dtw
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else "."))
from config import (
    ANCHORS, BLOOMBERG, BOOK_SYMBOLS, CLEAN_BARS_FMT, DTW_GRID, DTW_HALFLIFE_DAYS, DTW_MAX_POOL,
    DTW_WARP_MINUTES, DTW_WEIGHTS, ENTRY_LAG_MIN, FAMILY_OVERLAP,
    HOLD_MIN, IMPACT_MIN_T, IMPACT_WINDOW_MIN, LEAK_TEST_MINUTES, MIN_NEIGHBOURS,
    MIN_RELEASES, NY, OUT, PANEL, PANEL_LEAKED, PATH_FREQ_MIN, PATH_PRE_MIN,
    TRAIN_YEARS_SPAN, VALID_YEARS_SPAN,
)

pd.set_option("display.width", 170)


class Px:
    """Exact-timestamp lookup. Cleaning fills every minute inside a session, so NaN means
    the minute is outside any kept session — the path is invalid, never interpolated."""

    def __init__(self, s: pd.Series):
        self.ts = s.index.values.astype("datetime64[ns]").astype(np.int64)
        self.v = s.to_numpy(float)

    def at(self, stamps) -> np.ndarray:
        q = pd.DatetimeIndex(stamps).tz_convert("UTC").values.astype("datetime64[ns]").astype(np.int64)
        pos = np.searchsorted(self.ts, q)
        pc = np.clip(pos, 0, len(self.ts) - 1)
        out = np.full(len(q), np.nan)
        hit = (pos < len(self.ts)) & (self.ts[pc] == q)
        out[hit] = self.v[pc[hit]]
        return out


# %%
symbols = BOOK_SYMBOLS
pxs, contracts = {}, {}
for s in symbols:
    b = pd.read_parquet(CLEAN_BARS_FMT.format(symbol=s), columns=["close", "contract"]).sort_index()
    b.index = pd.to_datetime(b.index, utc=True)
    pxs[s], contracts[s] = Px(b.close), Px(b.contract.astype(float))
    print(f"{s}: {len(b):,} clean bars  {b.index.min().date()} -> {b.index.max().date()}")

# %% [markdown]
# ## 1. Event families
#
# **Intuition.** CPI MoM, Core CPI MoM and CPI Index NSA are several views of one number
# printed at one instant. Treating them as separate signals trades the same minute
# several times.
#
# **Calculation.** Any two release lines sharing ≥80% of their release timestamps are
# merged into one family, transitively. The family is named after its longest-history
# member plus its clock slot.
#
# *Naming caveat:* longest-history picks obscure labels — the NFP family comes out as
# "Change in Manufact. Payrolls @ 08:30" and CPI as "CPI Index NSA @ 08:30". The grouping
# is right, the label is not.


# %%
def load_lines() -> pd.DataFrame:
    b = pd.concat([pd.read_excel(BLOOMBERG, sheet_name=s)
                   for s in pd.ExcelFile(BLOOMBERG).sheet_names], ignore_index=True)
    b.columns = [c.split(".")[-1] if "DROPNA" in str(c) else c for c in b.columns]
    b = b[b.COUNTRY_NAME.astype(str).str.contains("United States", na=False)]
    b["RELEASE_DATE"] = pd.to_datetime(b.RELEASE_DATE, errors="coerce")
    b = b.dropna(subset=["RELEASE_DATE", "RELEASE_TIME", "EVENT_NAME"])
    t = pd.to_datetime(b.RELEASE_TIME.astype(str), format="%H:%M:%S", errors="coerce")
    b, t = b[t.notna()].copy(), t[t.notna()]
    local = (b.RELEASE_DATE.dt.normalize() + pd.to_timedelta(t.dt.hour, "h")
             + pd.to_timedelta(t.dt.minute, "m"))
    b["ts"] = local.dt.tz_localize(NY, ambiguous=True, nonexistent="shift_forward").dt.tz_convert("UTC")
    b["clock_et"] = b.ts.dt.tz_convert(NY).dt.strftime("%H:%M")
    return b[["ts", "clock_et", "EVENT_NAME"]].rename(columns={"EVENT_NAME": "line"})


def assign_families(lines: pd.DataFrame) -> dict[str, str]:
    stamps = {ln: set(g.ts) for ln, g in lines.groupby("line") if len(g) >= 12}
    names = sorted(stamps, key=lambda n: -len(stamps[n]))
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(names):
        for c in names[i + 1:]:
            if find(a) == find(c):
                continue
            shared = len(stamps[a] & stamps[c])
            if shared and shared / min(len(stamps[a]), len(stamps[c])) >= FAMILY_OVERLAP:
                parent[find(c)] = find(a)
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    out = {}
    for members in groups.values():
        best = max(members, key=lambda m: len(stamps[m]))
        clock = lines.loc[lines.line == best, "clock_et"].mode().iloc[0]
        for m in members:
            out[m] = f"{best} @ {clock}"
    return out


lines = load_lines()
fam = assign_families(lines)
lines["family"] = lines.line.map(fam)
lines = lines.dropna(subset=["family"])
fam_stamps = {f: sorted(set(g.ts)) for f, g in lines.groupby("family")}
print(f"{lines.line.nunique()} release lines -> {len(fam_stamps)} families")

# %% [markdown]
# **Findings:** 163 lines collapse to 67 families. The largest are the NFP complex (12
# lines: payrolls, hourly earnings, unemployment rate, participation) and the CPI complex
# (8 lines).

# %% [markdown]
# ## 2. The game panel — the anchor ladder
#
# **Intuition.** Entry at `T` is not available: the tape price at `T` predates the number.
# So each release gets a **ladder of five back-to-back 30-minute windows**, entering at
# `T+1`, `T+31`, `T+61`, `T+91` and `T+121`. The windows don't overlap, so no minute is
# traded twice.
#
# **Calculation.** For each anchor, the target is `log(P(entry+30)/P(entry))`. The path is
# the **price level** over the 31 minutes *ending exactly at entry*, z-scored across the
# window, so DTW matches the shape of the trajectory rather than its size. A game needs a
# bar at every path minute, at entry, at exit, at the print `T` and at `T+1`; since cleaning
# fills minutes inside a session, that means the whole window sits inside one session.
# A game is also dropped if the contract changes between the start of its path and its
# exit — ZN rolls at 19:00-20:00 ET, which an evening window could otherwise straddle.


# %%
def build_games(px: Px, contract: Px, fam_stamps: dict) -> pd.DataFrame:
    steps = np.arange(-PATH_PRE_MIN, LEAK_TEST_MINUTES + 1)
    recs = []
    for f, ts in fam_stamps.items():
        T = pd.DatetimeIndex(sorted(ts))
        has_print = np.isfinite(px.at(T)) & np.isfinite(px.at(T + pd.Timedelta(minutes=ENTRY_LAG_MIN)))
        for a in ANCHORS:
            entry = T + pd.Timedelta(minutes=a)
            grid = entry.values[:, None] + (steps * 60_000_000_000).astype("timedelta64[ns]")
            path = px.at(pd.DatetimeIndex(grid.ravel()).tz_localize("UTC")).reshape(len(T), len(steps))
            p0 = px.at(entry)
            p1 = px.at(entry + pd.Timedelta(minutes=HOLD_MIN))
            same_contract = (contract.at(entry - pd.Timedelta(minutes=PATH_PRE_MIN))
                             == contract.at(entry + pd.Timedelta(minutes=HOLD_MIN)))
            ok = (np.isfinite(path).all(1) & np.isfinite(p0) & np.isfinite(p1) & has_print
                  & same_contract)
            sd = path.std(1, keepdims=True)
            sd[sd < 1e-12] = 1.0
            z = (path - path.mean(1, keepdims=True)) / sd
            for i in np.where(ok)[0]:
                recs.append(dict(
                    family=f, event_ts=T[i], anchor=a,
                    clock_et=T[i].tz_convert(NY).strftime("%H:%M"),
                    entry_ts=entry[i],
                    exit_ts=entry[i] + pd.Timedelta(minutes=HOLD_MIN),
                    year=int(T[i].year),
                    fwd_ret=float(np.log(p1[i] / p0[i])),
                    path=z[i]))
    return pd.DataFrame(recs).sort_values("entry_ts").reset_index(drop=True)


games = {s: build_games(pxs[s], contracts[s], fam_stamps) for s in symbols}
for s, g in games.items():
    print(f"{s}: {len(g):,} games over {g.family.nunique()} families, {g.year.min()}-{g.year.max()}")

# %% [markdown]
# ## 3. The universe screen — `admitted`
#
# **Intuition.** Most scheduled statistics are ignored by the tape, and a release the
# market doesn't react to can't have a post-reaction drift.
#
# **Calculation.** Mean `|30-minute move|` over the family's release windows, divided by
# the same statistic **at the same clock slot, on days when some *other* family releases
# there but this one does not**. Measured against quiet days instead, every 08:30 release
# looks impactful simply because 08:30 is volatile. A family is admitted when the Welch t
# on log moves is ≥ 2. Recomputed each year from strictly prior years, on each asset's own
# moves.
#
# *Known flaw:* at a sparse clock slot the control is another big event. FOMC Rate Decision
# at 14:00 is benchmarked against FOMC Minutes, the ratio collapses toward 1, and it fails
# admission.


# %%
def impact_by_year(px: Px, fam_stamps: dict, years: list[int]) -> pd.DataFrame:
    by_clock: dict[str, set] = {}
    for f, ts in fam_stamps.items():
        for t in ts:
            by_clock.setdefault(t.tz_convert(NY).strftime("%H:%M"), set()).add(t)

    def moves(stamps, before: int) -> np.ndarray:
        s = pd.DatetimeIndex(sorted(t for t in stamps if t.year < before))
        if len(s) == 0:
            return np.array([])
        a, b = px.at(s), px.at(s + pd.Timedelta(minutes=IMPACT_WINDOW_MIN))
        v = np.abs(np.log(b / a)) * 1e4
        return v[np.isfinite(v) & (v > 0)]

    rows = []
    for year in years:
        for f, ts in fam_stamps.items():
            own = moves(set(ts), year)
            if len(own) < MIN_RELEASES:
                continue
            clock = pd.DatetimeIndex(sorted(ts))[0].tz_convert(NY).strftime("%H:%M")
            ctrl = moves(by_clock[clock] - set(ts), year)
            if len(ctrl) < 20:
                continue
            t = stats.ttest_ind(np.log(own), np.log(ctrl), equal_var=False).statistic
            rows.append(dict(year=year, family=f, impact=own.mean() / ctrl.mean(),
                             impact_t=float(t)))
    return pd.DataFrame(rows)


for s in symbols:
    years = sorted(games[s].year.unique())
    games[s] = games[s].merge(impact_by_year(pxs[s], fam_stamps, years),
                              on=["year", "family"], how="left")
    games[s]["admitted"] = (games[s].impact_t >= IMPACT_MIN_T).fillna(False)
    adm = games[s][games[s].admitted]
    print(f"{s}: {adm.family.nunique()} families admitted in some year, "
          f"{len(adm):,} admitted games")

# %% [markdown]
# **Findings:** On ES, at t ≥ 2 the screen admits **8 families**: NFP, CPI, ISM
# Manufacturing, ISM Services, FOMC Minutes, Retail Sales, Core PCE QoQ and Wholesale
# Inventories, from 2013. FOMC Rate Decision is wrongly excluded for the sparse-slot reason
# above. ZN admits **12 families** (5,420 games, against 3,420 on ES and 2,735 on NQ): the
# same core plus Durable Goods, Philly Fed, Business Inventories, the services PMI and Dallas
# Fed Services.

# %% [markdown]
# ## 4. The DTW features — `dtw_dir` and `dtw_disp`
#
# **Intuition.** The shape of the market's reaction to a print is the tape's own reading of
# the news. If past releases whose reaction looked like today's went on to drift a
# particular way, that is a forecast. If those past releases disagreed with each other,
# the forecast is less reliable.
#
# **Calculation.** DTW distance (Sakoe-Chiba band, `DTW_WARP_MINUTES`) between today's
# z-scored path and every earlier game on the **same asset, same clock slot and same
# anchor**, from any family. Candidates are those inside the lookback (optionally capped at
# the `DTW_MAX_POOL` most recent). The `k` nearest are weighted by `DTW_WEIGHTS` (uniform, or
# `1/distance`) × `exp(−age / DTW_HALFLIFE_DAYS)`. The book's setting is the original ES
# book's: band 5, 3-year lookback, cap 600, 2-year half-life, k = 15, 1/distance weights:
#
# | | |
# |---|---|
# | `dtw_dir` | weighted mean of the neighbours' forward returns — **the direction signal** |
# | `dtw_disp` | weighted sd of the neighbours' forward returns — **sizes the trade** |
#
# The factor is computed only for admitted games, but every game is in the neighbour pool,
# so non-admitted releases still contribute their forward returns. Both features are built
# for every (`k`, lookback) in `DTW_GRID` from one set of DTW distances; section 5 picks one.


# %%
def dtw_block(games: pd.DataFrame, query: np.ndarray) -> dict[tuple, pd.DataFrame]:
    """`dtw_dir` and `dtw_disp` for every (k, lookback_years) in DTW_GRID."""
    win = max(1, round(DTW_WARP_MINUTES / PATH_FREQ_MIN))
    settings = [(k, lb) for k in DTW_GRID["k"] for lb in DTW_GRID["lookback_years"]]
    out = {st: {c: np.full(len(games), np.nan) for c in ("dtw_dir", "dtw_disp")} for st in settings}
    for _, grp in games.groupby(["clock_et", "anchor"], sort=False):
        idx = grp.index.to_numpy()
        qpos = np.where(query[idx])[0]
        if len(qpos) == 0:
            continue
        t0 = grp.entry_ts.values.astype("datetime64[ns]").astype(np.int64)
        rets = grp.fwd_ret.to_numpy(float)
        paths = [np.asarray(p, dtype=np.double) for p in grp.path]
        # every earlier game against the queried columns, computed in C
        hi = int(qpos.max()) + 1
        D = dtw.distance_matrix_fast(paths[:hi], window=win, use_pruning=True,
                                     block=((0, hi), (int(qpos.min()), hi)), compact=False,
                                     parallel=True)
        for pos in qpos:
            age = (t0[pos] - t0[:pos]) / 86_400_000_000_000.0
            for lb in DTW_GRID["lookback_years"]:
                el = np.where((age > 0) & (age <= 365 * lb))[0]
                if len(el) < MIN_NEIGHBOURS:
                    continue
                if DTW_MAX_POOL:
                    el = el[-DTW_MAX_POOL:]
                d = D[pos, el]
                m = np.isfinite(d)
                el, d = el[m], d[m]
                if len(el) < MIN_NEIGHBOURS:
                    continue
                order = np.argsort(d)
                for k in DTW_GRID["k"]:
                    sel, dsel = el[order[:k]], d[order[:k]]
                    w = 1.0 / (dsel + 1e-6) if DTW_WEIGHTS == "inv_dist" else np.ones(len(sel))
                    w = w * np.exp(-age[sel] / DTW_HALFLIFE_DAYS)
                    w = w / w.sum()
                    r = rets[sel]
                    f = float(np.dot(w, r))
                    out[k, lb]["dtw_dir"][idx[pos]] = f
                    out[k, lb]["dtw_disp"][idx[pos]] = float(np.sqrt(np.dot(w, (r - f) ** 2)))
    return {st: pd.DataFrame(v, index=games.index) for st, v in out.items()}


feats = {}
for s in symbols:
    feats[s] = dtw_block(games[s], games[s].admitted.to_numpy())
    games[s] = games[s].drop(columns=["path"])
    n = {st: int(f.dtw_dir.notna().sum()) for st, f in feats[s].items()}
    print(f"{s}: factor computed for {min(n.values()):,}-{max(n.values()):,} games across settings")

# %% [markdown]
# ## 5. Scoring the DTW setting — training years
#
# `DTW_GRID` holds one setting, the original ES book's (k = 15, 3-year lookback); with more
# points in the grid this section picks the best on train.
#
# **Score.** For each setting, on admitted games in the training years (2013-2017):
#
# * `dir IC` — Spearman rank correlation of `dtw_dir` with the forward return. Does the
#   direction signal rank what happened next?
# * `disp IC` — Spearman rank correlation of `dtw_disp` with the *absolute* forward return.
#   Does the dispersion rank how big the move was?
#
# **Choice.** The setting with the highest `dir IC` averaged across the assets, one shared
# setting for all of them. `disp IC` is reported alongside but does not drive the choice —
# sizing is tuned later, in the backtest.
#
# Scored per asset as well as pooled: a setting that wins on the average but is weak on one
# asset is a warning, not a result.


# %%
def rank_ic(x: pd.Series, y: pd.Series) -> float:
    m = x.notna() & y.notna()
    return float(stats.spearmanr(x[m], y[m]).correlation) if m.sum() > 10 else np.nan


def score(span: tuple[int, int]) -> pd.DataFrame:
    rows = []
    for s in symbols:
        g = games[s]
        in_span = g.admitted & g.year.between(*span)
        for (k, lb), f in feats[s].items():
            m = in_span & f.dtw_dir.notna()
            rows.append({"asset": s, "k": k, "lookback_years": lb, "games": int(m.sum()),
                         "dir IC": rank_ic(f.dtw_dir[m], g.fwd_ret[m]),
                         "disp IC": rank_ic(f.dtw_disp[m], g.fwd_ret[m].abs())})
    return pd.DataFrame(rows)


train = score(TRAIN_YEARS_SPAN)
for col in ["dir IC", "disp IC"]:
    t = train.pivot_table(index=["k", "lookback_years"], columns="asset", values=col)[symbols]
    t["pooled"] = t.mean(axis=1)
    print(f"{col}, train {TRAIN_YEARS_SPAN[0]}-{TRAIN_YEARS_SPAN[1]}:")
    print(t.round(4).to_string(), "\n")
    if col == "dir IC":
        best_k, best_lb = t.pooled.idxmax()
print(f"games scored per asset: {train.groupby('asset').games.max().to_dict()}")
print(f"chosen: k = {best_k}, lookback = {best_lb} years")

# %% [markdown]
# **Findings:** `dtw_dir` ranks the next return negatively on the training years: ES −0.060
# (820 games), NQ −0.042 (525); ZN is flat at −0.003 (1,085). `dtw_disp` is strong: ES +0.137,
# NQ +0.242, ZN +0.124. The long-history
# setting in `experiments/tune_dtw.py` was positive here (ES +0.059) but did worse on
# 2021-2026, so the original setting was kept.

# %% [markdown]
# ### The chosen setting, year by year within training
#
# A setting that ranks well only because of one year is not stable.

# %%
rows = []
for s in symbols:
    g, f = games[s], feats[s][best_k, best_lb]
    for y in range(TRAIN_YEARS_SPAN[0], TRAIN_YEARS_SPAN[1] + 1):
        m = g.admitted & (g.year == y) & f.dtw_dir.notna()
        rows.append({"asset": s, "year": y, "games": int(m.sum()),
                     "dir IC": rank_ic(f.dtw_dir[m], g.fwd_ret[m]),
                     "disp IC": rank_ic(f.dtw_disp[m], g.fwd_ret[m].abs())})
by_year = pd.DataFrame(rows)
print(by_year.pivot(index="year", columns="asset", values="dir IC")[symbols].round(4).to_string())
print()
print(by_year.pivot(index="year", columns="asset", values="disp IC")[symbols].round(4).to_string())

# %% [markdown]
# **Findings:** dir IC flips sign year to year (ES +0.084, −0.104, +0.015, −0.151, −0.005 over
# 2013-2017; NQ +0.068, −0.011, −0.054, −0.135 over 2014-2017; ZN −0.078, +0.073, +0.012, −0.003
# over 2014-2017), so there is no stable direction in training. disp IC is positive in every
# year on all three (+0.05 to +0.25).

# %% [markdown]
# ### Validation check — the chosen setting only
#
# The chosen setting scored on 2018-2020, per asset. Only this one setting is shown, so the
# validation years are not used to choose between settings.

# %%
valid = score(VALID_YEARS_SPAN)
valid = valid[(valid.k == best_k) & (valid.lookback_years == best_lb)].set_index("asset")
chosen_train = train[(train.k == best_k) & (train.lookback_years == best_lb)].set_index("asset")
print(pd.DataFrame({"train dir IC": chosen_train["dir IC"], "valid dir IC": valid["dir IC"],
                    "train disp IC": chosen_train["disp IC"], "valid disp IC": valid["disp IC"],
                    "valid games": valid.games}).loc[symbols].round(4).to_string())

# %% [markdown]
# **Findings:** On validation dir IC is positive on ES and NQ, the opposite sign to training:
# ES +0.069 (840 games), NQ +0.091 (665). ZN stays at zero (−0.005, 1,130 games). disp IC holds
# up: ES +0.162, NQ +0.102, ZN +0.128.

# %% [markdown]
# ## 6. Validate and write
#
# The panel carries the chosen setting's `dtw_dir` and `dtw_disp`. The full training score
# table is written to `output/dtw_tuning.csv`.

# %%
panel = pd.concat([pd.concat([g, feats[s][best_k, best_lb]], axis=1).assign(symbol=s)
                   for s, g in games.items()], ignore_index=True)
panel = panel[["symbol"] + [c for c in panel.columns if c != "symbol"]]
assert (panel.entry_ts == panel.event_ts + pd.to_timedelta(panel.anchor, "m")).all()
assert (panel.exit_ts == panel.entry_ts + pd.Timedelta(minutes=HOLD_MIN)).all()
assert panel.fwd_ret.notna().all()
if LEAK_TEST_MINUTES:
    print(f"*** LEAK RUN: path runs {LEAK_TEST_MINUTES} min PAST entry on purpose. "
          f"Diagnostic only — not a tradeable panel. ***")
print("leak checks: path ends at entry, target starts at entry — OK")

dest = PANEL_LEAKED if LEAK_TEST_MINUTES else PANEL
panel.to_parquet(dest)
train.assign(chosen=(train.k == best_k) & (train.lookback_years == best_lb)).to_csv(
    OUT / "dtw_tuning.csv", index=False)
print(f"wrote {dest.name}  {len(panel):,} rows")
print(panel.groupby("symbol")[["dtw_dir", "dtw_disp"]].describe().T.round(5).to_string())
