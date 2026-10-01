"""The DTW event book — impact-ranked universe, 2016-2021 train / 2022-2026 test.

Design rules, all of which exist to keep the test period untouched:

  UNIVERSE   Every US Bloomberg release with a published date *and* time is a candidate.
             Release lines sharing >=80% of their timestamps are collapsed into one
             event family (CPI MoM, Core CPI MoM, CPI Index NSA ... are one print).
             Families are ranked by MARKET IMPACT measured on 2016-2021 only:
                 impact = mean |30-min move| over the release window
                          / mean |30-min move| in the same clock slot on non-release days
             The clock-slot denominator matters -- on raw numbers a 10:00 release looks
             big because 10:00 is busy. Top K families are admitted. No P&L is consulted.

  DIRECTION  Two signals, both standardised per family on training statistics only:
               dtw_z    the clock-pool DTW path factor (below)
               news_z   the causally standardised Bloomberg surprise, winsorised at 5s,
                        direction-aligned by the SIGN of the training-period jump
                        regression rather than by hand-assigned economics
             Blended as W_DTW*dtw_z + (1-W_DTW)*news_z.

  DTW        For an event at T, anchors T0 in {-30, 0, +30, +60}. Path is [T0, T0+60] of
             z-scored log returns at PATH_FREQ. Entry T0+60, exit T0+90 -- a 30-minute
             hold, and the path ends exactly where the position opens. Neighbours are
             every prior admitted release sharing the same CLOCK SLOT and offset, whatever
             the family. That pooling rule is the one change from Hoshea: his winning
             group beat his losing group because PCE at 08:30 was being matched against
             the dense Claims library, which is a clock-slot effect he restricted by hand.
             Offset -60 is built but never traded: its entry lands exactly on the release
             instant, which is the non-executable fill that invalidated Aaryen's backtest.

Run: uv run python jack/dtw_impact_book.py
"""
from __future__ import annotations
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
from dtaidistance import dtw
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PRICE = ROOT / "data/processed/es_1min_clean.parquet"
BLOOMBERG = ROOT / "data/raw/Bloomberg Economic Releases.xlsx"
FEATURES = ROOT / "jack/data/features.parquet"
OUT = ROOT / "jack/data"
NY = ZoneInfo("America/New_York")

POOL_START, TRAIN_START, SPLIT, END = 2014, 2016, 2022, 2027
TRADE_OFFSETS = [-30, 0, 30, 60]
ALL_OFFSETS = [-60] + TRADE_OFFSETS
PATH_MIN, HOLD_MIN = 60, 30
OVERLAP = 0.80
MIN_TRAIN_RELEASES = 30
TOP_K = 10
WARP_MINUTES = 15
RECENCY_HL_DAYS = 730
MIN_HISTORY = 20
WINSOR = 5.0
TICK_BPS = 0.5
GRID_FREQ, GRID_LOOKBACK, GRID_K = [3, 5], [2, 3], [15, 25]
W_DTW_GRID = [1.0, 0.75, 0.5, 0.25, 0.0]


# ---------------------------------------------------------------- price helpers
class Px:
    def __init__(self, s: pd.Series):
        self.ts = s.index.values.astype("datetime64[ns]").astype(np.int64)
        self.px = s.to_numpy(float)

    def at(self, stamps) -> np.ndarray:
        q = pd.DatetimeIndex(stamps).tz_convert("UTC").values.astype("datetime64[ns]").astype(np.int64)
        pos = np.searchsorted(self.ts, q)
        pc = np.clip(pos, 0, len(self.ts) - 1)
        out = np.full(len(q), np.nan)
        hit = (pos < len(self.ts)) & (self.ts[pc] == q)
        out[hit] = self.px[pc[hit]]
        return out


# ---------------------------------------------------------------- bloomberg
def load_lines() -> pd.DataFrame:
    b = pd.concat([pd.read_excel(BLOOMBERG, sheet_name=s) for s in pd.ExcelFile(BLOOMBERG).sheet_names],
                  ignore_index=True)
    b.columns = [c.split(".")[-1] if "DROPNA" in str(c) else c for c in b.columns]
    b = b[b.COUNTRY_NAME.astype(str).str.contains("United States", na=False)]
    b["RELEASE_DATE"] = pd.to_datetime(b.RELEASE_DATE, errors="coerce")
    b = b.dropna(subset=["RELEASE_DATE", "RELEASE_TIME", "EVENT_NAME"])
    t = pd.to_datetime(b.RELEASE_TIME.astype(str), format="%H:%M:%S", errors="coerce")
    b = b[t.notna()].copy()
    t = t[t.notna()]
    local = b.RELEASE_DATE.dt.normalize() + pd.to_timedelta(t.dt.hour, "h") + pd.to_timedelta(t.dt.minute, "m")
    b["ts"] = local.dt.tz_localize(NY, ambiguous=True, nonexistent="shift_forward").dt.tz_convert("UTC")
    b["clock_et"] = b.ts.dt.tz_convert(NY).dt.strftime("%H:%M")
    b = b[(b.ts.dt.year >= POOL_START) & (b.ts.dt.year < END)]
    return b[["ts", "clock_et", "EVENT_NAME", "SURVEY_MEDIAN", "ACTUAL"]].rename(columns={"EVENT_NAME": "line"})


def assign_families(lines: pd.DataFrame) -> dict:
    stamps = {ln: set(g.ts) for ln, g in lines.groupby("line") if len(g) >= 12}
    names = sorted(stamps, key=lambda n: -len(stamps[n]))
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for i, a in enumerate(names):
        for bl in names[i + 1:]:
            if find(a) == find(bl):
                continue
            shared = len(stamps[a] & stamps[bl])
            if shared and shared / min(len(stamps[a]), len(stamps[bl])) >= OVERLAP:
                parent[find(bl)] = find(a)
    groups = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    out = {}
    for root, members in groups.items():
        best = max(members, key=lambda m: len(stamps[m]))
        clock = lines[lines.line == best].clock_et.mode().iloc[0]
        for m in members:
            out[m] = f"{best} @ {clock}"
    return out


# ---------------------------------------------------------------- impact screen
def impact_screen(px: Px, fam_stamps: dict) -> pd.DataFrame:
    """Marginal market impact of each family, on TRAIN years only.

    The naive measure -- release-window move over the same clock slot on non-release days
    -- does not work. It scores CPI 1.88, PPI 1.83, Retail 1.76 and Trade Balance 1.75,
    all nearly identical, because it credits every 08:30 release with the fact that 08:30
    on a release day is a volatile minute. Minor releases ride along on the majors.

    The control here is like-for-like: the same clock slot, on days when *some other*
    family releases at that slot but this one does not. That asks the only question worth
    asking -- given a release is happening at 08:30, does it matter that it is this one?
    """
    def moves(stamps) -> np.ndarray:
        s = pd.DatetimeIndex(sorted(stamps))
        s = s[(s.year >= TRAIN_START) & (s.year < SPLIT)]
        if len(s) == 0:
            return np.array([])
        a, b = px.at(s), px.at(s + pd.Timedelta(minutes=30))
        v = np.abs(np.log(b / a)) * 1e4
        return v[np.isfinite(v)]

    by_clock: dict[str, set] = {}
    for f, ts in fam_stamps.items():
        for t in ts:
            by_clock.setdefault(t.tz_convert(NY).strftime("%H:%M"), set()).add(t)

    rows = []
    for fam, ts in fam_stamps.items():
        ts = set(ts)
        tr = [t for t in ts if TRAIN_START <= t.year < SPLIT]
        if len(tr) < MIN_TRAIN_RELEASES:
            continue
        clock = pd.DatetimeIndex(sorted(tr))[0].tz_convert(NY).strftime("%H:%M")
        own = moves(tr)
        ctrl = moves(by_clock[clock] - ts)          # same slot, this family absent
        if len(own) < MIN_TRAIN_RELEASES or len(ctrl) < 20:
            continue
        rows.append(dict(family=fam, clock=clock, n_train=len(tr), n_total=len(ts),
                         n_control=len(ctrl), abs_move_bps=round(float(own.mean()), 2),
                         control_bps=round(float(ctrl.mean()), 2),
                         impact=round(float(own.mean() / ctrl.mean()), 3),
                         t=round(float(stats.ttest_ind(np.log(own[own > 0]),
                                                       np.log(ctrl[ctrl > 0]),
                                                       equal_var=False).statistic), 2)))
    return pd.DataFrame(rows).sort_values("impact", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- games + dtw
def build_games(px: Px, admitted: pd.DataFrame, fam_stamps: dict, freq: int) -> pd.DataFrame:
    recs = []
    for fam in admitted.family:
        ts = pd.DatetimeIndex(sorted(fam_stamps[fam]))
        for off in ALL_OFFSETS:
            t0 = ts + pd.Timedelta(minutes=off)
            steps = np.arange(0, PATH_MIN + 1, freq)
            grid = t0.values[:, None] + (steps * 60_000_000_000).astype("timedelta64[ns]")
            flat = pd.DatetimeIndex(grid.ravel()).tz_localize("UTC")
            pth = px.at(flat).reshape(len(t0), len(steps))
            ent = px.at(t0 + pd.Timedelta(minutes=PATH_MIN))
            ext = px.at(t0 + pd.Timedelta(minutes=PATH_MIN + HOLD_MIN))
            ok = np.isfinite(pth).all(1) & np.isfinite(ent) & np.isfinite(ext)
            lr = np.diff(np.log(pth), axis=1)
            sd = lr.std(1); mu = lr.mean(1)
            z = np.where(sd[:, None] < 1e-12, 0.0, (lr - mu[:, None]) / np.where(sd[:, None] < 1e-12, 1, sd[:, None]))
            for i in np.where(ok)[0]:
                recs.append(dict(family=fam, event_ts=ts[i], offset_min=off, t0_utc=t0[i],
                                 entry_utc=t0[i] + pd.Timedelta(minutes=PATH_MIN),
                                 clock_et=t0[i].tz_convert(NY).strftime("%H:%M"),
                                 fwd=float(np.log(ext[i] / ent[i])), path=z[i]))
    return pd.DataFrame(recs).sort_values("t0_utc").reset_index(drop=True)


def dtw_factors(games: pd.DataFrame, freq: int) -> dict:
    """Two pooling rules off one distance matrix.

    clock  -- neighbours are any admitted release at the same clock slot and offset.
    family -- neighbours are restricted to the same event family (Hoshea's Experiment B).

    The distance matrix is identical for both; only the eligible set differs, so the
    comparison isolates the pooling rule and nothing else.
    """
    win = max(1, round(WARP_MINUTES / freq))
    res = {(pool, lb, k): np.full(len(games), np.nan)
           for pool in ("clock", "family") for lb in GRID_LOOKBACK for k in GRID_K}
    for _, grp in games.groupby(["clock_et", "offset_min"], sort=False):
        idx = grp.index.to_numpy()
        if len(idx) < MIN_HISTORY + 1:
            continue
        D = np.asarray(dtw.distance_matrix_fast(
            [np.asarray(p, dtype=np.double) for p in grp["path"]], window=win, compact=False))
        D[np.isinf(D)] = np.nan
        D = np.where(np.isnan(D), D.T, D)
        t0 = grp["t0_utc"].values.astype("datetime64[ns]").astype(np.int64)
        age_all = (t0[:, None] - t0[None, :]) / 86_400_000_000_000.0   # days, float
        rets = grp["fwd"].to_numpy(float)
        famcode = pd.factorize(grp["family"].to_numpy())[0]
        for pos in range(len(idx)):
            a = age_all[pos]
            finite = np.isfinite(D[pos])
            same_fam = famcode == famcode[pos]
            for pool in ("clock", "family"):
                base = (a > 0) & finite & (same_fam if pool == "family" else True)
                for lb in GRID_LOOKBACK:
                    el = np.where(base & (a <= 365 * lb))[0]
                    if len(el) < MIN_HISTORY:
                        continue
                    order = el[np.argsort(D[pos, el])]
                    for k in GRID_K:
                        sel = order[:min(k, len(order))]
                        w = (1.0 / (D[pos, sel] + 1e-6)) * np.exp(-a[sel] / RECENCY_HL_DAYS)
                        res[(pool, lb, k)][idx[pos]] = float(np.dot(w, rets[sel]) / w.sum())
    return res


# ---------------------------------------------------------------- news
def news_panel(lines: pd.DataFrame, fam: dict) -> pd.DataFrame:
    b = lines.dropna(subset=["SURVEY_MEDIAN", "ACTUAL"]).copy()
    b["family"] = b.line.map(fam)
    out = []
    for ln, g in b.groupby("line"):
        g = g.sort_values("ts")
        if len(g) < 40:
            continue
        s = g.ACTUAL - g.SURVEY_MEDIAN
        scale = s.expanding().std().shift(1)
        z = (s / scale).clip(-WINSOR, WINSOR)
        m = np.isfinite(z) & (scale > 0)
        out.append(pd.DataFrame({"ts": g.ts[m], "family": g.family[m], "z": z[m]}))
    if not out:
        return pd.DataFrame(columns=["ts", "family", "news_raw"])
    d = pd.concat(out)
    return d.groupby(["family", "ts"]).z.mean().rename("news_raw").reset_index()


def clustered_t(p, c):
    m = pd.DataFrame({"p": p, "c": c}).groupby("c").p.mean()
    return float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m)))) if len(m) > 2 else np.nan


def score(sub, sig):
    x = sub[sig].to_numpy(); y = sub.fwd.to_numpy() * 1e4
    g = np.sign(x) * y
    return dict(n=len(sub), ic=float(np.corrcoef(x, y)[0, 1]),
                hit=float((g > 0).mean()), gross=float(g.mean()),
                net=float(g.mean() - 2 * TICK_BPS),
                t=clustered_t(g - 2 * TICK_BPS, sub.event_ts.dt.date))


def main():
    close = pd.read_parquet(PRICE)["close"]
    close.index = pd.to_datetime(close.index, utc=True)
    close = close[~close.index.duplicated()].sort_index()
    px = Px(close)

    lines = load_lines()
    fam = assign_families(lines)
    lines["family"] = lines.line.map(fam)
    lines = lines.dropna(subset=["family"])
    fam_stamps = {f: sorted(set(g.ts)) for f, g in lines.groupby("family")}
    print(f"{lines.line.nunique()} release lines -> {len(fam_stamps)} families")

    screen = impact_screen(px, fam_stamps)
    screen.to_csv(OUT / "impact_screen.csv", index=False)
    print(f"\n===== IMPACT SCREEN (train 2016-2021 only), top 20 of {len(screen)} =====")
    print(screen.head(20).to_string(index=False))
    admitted = screen.head(TOP_K).reset_index(drop=True)
    print(f"\nADMITTED (top {TOP_K}):")
    print(admitted.to_string(index=False))

    news = news_panel(lines, fam)

    all_panels = []
    for freq in GRID_FREQ:
        g = build_games(px, admitted, fam_stamps, freq)
        print(f"\nfreq {freq}min: {len(g)} games across {g.family.nunique()} families", flush=True)
        res = dtw_factors(g, freq)
        base = g.drop(columns=["path"])
        for (pool, lb, k), v in res.items():
            p = base.copy(); p["dtw_factor"] = v
            p["freq"], p["lookback"], p["k"], p["pool"] = freq, lb, k, pool
            all_panels.append(p)
    panel = pd.concat(all_panels, ignore_index=True)
    panel["yr"] = panel.t0_utc.dt.year
    panel = panel[(panel.yr >= TRAIN_START) & panel.dtw_factor.notna()]
    panel = panel.merge(news.rename(columns={"ts": "event_ts"}), on=["family", "event_ts"], how="left")
    panel.to_parquet(OUT / "dtw_impact_panel.parquet")
    print(f"\npanel {len(panel)} rows; {panel.news_raw.notna().mean():.1%} carry a consensus")

    trade = panel[panel.offset_min.isin(TRADE_OFFSETS)]
    tr = trade[trade.yr < SPLIT]
    print("\n===== DTW HYPERPARAMETER GRID (selected on 2016-2021) =====")
    rows = []
    for (pool, f, lb, k), g in trade.groupby(["pool", "freq", "lookback", "k"]):
        s, c = score(g[g.yr < SPLIT], "dtw_factor"), score(g[g.yr >= SPLIT], "dtw_factor")
        rows.append(dict(pool=pool, freq=f, lookback=lb, k=k, sel_n=s["n"], sel_ic=s["ic"],
                         sel_gross=s["gross"], sel_t=s["t"], con_n=c["n"], con_ic=c["ic"],
                         con_hit=c["hit"], con_gross=c["gross"], con_net=c["net"], con_t=c["t"]))
    grid = pd.DataFrame(rows).sort_values("sel_ic", ascending=False)
    grid.to_csv(OUT / "dtw_impact_grid.csv", index=False)
    pd.set_option("display.width", 250)
    print(grid.round(4).to_string(index=False))
    if len(grid) > 2:
        print(f"rank corr select->confirm IC: {stats.spearmanr(grid.sel_ic, grid.con_ic).correlation:+.3f}")
    print("\nmean selection IC by pooling rule:")
    print(grid.groupby("pool")[["sel_ic", "con_ic", "con_gross", "con_t"]].mean().round(4).to_string())
    best = grid.iloc[0]
    print(f"\nBEST CELL: pool={best['pool']} freq={int(best.freq)} lookback={int(best.lookback)} k={int(best.k)}")
    json.dump({"pool": str(best["pool"]), **{kk: int(best[kk]) for kk in ["freq", "lookback", "k"]}},
              open(OUT / "dtw_impact_best.json", "w"))
    print("\nwrote impact_screen.csv / dtw_impact_panel.parquet / dtw_impact_grid.csv")


if __name__ == "__main__":
    main()
