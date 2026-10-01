"""Is there a *rule* that says which events the path signal works on?

Every strategy proposal so far has named the events that worked — CPI, PCE — which is
selection after the fact and generalises to nothing. The question that does generalise
is: given a release we have never traded, can we predict whether a directional signal
will work on it, from properties measurable *before* looking at any P&L?

Four candidate properties, all observable in advance:

  vol_multiple   how much larger the release-window move is than a typical move in the
                 same clock slot on non-release days. Jack's feature work found this
                 ranks CPI 4.7x, NFP 3.6x, FOMC 3.1x, PPI/Retail ~1.9x and nothing else
                 above 1.6x — so it is already known to separate events.
  pool_density   median number of prior same-event neighbours available at match time.
                 Hoshea's grid search implicitly optimised for this without saying so.
  n_releases     how much history exists at all.
  has_consensus  whether Bloomberg carries a survey median for the release, i.e. whether
                 there is a genuine surprise to be had rather than a derivable number.

The signal itself is Hoshea's DTW event-game factor, held fixed, run over all 13 event
types in the expanded calendar with strictly same-event matching (his Experiment B
design, which is the event-agnostic one). Per-event performance is then regressed on the
four properties, with a 2012-2021 / 2022-2026 split so the rule is fitted on one period
and checked on the other.

Run: uv run python jack/event_tradeability_rule.py
"""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dtaidistance import dtw

ROOT = Path(__file__).resolve().parents[1]
PRICE_FILE = ROOT / "data" / "processed" / "es_1min_bars_2010_2026.parquet"
EVENT_FILE = ROOT / "data" / "processed" / "macro_event_calendar_expanded_2010_2026.parquet"
BLOOMBERG_FILE = ROOT / "data" / "raw" / "Bloomberg Economic Releases.xlsx"
OUT = ROOT / "jack" / "data"

# Hoshea's factor, held fixed at the middle of his grid so nothing is tuned per event.
EVENT_OFFSETS_MIN = [-60, -30, 0, 30, 60]
PATH_MINUTES, PATH_FREQ_MIN = 60, 3
ENTRY_DELAY_MINUTES, HOLD_MINUTES = 60, 30
K_NEIGHBORS, MIN_HISTORY = 15, 12
LOOKBACK_YEARS = 3
RECENCY_HALFLIFE_DAYS = 365 * 2
DTW_WINDOW_STEPS = 5
SPLIT = pd.Timestamp("2022-01-01", tz="UTC")
TICK_BPS = 0.5
NY_TZ = ZoneInfo("America/New_York")

# Bloomberg event names that correspond to each calendar event type, for has_consensus.
CONSENSUS_NAMES = {
    "CPI": "CPI MoM", "NFP": "Change in Nonfarm Payrolls", "PCE": "Core PCE Price Index MoM",
    "GDP": "GDP Annualized QoQ", "PPI": "PPI Final Demand MoM", "Retail Sales": "Retail Sales Advance MoM",
    "Industrial Production": "Industrial Production MoM", "Initial Jobless Claims": "Initial Jobless Claims",
    "JOLTS": "JOLTS Job Openings", "Import/Export Prices": "Import Price Index MoM",
    "Wholesale Trade": "Wholesale Inventories MoM", "Employment Cost Index": "Employment Cost Index",
    "FOMC": "FOMC Rate Decision (Upper Bound)",
}


def zscore(x):
    sd = float(np.std(x))
    return np.zeros_like(x) if sd < 1e-12 else (x - float(np.mean(x))) / sd


def build_games(close: pd.Series, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in events.itertuples(index=False):
        for offset in EVENT_OFFSETS_MIN:
            t0 = row.timestamp_utc + pd.Timedelta(minutes=offset)
            idx = pd.date_range(t0, t0 + pd.Timedelta(minutes=PATH_MINUTES),
                                freq=f"{PATH_FREQ_MIN}min", tz="UTC")
            px = close.reindex(idx)
            if px.isna().any():
                continue
            logret = np.diff(np.log(px.to_numpy(dtype=float)))
            if len(logret) == 0 or not np.isfinite(logret).all():
                continue
            entry = t0 + pd.Timedelta(minutes=ENTRY_DELAY_MINUTES)
            lab = close.reindex([entry, entry + pd.Timedelta(minutes=HOLD_MINUTES)])
            if lab.isna().any():
                continue
            rows.append({"event_type": row.event_type, "event_ts": row.timestamp_utc,
                         "offset_min": offset, "t0_utc": t0,
                         "clock_et": t0.tz_convert(NY_TZ).strftime("%H:%M"),
                         "future_logret": float(np.log(lab.iloc[1]) - np.log(lab.iloc[0])),
                         "path": zscore(logret)})
    return pd.DataFrame(rows).sort_values("t0_utc").reset_index(drop=True)


def compute_factor(games: pd.DataFrame) -> pd.DataFrame:
    """Strictly same-event-type matching — the event-agnostic design."""
    out = games.copy()
    out["dtw_factor"] = np.nan
    out["n_neighbours"] = np.nan
    lookback = pd.Timedelta(days=365 * LOOKBACK_YEARS)
    for _, grp in out.groupby(["event_type", "offset_min", "clock_et"], sort=False):
        idx = grp.index.to_numpy()
        t0s = out.loc[idx, "t0_utc"].tolist()
        paths = out.loc[idx, "path"].tolist()
        rets = out.loc[idx, "future_logret"].to_numpy(dtype=float)
        for pos, row_idx in enumerate(idx):
            hist = [h for h in range(pos) if t0s[h] >= t0s[pos] - lookback]
            out.at[row_idx, "n_neighbours"] = len(hist)
            if len(hist) < MIN_HISTORY:
                continue
            x = np.asarray(paths[pos], dtype=np.double)
            d = np.array([dtw.distance_fast(x, np.asarray(paths[h], dtype=np.double),
                                            window=DTW_WINDOW_STEPS, use_pruning=True) for h in hist])
            k = min(K_NEIGHBORS, len(hist))
            nn = np.argpartition(d, k - 1)[:k]
            gaps = np.array([(t0s[pos] - t0s[hist[j]]).days for j in nn], dtype=float)
            w = (1.0 / (d[nn] + 1e-6)) * np.exp(-gaps / RECENCY_HALFLIFE_DAYS)
            out.at[row_idx, "dtw_factor"] = float(np.dot(w, rets[[hist[j] for j in nn]]) / w.sum())
    return out


def vol_multiples(close: pd.Series, events: pd.DataFrame) -> pd.Series:
    """Mean |30-min return| in the release window, over the same statistic for the same
    clock slot on days with no release of any kind. Measurable before any P&L."""
    ret30 = np.log(close).diff(30)
    abs30 = ret30.abs()
    clock = pd.Series(abs30.index.tz_convert(NY_TZ).strftime("%H:%M"), index=abs30.index)
    event_minutes = pd.DatetimeIndex(events.timestamp_utc.unique())
    is_event_day = pd.Series(abs30.index.normalize().isin(event_minutes.normalize()), index=abs30.index)
    baseline = abs30[~is_event_day].groupby(clock[~is_event_day]).mean()
    out = {}
    for event_type, g in events.groupby("event_type"):
        stamps = pd.DatetimeIndex(g.timestamp_utc) + pd.Timedelta(minutes=30)
        vals = abs30.reindex(stamps).dropna()
        if len(vals) < 20:
            continue
        slots = pd.Series(pd.DatetimeIndex(vals.index).tz_convert(NY_TZ).strftime("%H:%M"), index=vals.index)
        base = slots.map(baseline)
        ok = base.notna() & (base > 0)
        out[event_type] = float((vals[ok] / base[ok]).mean())
    return pd.Series(out, name="vol_multiple")


def has_consensus(events: pd.DataFrame) -> pd.Series:
    sheets = pd.ExcelFile(BLOOMBERG_FILE)
    frames = []
    for sheet in sheets.sheet_names:
        part = pd.read_excel(BLOOMBERG_FILE, sheet_name=sheet)
        part.columns = [c.split(".")[-1] if "DROPNA" in str(c) else c for c in part.columns]
        frames.append(part)
    b = pd.concat(frames, ignore_index=True)
    b = b[b.SURVEY_MEDIAN.notna() & b.ACTUAL.notna()]
    counts = b.EVENT_NAME.value_counts()
    out = {}
    for event_type in events.event_type.unique():
        name = CONSENSUS_NAMES.get(event_type)
        out[event_type] = int(counts.get(name, 0)) if name else 0
    return pd.Series(out, name="n_consensus")


def score(df: pd.DataFrame) -> dict:
    v = df.dropna(subset=["dtw_factor"])
    if len(v) < 60:
        return {"n": len(v), "ic": np.nan, "hit": np.nan, "bps": np.nan, "t": np.nan}
    pnl = np.sign(v.dtw_factor) * v.future_logret * 1e4
    m, n = pnl.mean(), len(pnl)
    resid = pd.DataFrame({"p": pnl.to_numpy(), "c": v.event_ts.to_numpy()}).groupby("c").p.apply(lambda s: (s - m).sum())
    return {"n": n, "ic": float(v.dtw_factor.corr(v.future_logret)),
            "hit": float((np.sign(v.dtw_factor) == np.sign(v.future_logret)).mean()),
            "bps": float(m), "t": float(m / (np.sqrt((resid ** 2).sum()) / n))}


def main() -> None:
    close = pd.read_parquet(PRICE_FILE, columns=["close"]).sort_index()
    close.index = close.index.tz_localize("UTC") if close.index.tz is None else close.index.tz_convert("UTC")
    close = close["close"].astype(float)
    events = pd.read_parquet(EVENT_FILE)
    events["timestamp_utc"] = pd.to_datetime(events.timestamp_utc, utc=True)
    events = events.sort_values("timestamp_utc").reset_index(drop=True)
    print(f"event types: {events.event_type.nunique()}, releases: {len(events):,}")

    games = build_games(close, events)
    print(f"games built: {len(games):,}")
    scored = compute_factor(games)
    scored.drop(columns=["path"]).to_parquet(OUT / "event_tradeability_panel.parquet", index=False)

    vm = vol_multiples(close, events)
    nc = has_consensus(events)

    rows = []
    for event_type, g in scored.groupby("event_type"):
        full, sel, con = score(g), score(g[g.t0_utc < SPLIT]), score(g[g.t0_utc >= SPLIT])
        rows.append({
            "event_type": event_type,
            "n_releases": g.event_ts.nunique(),
            "vol_multiple": vm.get(event_type, np.nan),
            "pool_density": float(g.n_neighbours.median()),
            "n_consensus": int(nc.get(event_type, 0)),
            "full_ic": full["ic"], "full_hit": full["hit"],
            "sel_ic": sel["ic"], "sel_bps": sel["bps"], "sel_t": sel["t"],
            "con_n": con["n"], "con_ic": con["ic"], "con_bps": con["bps"], "con_t": con["t"],
        })
    tab = pd.DataFrame(rows).sort_values("con_bps", ascending=False)
    tab.to_csv(OUT / "event_tradeability_rule.csv", index=False)

    print("\n" + "=" * 118)
    print("PER EVENT — same-event DTW factor, one fixed parameter set, no per-event tuning")
    print("=" * 118)
    print(tab.round(3).to_string(index=False))

    print("\n" + "=" * 118)
    print("DOES ANY OBSERVABLE PROPERTY PREDICT WHICH EVENTS WORK?")
    print("=" * 118)
    ok = tab.dropna(subset=["con_bps", "sel_bps", "vol_multiple"])
    print(f"  events with both periods scored: {len(ok)}\n")
    print(f"  {'property':<16} {'vs selection bps':>18} {'vs confirm bps':>17} {'vs confirm IC':>15}")
    for prop in ("vol_multiple", "pool_density", "n_releases", "n_consensus"):
        print(f"  {prop:<16} {ok[prop].corr(ok.sel_bps, method='spearman'):>18.3f} "
              f"{ok[prop].corr(ok.con_bps, method='spearman'):>17.3f} "
              f"{ok[prop].corr(ok.con_ic, method='spearman'):>15.3f}")

    print(f"\n  persistence — does an event that worked in 2012-2021 still work in 2022-2026?")
    print(f"    rank corr of selection bps vs confirmation bps across events: "
          f"{ok.sel_bps.corr(ok.con_bps, method='spearman'):+.3f}")
    print(f"    rank corr of selection IC  vs confirmation IC  across events: "
          f"{ok.sel_ic.corr(ok.con_ic, method='spearman'):+.3f}")

    print("\n  a rule fitted on 2012-2021 only: take every event with positive selection bps,")
    print("  then measure what that basket does in 2022-2026 —")
    picked = ok[ok.sel_bps > 0]
    rest = ok[ok.sel_bps <= 0]
    panel = scored[scored.t0_utc >= SPLIT]
    for label, sub in (("picked", picked), ("rejected", rest)):
        p = panel[panel.event_type.isin(sub.event_type)]
        s = score(p)
        print(f"    {label:<9} {len(sub):>2} events  n={s['n']:>5}  IC {s['ic']:>+7.4f}  "
              f"hit {s['hit']:>6.1%}  {s['bps']:>+6.2f} bps  clustered t {s['t']:>+5.2f}  "
              f"net {s['bps']-TICK_BPS:>+6.2f}")
    print(f"\nsaved -> {OUT/'event_tradeability_rule.csv'}")


if __name__ == "__main__":
    main()
