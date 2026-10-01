"""Independent checks on teammates' headline numbers.

Two questions, both of which decide whether a reported result is tradeable:

1.  Hoshea's DTW event factor (`hoshea_zeng/outputs/dtw_simple_generalized_*.parquet`):
    the aggregate IC of 0.116 is reported over a pooled panel. Where does it live,
    does it survive clustering by release date, and does it clear costs?

2.  Aaryen's event-surprise backtest (`aaryen/code/`): entry price is the open of the
    1-minute bar at the release timestamp (`ENTRY_LAG_MINUTES = 0`). Sweeping the entry
    lag on a plain sign rule over the same releases separates the announcement jump,
    which is not executable, from the drift that follows it, which is.

Run: uv run python jack/verify_teammate_claims.py
Requires origin/main fetched (for Hoshea's parquets) and data/raw + data/processed local.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ES_FILE = ROOT / "data" / "processed" / "es_1min_clean.parquet"
BLOOMBERG_FILE = ROOT / "data" / "raw" / "Bloomberg Economic Releases.xlsx"
WINSOR_SIGMA = 5.0
TICK_BPS = 0.5  # one ES tick as a fraction of notional, matching jack/backtest.py


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def from_git(ref_path: str) -> Path:
    """Materialise a file from git without touching the working tree."""
    out = Path(tempfile.mkdtemp()) / Path(ref_path).name
    with out.open("wb") as fh:
        subprocess.run(["git", "show", ref_path], cwd=ROOT, stdout=fh, check=True)
    return out


def clustered_mean_t(df: pd.DataFrame, col: str, cluster: str) -> tuple[float, float, int]:
    """Mean of `col` with a one-way cluster-robust t-statistic.

    The panel stacks five overlapping anchors per release, so the naive t treats
    five correlated draws as five independent ones.
    """
    mean = df[col].mean()
    n = len(df)
    resid_sums = df.groupby(cluster)[col].apply(lambda s: (s - mean).sum())
    se = np.sqrt((resid_sums ** 2).sum()) / n
    return mean, mean / se, df[cluster].nunique()


def load_es_opens() -> tuple[pd.DatetimeIndex, np.ndarray]:
    es = pd.read_parquet(ES_FILE, columns=["open"])
    es = es[~es.index.duplicated(keep="first")].sort_index()
    return es.index, es["open"].to_numpy()


def make_bar_lookup(idx: pd.DatetimeIndex, opens: np.ndarray, tol=pd.Timedelta(minutes=5)):
    """Aaryen's `lookup_open`: nearest bar at/after ts, tolerance-guarded."""

    def lookup(ts: pd.Timestamp) -> float:
        pos = idx.searchsorted(ts)
        if pos >= len(idx) or abs(idx[pos] - ts) > tol:
            return np.nan
        return opens[pos]

    return lookup


def load_bloomberg() -> pd.DataFrame:
    sheets = pd.ExcelFile(BLOOMBERG_FILE)
    frames = []
    for sheet in sheets.sheet_names:
        part = pd.read_excel(BLOOMBERG_FILE, sheet_name=sheet)
        part.columns = [c.split(".")[-1] if "DROPNA" in str(c) else c for c in part.columns]
        frames.append(part)
    bloom = pd.concat(frames, ignore_index=True)
    bloom["RELEASE_DATE"] = pd.to_datetime(bloom.RELEASE_DATE, errors="coerce")
    bloom["RELEASE_TIME"] = bloom.RELEASE_TIME.astype(str)
    bloom = bloom.dropna(subset=["RELEASE_DATE"])
    return bloom[bloom.SURVEY_MEDIAN.notna() & bloom.ACTUAL.notna()]


def surprise_series(bloom: pd.DataFrame, event_name: str, release_time: str) -> pd.DataFrame:
    """Causally z-scored, winsorised surprise — same construction as macro_surprise.py."""
    g = bloom[(bloom.EVENT_NAME == event_name) & (bloom.RELEASE_TIME == release_time)]
    g = g.sort_values("RELEASE_DATE").copy()
    g["surprise"] = g.ACTUAL - g.SURVEY_MEDIAN
    scale = g.surprise.expanding().std().shift(1)  # prior releases only
    g["z"] = (g.surprise / scale).clip(-WINSOR_SIGMA, WINSOR_SIGMA)
    g = g[np.isfinite(g.z) & (scale > 0) & (g.z != 0)]
    stamp = pd.to_datetime(g.RELEASE_DATE.dt.strftime("%Y-%m-%d") + " " + release_time)
    g["ts"] = (stamp.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
               .dt.tz_convert("UTC"))
    return g.dropna(subset=["ts"])[["ts", "z"]]


# --------------------------------------------------------------------------- #
# 1. Hoshea's DTW event factor
# --------------------------------------------------------------------------- #
def check_dtw_factor() -> None:
    print("=" * 78)
    print("1. Hoshea — DTW event factor, where the IC actually lives")
    print("=" * 78)

    best = pd.read_parquet(from_git("origin/main:hoshea_zeng/outputs/dtw_simple_generalized_best_factor.parquet"))
    best["year"] = best.t0_utc.dt.year
    best["pnl_bps"] = np.sign(best.dtw_factor) * best.future_logret * 1e4

    ic = best.dtw_factor.corr(best.future_logret)
    hit = (np.sign(best.dtw_factor) == np.sign(best.future_logret)).mean()
    print(f"\npanel: {len(best):,} games, {best.event_timestamp_utc.nunique():,} releases, "
          f"{best.year.min()}-{best.year.max()}")
    print(f"reported aggregate: Pearson IC {ic:+.4f}, hit {hit:.2%}  (README: +0.1162, 51.51%)")

    print("\nby event type:")
    print(f"  {'event':<24} {'n':>5} {'pearson':>9} {'hit':>7} {'bps/game':>9} {'t_clust':>8}")
    for event, g in best.groupby("event_type"):
        mean, t, _ = clustered_mean_t(g, "pnl_bps", "event_timestamp_utc")
        h = (np.sign(g.dtw_factor) == np.sign(g.future_logret)).mean()
        print(f"  {event:<24} {len(g):>5} {g.dtw_factor.corr(g.future_logret):>+9.4f} "
              f"{h:>6.1%} {mean:>+9.3f} {t:>+8.2f}")
    missing = {"FOMC", "JOLTS", "PCE", "Initial Jobless Claims"} - set(best.event_type)
    if missing:
        print(f"  (absent from the scored panel despite being in the event group: {sorted(missing)})")

    print("\nquintiles of dtw_factor, mean forward return in bps:")
    best["q"] = pd.qcut(best.dtw_factor, 5, labels=False)
    print("  " + "  ".join(f"Q{q+1} {v:+.2f}" for q, v in (1e4 * best.groupby("q").future_logret.mean()).items()))

    print("\nquintiles of |dtw_factor|, mean |forward return| in bps  (magnitude, not direction):")
    best["aq"] = pd.qcut(best.dtw_factor.abs(), 5, labels=False)
    abs_q = 1e4 * best.groupby("aq").future_logret.apply(lambda s: s.abs().mean())
    print("  " + "  ".join(f"Q{q+1} {v:.2f}" for q, v in abs_q.items()))

    print("\nPCE only, robustness (sign trade, clustered by release date):")
    pce = best[best.event_type == "PCE"]
    splits = [("all", pce), ("excl 2020", pce[pce.year != 2020]),
              ("excl 2020 & 2026", pce[~pce.year.isin([2020, 2026])]),
              ("2013-2019", pce[pce.year <= 2019]), ("2020-2026", pce[pce.year >= 2020])]
    for label, sub in splits:
        mean, t, dates = clustered_mean_t(sub, "pnl_bps", "event_timestamp_utc")
        print(f"  {label:<18} n={len(sub):>4} dates={dates:>4} gross {mean:>+6.2f} bps  "
              f"t={t:>+5.2f}  net@1 tick {mean - TICK_BPS:>+6.2f}")

    print("\nextreme-quintile sign trade (long Q5 / short Q1), all events:")
    ext = best[best.q.isin([0, 4])].copy()
    ext["pnl_bps"] = np.where(ext.q == 4, 1, -1) * ext.future_logret * 1e4
    mean, t, dates = clustered_mean_t(ext, "pnl_bps", "event_timestamp_utc")
    years = (best.t0_utc.max() - best.t0_utc.min()).days / 365.25
    print(f"  n={len(ext)} ({len(ext)/years:.0f}/yr) gross {mean:+.3f} bps  t_clust={t:+.2f}  "
          f"net@1 tick {mean - TICK_BPS:+.3f}")

    print("\nthe all-30-minute-anchor extension (same factor, away from event windows):")
    all30 = pd.read_parquet(from_git("origin/main:hoshea_zeng/outputs/dtw_simple_generalized_all30_event_factor.parquet"))
    v = all30.dropna(subset=["dtw_factor_event_pool"])
    pnl = np.sign(v.dtw_factor_event_pool) * v.future_logret * 1e4
    h = (np.sign(v.dtw_factor_event_pool) == np.sign(v.future_logret)).mean()
    print(f"  n={len(v):,}  Pearson IC {v.dtw_factor_event_pool.corr(v.future_logret):+.4f}  "
          f"hit {h:.2%}  sign trade {pnl.mean():+.3f} bps  t={pnl.mean()/(pnl.std()/np.sqrt(len(pnl))):+.2f}")


# --------------------------------------------------------------------------- #
# 2. Entry-lag sweep
# --------------------------------------------------------------------------- #
ENTRY_LAG_SPEC = [
    # (Bloomberg event, release time, hypothesised sign of the trade vs the surprise)
    ("CPI MoM", "08:30:00", -1),
    ("Change in Nonfarm Payrolls", "08:30:00", +1),
    ("Core PCE Price Index MoM", "08:30:00", -1),
    ("GDP Annualized QoQ", "08:30:00", +1),
]


def check_entry_lag() -> None:
    print("\n" + "=" * 78)
    print("2. Aaryen — how much of the event P&L survives a one-minute entry lag")
    print("=" * 78)
    print("\nAaryen's construction, with the model replaced by sign(surprise):")
    print("  entry = open of the bar at T + lag,  exit = open of the bar at T + lag + window.")
    print("  lag=0 transacts at the pre-release price using post-release information.\n")

    idx, opens = load_es_opens()
    lookup = make_bar_lookup(idx, opens)
    bloom = load_bloomberg()
    start = pd.Timestamp("2013-04-01", tz="UTC")  # Aaryen's sample cutoff

    for event_name, release_time, sign in ENTRY_LAG_SPEC:
        events = surprise_series(bloom, event_name, release_time)
        events = events[events.ts >= start]
        print(f"{event_name}  (trade sign {sign:+d} × sign of surprise), {len(events)} releases")
        print(f"  {'window':>7} {'lag':>4} {'n':>4} {'bps/trade':>10} {'hit':>7} {'t':>7}")
        for window in (10, 30):
            for lag in (0, 1, 2, 5):
                pnl = []
                for row in events.itertuples(index=False):
                    entry = lookup(row.ts + pd.Timedelta(minutes=lag))
                    exit_ = lookup(row.ts + pd.Timedelta(minutes=lag + window))
                    if np.isnan(entry) or np.isnan(exit_) or entry <= 0:
                        continue
                    pnl.append(sign * np.sign(row.z) * 1e4 * np.log(exit_ / entry))
                pnl = np.asarray(pnl)
                if len(pnl) < 10:
                    continue
                t = pnl.mean() / (pnl.std() / np.sqrt(len(pnl)))
                print(f"  {window:>6}m {lag:>4} {len(pnl):>4} {pnl.mean():>10.2f} "
                      f"{(pnl > 0).mean():>6.1%} {t:>+7.2f}")
        print()


if __name__ == "__main__":
    check_dtw_factor()
    check_entry_lag()
