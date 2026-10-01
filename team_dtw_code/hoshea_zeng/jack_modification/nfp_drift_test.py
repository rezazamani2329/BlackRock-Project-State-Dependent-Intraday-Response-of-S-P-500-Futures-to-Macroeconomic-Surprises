"""Item 1b: does the NFP post-print drift survive a proper test?

The entry-lag sweep in strategy_analysis.md turned up one cell that *gained* from
lagging the entry: NFP, 10-minute window, lag 1, +3.04 bps at 61.0% hit, t = +2.12.
That was the best of 32 cells looked at (4 releases x 2 windows x 4 lags), so t = 2.12
is not evidence on its own. This tests it the way the CPI overlay was tested:

  * full exit-window profile, so the decay shape is visible rather than one lucky exit;
  * split at 2020, requiring the sign and magnitude to survive a regime break;
  * a |z| filter gradient, which a real effect produces monotonically;
  * a best-of-N null: how large a |t| does the best of 32 pure-noise features reach on
    the same returns?

Run: uv run python jack/nfp_drift_test.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ES_FILE = ROOT / "data" / "processed" / "es_1min_clean.parquet"
BLOOMBERG_FILE = ROOT / "data" / "raw" / "Bloomberg Economic Releases.xlsx"
WINSOR_SIGMA = 5.0
TICK_BPS = 0.5
RNG = np.random.default_rng(0)

es = pd.read_parquet(ES_FILE, columns=["open"])
es = es[~es.index.duplicated(keep="first")].sort_index()
ES_IDX, ES_OPEN = es.index, es["open"].to_numpy()


def lookup_open(ts: pd.Timestamp, tol=pd.Timedelta(minutes=5)) -> float:
    pos = ES_IDX.searchsorted(ts)
    if pos >= len(ES_IDX) or abs(ES_IDX[pos] - ts) > tol:
        return np.nan
    return ES_OPEN[pos]


def load_bloomberg() -> pd.DataFrame:
    sheets = pd.ExcelFile(BLOOMBERG_FILE)
    frames = []
    for sheet in sheets.sheet_names:
        part = pd.read_excel(BLOOMBERG_FILE, sheet_name=sheet)
        part.columns = [c.split(".")[-1] if "DROPNA" in str(c) else c for c in part.columns]
        frames.append(part)
    b = pd.concat(frames, ignore_index=True)
    b["RELEASE_DATE"] = pd.to_datetime(b.RELEASE_DATE, errors="coerce")
    b["RELEASE_TIME"] = b.RELEASE_TIME.astype(str)
    b = b.dropna(subset=["RELEASE_DATE"])
    return b[b.SURVEY_MEDIAN.notna() & b.ACTUAL.notna()]


def surprise_series(b: pd.DataFrame, name: str, time: str) -> pd.DataFrame:
    g = b[(b.EVENT_NAME == name) & (b.RELEASE_TIME == time)].sort_values("RELEASE_DATE").copy()
    g["surprise"] = g.ACTUAL - g.SURVEY_MEDIAN
    scale = g.surprise.expanding().std().shift(1)
    g["z"] = (g.surprise / scale).clip(-WINSOR_SIGMA, WINSOR_SIGMA)
    g = g[np.isfinite(g.z) & (scale > 0) & (g.z != 0)]
    stamp = pd.to_datetime(g.RELEASE_DATE.dt.strftime("%Y-%m-%d") + " " + time)
    g["ts"] = (stamp.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
               .dt.tz_convert("UTC"))
    return g.dropna(subset=["ts"])[["ts", "z"]]


def trades(events: pd.DataFrame, lag: int, window: int, sign: int) -> pd.DataFrame:
    rows = []
    for r in events.itertuples(index=False):
        entry = lookup_open(r.ts + pd.Timedelta(minutes=lag))
        exit_ = lookup_open(r.ts + pd.Timedelta(minutes=lag + window))
        if np.isnan(entry) or np.isnan(exit_) or entry <= 0:
            continue
        rows.append({"ts": r.ts, "z": r.z,
                     "ret_bps": 1e4 * np.log(exit_ / entry),
                     "pnl_bps": sign * np.sign(r.z) * 1e4 * np.log(exit_ / entry)})
    return pd.DataFrame(rows)


def tstat(x: np.ndarray) -> float:
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def main() -> None:
    b = load_bloomberg()
    nfp = surprise_series(b, "Change in Nonfarm Payrolls", "08:30:00")
    nfp = nfp[nfp.ts >= pd.Timestamp("2013-04-01", tz="UTC")]
    print(f"NFP releases with a usable surprise, 2013-04 onward: {len(nfp)}\n")

    print("=" * 74)
    print("1. Exit-window profile, entry at T+1 (long on a strong print)")
    print("=" * 74)
    print(f"  {'exit':>6} {'n':>4} {'gross bps':>10} {'net bps':>9} {'hit':>7} {'t':>7}")
    for window in (5, 10, 15, 20, 30, 45, 60):
        t = trades(nfp, 1, window, +1)
        p = t.pnl_bps.to_numpy()
        print(f"  {window:>5}m {len(p):>4} {p.mean():>10.2f} {p.mean()-TICK_BPS:>9.2f} "
              f"{(p > 0).mean():>6.1%} {tstat(p):>+7.2f}")

    print("\n" + "=" * 74)
    print("2. Split at 2020 — does the sign and size survive a regime break?")
    print("=" * 74)
    print(f"  {'exit':>6} {'full t':>8} {'<=2020 bps':>11} {'t':>7} {'2021+ bps':>10} {'t':>7} {'holds':>7}")
    for window in (10, 15, 30):
        t = trades(nfp, 1, window, +1)
        early = t[t.ts < "2021-01-01"].pnl_bps.to_numpy()
        late = t[t.ts >= "2021-01-01"].pnl_bps.to_numpy()
        holds = "yes" if np.sign(early.mean()) == np.sign(late.mean()) else "NO"
        print(f"  {window:>5}m {tstat(t.pnl_bps.to_numpy()):>+8.2f} {early.mean():>11.2f} "
              f"{tstat(early):>+7.2f} {late.mean():>10.2f} {tstat(late):>+7.2f} {holds:>7}")

    print("\n" + "=" * 74)
    print("3. Surprise-size gradient at the 10-minute exit (a real effect is monotone)")
    print("=" * 74)
    t = trades(nfp, 1, 10, +1)
    print(f"  {'filter':>10} {'n':>4} {'gross bps':>10} {'hit':>7} {'t':>7}")
    for thresh in (0.0, 0.5, 1.0, 1.5):
        sub = t[t.z.abs() > thresh].pnl_bps.to_numpy()
        if len(sub) < 10:
            continue
        print(f"  |z|>{thresh:<5} {len(sub):>4} {sub.mean():>10.2f} {(sub > 0).mean():>6.1%} {tstat(sub):>+7.2f}")

    print("\n" + "=" * 74)
    print("4. Best-of-32 null — the sweep looked at 32 cells to find this one")
    print("=" * 74)
    base = trades(nfp, 1, 10, +1)
    rets = base.ret_bps.to_numpy()
    real_t = abs(tstat(base.pnl_bps.to_numpy()))
    best = np.empty(4000)
    for i in range(4000):
        # 32 pure-noise sign features scored on the same NFP returns
        signs = RNG.choice([-1.0, 1.0], size=(32, len(rets)))
        ts_ = np.abs((signs * rets).mean(1) / ((signs * rets).std(1, ddof=1) / np.sqrt(len(rets))))
        best[i] = ts_.max()
    print(f"  real best-of-32 |t| : {real_t:.2f}")
    print(f"  null median         : {np.median(best):.2f}")
    print(f"  null 95th pct       : {np.percentile(best, 95):.2f}")
    print(f"  P(best-of-32 >= real): {(best >= real_t).mean():.3f}")


if __name__ == "__main__":
    main()
