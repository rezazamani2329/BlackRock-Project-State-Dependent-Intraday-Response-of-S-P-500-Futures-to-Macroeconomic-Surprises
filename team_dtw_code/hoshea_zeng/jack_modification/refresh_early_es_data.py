"""Re-pull ES 1-minute bars for 2010-2016 from Databento and splice into the raw file.

The existing `data/raw/ES_1m_2010_2026_full.parquet` is short in its early years -- not
because days are missing, but because each day carries far fewer bars than a full session:
~283/day in 2012-06 against ~1,380 for a complete session. That was read as a vendor
coverage limit and it is why `cleaning.py` sets the modelling start at 2016.

It is not a vendor limit. GLBX.MDP3 publishes ohlcv-1m from 2010-06-06, and a fresh pull
of the same symbol and schema returns 1,110 bars/day for that month. Databento has
backfilled early GLBX history since the original download. Months from 2016 onward come
back byte-identical, so only the early range needs replacing.

This writes a NEW raw file rather than editing the existing one in place, per the rule in
data/README.md. Swap it in deliberately, then re-run jack/cleaning.py.

Run: uv run python jack/refresh_early_es_data.py
"""
from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import databento as db
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw/ES_1m_2010_2026_full.parquet"
DST = ROOT / "data/raw/ES_1m_2010_2026_full_v2.parquet"
REFRESH_START = "2010-06-06"     # first date GLBX.MDP3 carries ohlcv-1m
SPLICE = pd.Timestamp("2016-07-01", tz="UTC")   # keep existing data from here on
OVERLAP_CHECK = pd.Timestamp("2016-01-01", tz="UTC")


def main() -> None:
    load_dotenv(ROOT / ".env")
    key = os.getenv("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError("DATABENTO_API_KEY not set; copy .env.example to .env")
    client = db.Historical(key)

    cost = client.metadata.get_cost(dataset="GLBX.MDP3", schema="ohlcv-1m",
                                    symbols=["ES.c.0"], stype_in="continuous",
                                    start=REFRESH_START, end=str(SPLICE.date()))
    print(f"quoted cost for the refresh range: ${cost:.4f}", flush=True)

    months = pd.date_range(REFRESH_START, SPLICE.tz_localize(None), freq="MS")
    spans = [(months[i].strftime("%Y-%m-%d"), months[i + 1].strftime("%Y-%m-%d"))
             for i in range(len(months) - 1)]

    def fetch(span):
        a, b = span
        df = client.timeseries.get_range(dataset="GLBX.MDP3", symbols=["ES.c.0"],
                                         stype_in="continuous", schema="ohlcv-1m",
                                         start=a, end=b).to_df()
        return a, df

    parts, done = [], 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        for a, df in ex.map(fetch, spans):
            if not df.empty:
                parts.append(df)
            done += 1
            print(f"  [{done}/{len(spans)}] {a}  {len(df):>6,} bars", flush=True)
    fresh = pd.concat(parts).sort_index()
    fresh = fresh[~fresh.index.duplicated(keep="first")]

    old = pd.read_parquet(SRC)
    old.index = pd.DatetimeIndex(old.index)

    # sanity: the overlap window must agree, or the two pulls are not the same series
    ov_new = fresh[(fresh.index >= OVERLAP_CHECK) & (fresh.index < SPLICE)]
    ov_old = old[(old.index >= OVERLAP_CHECK) & (old.index < SPLICE)]
    common = ov_new.index.intersection(ov_old.index)
    if len(common):
        diff = (ov_new.loc[common, "close"] - ov_old.loc[common, "close"]).abs().max()
        print(f"\noverlap {OVERLAP_CHECK.date()}..{SPLICE.date()}: "
              f"{len(common):,} shared bars, max |close diff| = {diff:.6f}")
        print(f"  new-only bars in overlap: {len(ov_new) - len(common):,}   "
              f"old-only: {len(ov_old) - len(common):,}")

    combined = pd.concat([fresh[fresh.index < SPLICE],
                          old[old.index >= SPLICE]]).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.to_parquet(DST)

    print(f"\nwrote {DST.relative_to(ROOT)}  {len(combined):,} rows "
          f"(was {len(old):,}, +{len(combined) - len(old):,})")
    print("\nbars per day by year:")
    for src, lab in [(old, "old"), (combined, "new")]:
        idx = pd.DatetimeIndex(src.index)
        s = pd.DataFrame({"y": idx.year, "d": idx.normalize()}).groupby("y").agg(
            bars=("d", "size"), days=("d", "nunique"))
        s[lab] = (s.bars / s.days).round(0).astype(int)
        print(f"  {lab}: " + "  ".join(f"{y}:{v}" for y, v in s[lab].items() if y <= 2017))


if __name__ == "__main__":
    main()
