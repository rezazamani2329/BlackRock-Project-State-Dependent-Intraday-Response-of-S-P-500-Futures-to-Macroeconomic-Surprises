"""Pull 1-minute bars for the extra assets from Databento.

    uv run python combined/pull.py                  # print the cost of each pull, download nothing
    uv run python combined/pull.py --pull           # download whatever is not already on disk
    uv run python combined/pull.py --pull YM        # only these symbols (if missing on disk)

Covers ASSETS and the candidate EXTRA_ASSETS in config.py (e.g. YM, the E-mini Dow).

Same source and window as the ES export (`GLBX.MDP3`, `ohlcv-1m`, 2010-07 -> 2026-07), in
monthly chunks. NQ uses the calendar roll like ES; the Treasuries use the volume roll,
because their volume migrates around month-end before delivery rather than on a third
Friday. `instrument_id` is kept so cleaning can drop the sessions a roll lands in.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import databento as db
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ALL_ASSETS as ASSETS, ROOT

START, END = "2010-07-01", "2026-07-01"
DATASET, SCHEMA = "GLBX.MDP3", "ohlcv-1m"


def main(pull: bool, only: list[str] | None = None) -> None:
    load_dotenv(ROOT / ".env")
    key = os.getenv("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError("DATABENTO_API_KEY not set — copy .env.example to .env")
    client = db.Historical(key)

    todo = {s: a for s, a in ASSETS.items()
            if not a["raw"].exists() and (not only or s in only)}
    for sym, a in ASSETS.items():
        state = "on disk" if a["raw"].exists() else ("missing, will pull" if sym in todo else "missing, skipped")
        print(f"{sym}: {a['databento']:8s} -> {a['raw'].name}  ({state})")

    total = 0.0
    for sym, a in todo.items():
        cost = client.metadata.get_cost(dataset=DATASET, symbols=[a["databento"]],
                                        stype_in="continuous", schema=SCHEMA,
                                        start=START, end=END)
        total += cost
        print(f"  cost {sym}: ${cost:,.2f}")
    print(f"total for missing files: ${total:,.2f}")
    if not pull or not todo:
        return

    months = pd.date_range(START, END, freq="MS")
    for sym, a in todo.items():
        frames = []
        for s, e in zip(months[:-1], months[1:]):
            try:
                df = client.timeseries.get_range(
                    dataset=DATASET, symbols=[a["databento"]], stype_in="continuous",
                    schema=SCHEMA, start=s.strftime("%Y-%m-%d"), end=e.strftime("%Y-%m-%d"),
                ).to_df()
            except Exception as exc:              # keep going; report the gap
                print(f"  {sym} {s:%Y-%m} failed: {exc}")
                continue
            if not df.empty:
                frames.append(df)
            print(f"  {sym} {s:%Y-%m}: {len(df):,} bars", end="\r")
        out = pd.concat(frames).sort_index()
        out = out[~out.index.duplicated(keep="first")]
        out.to_parquet(a["raw"], compression="zstd")
        print(f"\n{sym}: wrote {a['raw'].name}  {len(out):,} bars  "
              f"{out.index.min()} -> {out.index.max()}")


if __name__ == "__main__":
    only = [a.upper() for a in sys.argv[1:] if not a.startswith("--")]
    unknown = [a for a in only if a not in ASSETS]
    if unknown:
        raise SystemExit(f"unknown symbol(s) {unknown}; known: {list(ASSETS)}")
    main(pull="--pull" in sys.argv, only=only or None)
