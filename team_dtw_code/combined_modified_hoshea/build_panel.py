"""Rebuild features.parquet for one (wait, path_pre, weighting, k, lookback) combo.

Default: the train-IC winner in output/hp_search_chosen.csv.
Then run backtest.py against that panel.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import ASSETS, OUT, PANEL, TRAIN_YEARS_SPAN, anchors_for_wait, path_len_minutes
from engine import admit_games, build_games, dtw_block, impact_by_year, load_families, load_prices

CHOSEN_PATH = OUT / "hp_search_chosen.csv"


def _from_chosen() -> dict:
    if not CHOSEN_PATH.exists():
        raise SystemExit(f"no {CHOSEN_PATH.name}; run hyperparam_search.py first or pass flags")
    row = pd.read_csv(CHOSEN_PATH).iloc[0]
    return dict(
        wait_min=int(row.wait_min),
        path_pre_min=int(row.path_pre_min),
        weighting=str(row.weighting),
        k=int(row.k),
        lookback_years=int(row.lookback_years),
    )


def build_panel(wait_min: int, path_pre_min: int, weighting: str, k: int, lookback_years: int) -> pd.DataFrame:
    pxs, contracts = load_prices()
    fam_stamps = load_families()
    parts = []
    for s in ASSETS:
        years = sorted({t.year for ts in fam_stamps.values() for t in ts})
        g = build_games(pxs[s], contracts[s], fam_stamps, wait_min, path_pre_min)
        g = admit_games(g, impact_by_year(pxs[s], fam_stamps, years))
        print(f"{s}: {len(g):,} games, DTW k={k} lb={lookback_years} {weighting} …")
        feats = dtw_block(g, g.admitted.to_numpy(), [k], [lookback_years], [weighting])
        g = pd.concat([g.drop(columns=["path"]), feats[k, lookback_years, weighting]], axis=1)
        g["symbol"] = s
        parts.append(g)
    panel = pd.concat(parts, ignore_index=True)
    panel = panel[["symbol"] + [c for c in panel.columns if c != "symbol"]]
    panel.to_parquet(PANEL)
    print(f"wrote {PANEL}  {len(panel):,} rows")
    print(f"wait={wait_min} path_pre={path_pre_min} path_len={path_len_minutes(path_pre_min, wait_min)} "
          f"anchors={anchors_for_wait(wait_min)} weighting={weighting} k={k} lookback={lookback_years}")
    print(f"train span used only to choose this combo was {TRAIN_YEARS_SPAN}")
    return panel


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--wait", type=int)
    p.add_argument("--path-pre", type=int)
    p.add_argument("--weighting")
    p.add_argument("--k", type=int)
    p.add_argument("--lookback", type=int)
    args = p.parse_args(argv)
    have_all = all(v is not None for v in (args.wait, args.path_pre, args.weighting, args.k, args.lookback))
    spec = (dict(wait_min=args.wait, path_pre_min=args.path_pre, weighting=args.weighting,
                 k=args.k, lookback_years=args.lookback) if have_all else _from_chosen())
    if args.wait is not None:
        spec["wait_min"] = args.wait
    if args.path_pre is not None:
        spec["path_pre_min"] = args.path_pre
    if args.weighting is not None:
        spec["weighting"] = args.weighting
    if args.k is not None:
        spec["k"] = args.k
    if args.lookback is not None:
        spec["lookback_years"] = args.lookback
    build_panel(**spec)


if __name__ == "__main__":
    main()
