"""Train/valid/test Sharpe for a few HP specs. Not a selection run — comparison only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import (
    ASSETS,
    CLEAN_BARS_FMT,
    COST_TICKS_PER_SIDE,
    GATE_GRID,
    MAX_POSITION,
    NY,
    SESSION_SHIFT,
    SIZING_GRID,
    TEST_YEARS_SPAN,
    TRAIN_YEARS_SPAN,
    VALID_YEARS_SPAN,
)
from engine import admit_games, build_games, dtw_block, impact_by_year, load_families, load_prices

COST = "quarter tick"
SYMBOLS = list(ASSETS)

SPECS = [
    dict(name="baseline (combined)", wait=1, pre=30, weighting="inv_dist_recency", k=10, lb=3),
    dict(name="train-IC winner", wait=5, pre=45, weighting="recency", k=15, lb=5),
    dict(name="valid-IC cluster", wait=1, pre=30, weighting="rank_recency", k=25, lb=2),
    dict(name="baseline, k=25 (disp)", wait=1, pre=30, weighting="inv_dist_recency", k=25, lb=3),
]


def session_of(ts: pd.Series) -> pd.Series:
    return (ts.dt.tz_convert(NY) + SESSION_SHIFT).dt.normalize().dt.tz_localize(None)


def in_years(days: pd.DatetimeIndex, span: tuple[int, int]) -> pd.DatetimeIndex:
    return days[(days.year >= span[0]) & (days.year <= span[1])]


def sharpe(daily: pd.Series) -> float:
    sd = daily.std(ddof=1)
    return float(daily.mean() / sd * np.sqrt(252)) if sd else np.nan


def make_rule(fit: pd.DataFrame, q: float, sizing: str) -> dict:
    return {
        "q": q,
        "sizing": sizing,
        "tail_lo": fit.dtw_dir.quantile(q),
        "tail_hi": fit.dtw_dir.quantile(1 - q),
        "disp_ref": fit.dtw_disp.median(),
    }


def trades_for(g_asset: pd.DataFrame, bars_s: pd.DataFrame, s: str, span, rule) -> pd.DataFrame:
    g = g_asset[g_asset.year.between(*span)].copy()
    in_tail = (g.dtw_dir <= rule["tail_lo"]) | (g.dtw_dir >= rule["tail_hi"])
    size = (
        1.0
        if rule["sizing"] == "flat"
        else np.clip(rule["disp_ref"] / g.dtw_disp, 0.0, MAX_POSITION)
    )
    g["position"] = np.where(in_tail, np.sign(g.dtw_dir) * size, 0.0)
    t = (
        g[g.position != 0]
        .groupby("entry_ts")
        .agg(position=("position", "mean"), fwd_ret=("fwd_ret", "first"))
        .reset_index()
    )
    t = t[t.position != 0]
    if t.empty:
        return t
    t["session"] = session_of(t.entry_ts)
    tick_bps = ASSETS[s]["tick"] / bars_s.close.reindex(t.entry_ts).to_numpy() * 1e4
    t["gross"] = t.position * t.fwd_ret * 1e4
    for name, ticks in COST_TICKS_PER_SIDE.items():
        t[name] = t.gross - t.position.abs() * 2 * ticks * tick_bps
    return t


def daily_pnl(t: pd.DataFrame, days: pd.DatetimeIndex, col: str) -> pd.Series:
    if t.empty:
        return pd.Series(0.0, index=days)
    return t.groupby("session")[col].sum().reindex(days).fillna(0.0)


def eval_span(panel: pd.DataFrame, bars: dict, days: dict, span, rules: dict) -> dict:
    out = {}
    trades_all = []
    dailies = {}
    for s in SYMBOLS:
        g = panel[panel.symbol == s]
        t = trades_for(g, bars[s], s, span, rules[s])
        d = daily_pnl(t, in_years(days[s], span), COST)
        out[f"{s}_sharpe"] = sharpe(d)
        out[f"{s}_trades"] = len(t)
        out[f"{s}_bps"] = float(t[COST].mean()) if len(t) else np.nan
        trades_all.append(t)
        dailies[s] = d
    # equal-weight daily average across assets (not the combined book's risk weights)
    idx = dailies[SYMBOLS[0]].index
    for s in SYMBOLS[1:]:
        idx = idx.union(dailies[s].index)
    pooled = sum(dailies[s].reindex(idx).fillna(0.0) for s in SYMBOLS) / 3
    out["pooled_sharpe"] = sharpe(pooled)
    out["trades"] = int(sum(out[f"{s}_trades"] for s in SYMBOLS))
    return out


def pick_gate(panel: pd.DataFrame, bars: dict, days: dict) -> tuple[float, str]:
    rows = []
    for q in GATE_GRID:
        for sizing in SIZING_GRID:
            rules = {
                s: make_rule(
                    panel[(panel.symbol == s) & panel.year.between(*TRAIN_YEARS_SPAN)],
                    q,
                    sizing,
                )
                for s in SYMBOLS
            }
            m = eval_span(panel, bars, days, TRAIN_YEARS_SPAN, rules)
            rows.append((m["pooled_sharpe"], q, sizing))
    rows.sort(reverse=True)
    return rows[0][1], rows[0][2]


def main():
    pxs, contracts = load_prices()
    fam = load_families()
    bars, days = {}, {}
    for s in SYMBOLS:
        b = pd.read_parquet(
            CLEAN_BARS_FMT.format(symbol=s), columns=["close", "contract", "session_date"]
        )
        b.index = pd.to_datetime(b.index, utc=True)
        bars[s] = b.sort_index()
        r = np.log(bars[s].close).diff() * 1e4
        r[bars[s].contract.ne(bars[s].contract.shift())] = 0.0
        bnh = r.fillna(0.0).groupby(pd.DatetimeIndex(bars[s].session_date)).sum()
        days[s] = bnh.index

    impacts = {}
    years = sorted({t.year for ts in fam.values() for t in ts})
    for s in SYMBOLS:
        impacts[s] = impact_by_year(pxs[s], fam, years)
        print(f"{s}: impact {len(impacts[s])} family-years")

    by_con: dict[tuple, dict] = {}
    for wait, pre in {(sp["wait"], sp["pre"]) for sp in SPECS}:
        ks = sorted({sp["k"] for sp in SPECS if sp["wait"] == wait and sp["pre"] == pre})
        lbs = sorted({sp["lb"] for sp in SPECS if sp["wait"] == wait and sp["pre"] == pre})
        ws = sorted({sp["weighting"] for sp in SPECS if sp["wait"] == wait and sp["pre"] == pre})
        print(f"\nDTW wait={wait} pre={pre} k={ks} lb={lbs} w={ws}")
        feats = {}
        games = {}
        for s in SYMBOLS:
            g = admit_games(build_games(pxs[s], contracts[s], fam, wait, pre), impacts[s])
            print(f"  {s}: {len(g):,} games, {int(g.admitted.sum()):,} admitted")
            f = dtw_block(g, g.admitted.to_numpy(), ks, lbs, ws)
            games[s] = g.drop(columns=["path"])
            feats[s] = f
        by_con[wait, pre] = dict(games=games, feats=feats)

    rows = []
    for sp in SPECS:
        games, feats = by_con[sp["wait"], sp["pre"]]["games"], by_con[sp["wait"], sp["pre"]]["feats"]
        parts = []
        for s in SYMBOLS:
            f = feats[s][sp["k"], sp["lb"], sp["weighting"]]
            parts.append(pd.concat([games[s], f], axis=1).assign(symbol=s))
        panel = pd.concat(parts, ignore_index=True)
        panel = panel[panel.admitted & panel.dtw_dir.notna()]

        for mode in ("frozen_q0.4_invdisp", "retune_gate_on_train"):
            if mode == "frozen_q0.4_invdisp":
                q, sizing = 0.4, "inverse_disp"
            else:
                q, sizing = pick_gate(panel, bars, days)
            rules = {
                s: make_rule(
                    panel[(panel.symbol == s) & panel.year.between(*TRAIN_YEARS_SPAN)],
                    q,
                    sizing,
                )
                for s in SYMBOLS
            }
            rec = dict(
                spec=sp["name"],
                wait=sp["wait"],
                pre=sp["pre"],
                weighting=sp["weighting"],
                k=sp["k"],
                lb=sp["lb"],
                mode=mode,
                q=q,
                sizing=sizing,
            )
            for label, span in (
                ("train", TRAIN_YEARS_SPAN),
                ("valid", VALID_YEARS_SPAN),
                ("test", TEST_YEARS_SPAN),
            ):
                m = eval_span(panel, bars, days, span, rules)
                rec[f"{label}_pooled"] = m["pooled_sharpe"]
                rec[f"{label}_trades"] = m["trades"]
                for s in SYMBOLS:
                    rec[f"{label}_{s}"] = m[f"{s}_sharpe"]
            rows.append(rec)
            print(
                f"{sp['name']:28s} {mode:22s} q={q} {sizing:13s}  "
                f"train {rec['train_pooled']:+.3f}  valid {rec['valid_pooled']:+.3f}  "
                f"test {rec['test_pooled']:+.3f}"
            )

    out = pd.DataFrame(rows)
    dest = HERE / "output" / "hp_backtest_compare.csv"
    out.to_csv(dest, index=False)
    print(f"\nwrote {dest}")
    cols = [
        "spec",
        "mode",
        "q",
        "sizing",
        "train_pooled",
        "valid_pooled",
        "test_pooled",
        "train_ES",
        "valid_ES",
        "test_ES",
        "train_NQ",
        "valid_NQ",
        "test_NQ",
        "train_ZN",
        "valid_ZN",
        "test_ZN",
    ]
    print(out[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
