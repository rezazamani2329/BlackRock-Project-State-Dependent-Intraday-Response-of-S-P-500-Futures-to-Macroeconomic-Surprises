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
# # 超参搜索 — waiting time × path length × neighbour weighting
#
# 在 `combined/` 的事件反应书上扫三组构造，**不改 `combined/` 里的代码**。
#
# 1. **waiting time** `w ∈ {1, 2, 5}`。锚点整条梯子平移：`T+w, T+w+30, …, T+w+120`。
# 2. **path** `path_pre ∈ {30, 45, 60}`。输入 DTW 的路径长度是 `path_pre + w`，在入场分钟结束。
# 3. **weighting**。k 近邻选好之后，用哪一组权重把邻居的远期收益聚成 `dtw_dir`。
#
# 对每个 `(w, path_pre)` 只算一遍 DTW 距离，再套全部 `k × lookback × weighting`。
# 打分只看 train（2013-2017）和 valid（2018-2020）。test 不进这张表。
#
# 中断后重跑会跳过已经写进 `output/hp_search_by_asset.csv` 的构造。

# %%
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path("combined_modified_hoshea").resolve()
sys.path.insert(0, str(HERE))

from config import (
    ASSETS,
    DTW_GRID,
    OUT,
    PATH_PRE_GRID_MIN,
    TRAIN_YEARS_SPAN,
    VALID_YEARS_SPAN,
    WAIT_GRID_MIN,
    WEIGHTING_GRID,
    anchors_for_wait,
    path_len_minutes,
)
from engine import (
    admit_games,
    build_games,
    dtw_block,
    impact_by_year,
    load_families,
    load_prices,
    score_span,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)

ASSET_PATH = OUT / "hp_search_by_asset.csv"
POOLED_PATH = OUT / "hp_search_pooled.csv"
CHOSEN_PATH = OUT / "hp_search_chosen.csv"


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="DTW construction search")
    p.add_argument("--assets", default=",".join(ASSETS), help="comma-separated, e.g. ES or ES,NQ")
    p.add_argument("--waits", default=",".join(map(str, WAIT_GRID_MIN)))
    p.add_argument("--path-pres", default=",".join(map(str, PATH_PRE_GRID_MIN)))
    p.add_argument("--weightings", default=",".join(WEIGHTING_GRID))
    if argv is None:
        argv = sys.argv[1:] if "ipykernel" not in sys.modules else []
    return p.parse_args(argv)


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _strs(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _done_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    prev = pd.read_csv(path)
    return set(zip(prev.wait_min, prev.path_pre_min, prev.asset))


def _write_rows(path: Path, rows: list[dict]) -> None:
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
        new = new.drop_duplicates(
            ["wait_min", "path_pre_min", "weighting", "k", "lookback_years", "asset"],
            keep="last",
        )
    new.to_csv(path, index=False)


def pooled_table(by_asset: pd.DataFrame) -> pd.DataFrame:
    keys = ["wait_min", "path_pre_min", "path_len", "weighting", "k", "lookback_years"]
    g = by_asset.groupby(keys, as_index=False)
    out = g.agg(
        n_assets=("asset", "nunique"),
        train_games=("train_games", "sum"),
        valid_games=("valid_games", "sum"),
        train_dir_ic=("train_dir_ic", "mean"),
        valid_dir_ic=("valid_dir_ic", "mean"),
        train_disp_ic=("train_disp_ic", "mean"),
        valid_disp_ic=("valid_disp_ic", "mean"),
        train_hit=("train_hit", "mean"),
        valid_hit=("valid_hit", "mean"),
        train_q51=("train_q51", "mean"),
        valid_q51=("valid_q51", "mean"),
    )
    return out.sort_values("train_dir_ic", ascending=False).reset_index(drop=True)


def run_search(
    symbols: list[str] | None = None,
    waits: list[int] | None = None,
    path_pres: list[int] | None = None,
    weightings: list[str] | None = None,
) -> pd.DataFrame:
    symbols = symbols or list(ASSETS)
    waits = waits or list(WAIT_GRID_MIN)
    path_pres = path_pres or list(PATH_PRE_GRID_MIN)
    weightings = weightings or list(WEIGHTING_GRID)
    ks, lookbacks = DTW_GRID["k"], DTW_GRID["lookback_years"]

    print("grid:")
    print(f"  assets     {symbols}")
    print(f"  waits      {waits}   anchors e.g. w=2 -> {anchors_for_wait(2)}")
    print(f"  path_pre   {path_pres}   path_len = path_pre + wait")
    print(f"  weighting  {weightings}")
    print(f"  k × lb     {ks} × {lookbacks}")
    n_dtw = len(waits) * len(path_pres) * len(symbols)
    n_rows = n_dtw * len(weightings) * len(ks) * len(lookbacks)
    print(f"  DTW jobs   {n_dtw}    scored rows {n_rows}")

    pxs, contracts = load_prices()
    fam_stamps = load_families()
    impacts = {}
    for s in symbols:
        years = sorted({t.year for ts in fam_stamps.values() for t in ts})
        impacts[s] = impact_by_year(pxs[s], fam_stamps, years)
        print(f"{s}: impact screen {len(impacts[s]):,} family-years")

    done = _done_keys(ASSET_PATH)
    job = 0
    for wait in waits:
        for pre in path_pres:
            plen = path_len_minutes(pre, wait)
            anchors = anchors_for_wait(wait)
            for s in symbols:
                job += 1
                key = (wait, pre, s)
                if key in done:
                    print(f"[{job}/{n_dtw}] skip {s} wait={wait} pre={pre} (already in csv)")
                    continue
                t0 = time.time()
                print(f"[{job}/{n_dtw}] {s}  wait={wait}  pre={pre}  path={plen}  "
                      f"anchors={anchors}", flush=True)
                g = build_games(pxs[s], contracts[s], fam_stamps, wait, pre)
                g = admit_games(g, impacts[s])
                n_adm = int(g.admitted.sum())
                print(f"         {len(g):,} games, {n_adm:,} admitted, DTW …", flush=True)
                feats = dtw_block(g, g.admitted.to_numpy(), ks, lookbacks, weightings)
                rows = []
                for (k, lb, scheme), f in feats.items():
                    tr = score_span(g, f, TRAIN_YEARS_SPAN)
                    va = score_span(g, f, VALID_YEARS_SPAN)
                    rows.append({
                        "wait_min": wait,
                        "path_pre_min": pre,
                        "path_len": plen,
                        "anchors": ",".join(map(str, anchors)),
                        "weighting": scheme,
                        "k": k,
                        "lookback_years": lb,
                        "asset": s,
                        "n_games": len(g),
                        "n_admitted": n_adm,
                        "train_games": tr["games"],
                        "train_dir_ic": tr["dir_ic"],
                        "train_disp_ic": tr["disp_ic"],
                        "train_hit": tr["hit"],
                        "train_q51": tr["q51"],
                        "valid_games": va["games"],
                        "valid_dir_ic": va["dir_ic"],
                        "valid_disp_ic": va["disp_ic"],
                        "valid_hit": va["hit"],
                        "valid_q51": va["q51"],
                    })
                _write_rows(ASSET_PATH, rows)
                done.add(key)
                print(f"         wrote {len(rows)} rows in {time.time() - t0:.0f}s", flush=True)

    by_asset = pd.read_csv(ASSET_PATH)
    pooled = pooled_table(by_asset)
    pooled.to_csv(POOLED_PATH, index=False)
    print(f"\nwrote {ASSET_PATH.name}  {len(by_asset):,} rows")
    print(f"wrote {POOLED_PATH.name}  {len(pooled):,} rows")
    return by_asset


def show_results() -> pd.DataFrame:
    by_asset = pd.read_csv(ASSET_PATH)
    pooled = pooled_table(by_asset)
    pooled.to_csv(POOLED_PATH, index=False)

    print("top 15 by train pooled dir IC:")
    print(pooled.head(15).round(4).to_string(index=False))
    print("\nbottom 8 by train pooled dir IC:")
    print(pooled.tail(8).round(4).to_string(index=False))

    best = pooled.iloc[0]
    print(f"\nchosen on train pooled dir IC: wait={int(best.wait_min)}  "
          f"path_pre={int(best.path_pre_min)}  path_len={int(best.path_len)}  "
          f"weighting={best.weighting}  k={int(best.k)}  lookback={int(best.lookback_years)}  "
          f"train dir IC={best.train_dir_ic:.4f}  valid dir IC={best.valid_dir_ic:.4f}")
    best.to_frame("chosen").T.to_csv(CHOSEN_PATH, index=False)

    for axis, cols in [
        ("wait_min", ["wait_min"]),
        ("path_pre_min", ["path_pre_min"]),
        ("weighting", ["weighting"]),
        ("k × lookback", ["k", "lookback_years"]),
    ]:
        m = pooled.groupby(cols)[["train_dir_ic", "valid_dir_ic", "train_disp_ic",
                                  "valid_disp_ic", "train_hit", "valid_hit"]].mean()
        print(f"\n{axis}  (mean over the other axes):")
        print(m.round(4).to_string())
    return pooled


# %% [markdown]
# ## 跑网格
#
# 完整网格是 9 组路径构造 × 3 个品种的 DTW，大约数小时。可先
# `python hyperparam_search.py --assets ES --waits 1 --path-pres 30` 冒烟。
# 中断后重跑会跳过已完成的 `(wait, path_pre, asset)`。

# %%
if __name__ == "__main__":
    if "ipykernel" not in sys.modules:
        args = _parse_args()
        run_search(
            symbols=_strs(args.assets),
            waits=_ints(args.waits),
            path_pres=_ints(args.path_pres),
            weightings=_strs(args.weightings),
        )
    else:
        run_search()

# %% [markdown]
# ## 结果表
#
# 每个品种一行一组超参；`hp_search_pooled.csv` 是三品种 train/valid 指标的等权平均。
# 选择仍按 **train 上的 pooled dir IC**，和 `combined/features.py` 一样。valid 只用来核对。

# %%
if ASSET_PATH.exists():
    pooled = show_results()
else:
    print(f"no {ASSET_PATH.name} yet — run the search cell first")

# %% [markdown]
# **Findings:** 27 组路径 × 7 种权重全部跑完（567 个 pooled 组合）。train 上最好的是
# wait=5 / path_pre=45 / recency / k=15 / lookback=5：pooled dir IC **+0.030**，但 valid
# 立刻掉到 **−0.0006**。排在前面的几乎全是 wait=5 + path_pre=45，valid 也都在零附近。
# 按轴平均：wait=5 的 train dir IC 最不负（−0.004，vs wait=1 的 −0.020）；path_pre=45
# 好于 30/60；七种权重彼此差不到 0.01，`uniform` 略好。`dtw_disp` 仍然稳（train +0.13
# 到 +0.21）。combined 原设定（wait=1, path=31, inv_dist_recency, k=10, lb=3）train
# −0.015、valid **+0.056**，和这次 train 赢家正好相反——不要按 train dir IC 直接定稿。
