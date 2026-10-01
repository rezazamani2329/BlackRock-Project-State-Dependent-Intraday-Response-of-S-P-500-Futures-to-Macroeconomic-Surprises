"""
make_paper_figures_ym.py
========================
Four-market (ES, NQ, YM, ZN) versions of the two overview figures used in the paper,
plus the CPI numbers quoted in the text.

  figures/fig01_markets_overview_4m.png   daily roll-adjusted price, 2010-2026
  figures/fig02_cpi_event_paths_4m.png    average path around CPI releases, hot vs cool prints
  results/paper_cpi_numbers_4m.csv        jump, 30-minute level and drift slopes per market

Run from the project folder (the one with src/, data/ and figures/):

    uv run --project ../blackrock-intraday python scripts/make_paper_figures_ym.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.state_conditioning as sc   # noqa: E402
import src.multi_asset as ma          # noqa: E402

FIG = ROOT / "figures"
RES = ROOT / "results"
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
EVENT_FILE = PROC / "macro_event_model_data_2016_2026.parquet"
TEAM_DATA = next((p for p in [ROOT.parent / "data" / "processed",
                              ROOT.parent / "blackrock-intraday" / "data" / "processed"]
                  if (p / "futures_1min_clean_v3_YM.parquet").exists()),
                 ROOT.parent / "blackrock-intraday" / "data" / "processed")
MKTS = ["ES", "NQ", "YM", "ZN"]
COLORS = {"ES": "#1f4e79", "NQ": "#7a3e9d", "YM": "#b5651d", "ZN": "#2e7d32"}
NAMES = {"ES": "ES — E-mini S&P 500", "NQ": "NQ — E-mini Nasdaq-100",
         "YM": "YM — E-mini Dow", "ZN": "ZN — 10-year T-note"}

print("loading 1-minute bars ...")
bars = {
    "ES": sc.load_es_minute(RAW / "ES_1m_2010_2026_full.parquet"),
    "NQ": ma.load_bars(TEAM_DATA / "futures_1min_clean_v3_NQ.parquet", "team_v3"),
    "YM": ma.load_bars(TEAM_DATA / "futures_1min_clean_v3_YM.parquet", "team_v3"),
    "ZN": ma.load_bars(TEAM_DATA / "futures_1min_clean_v3_ZN.parquet", "team_v3"),
}
adj = {m: sc.build_adjusted_log_price(b)[0] for m, b in bars.items()}

# ---------------------------------------------------------------- figure 1
fig, axes = plt.subplots(4, 1, figsize=(12, 10.5), sharex=True)
for ax, m in zip(axes, MKTS):
    d = sc.build_daily_close(adj[m])["logp_close"]
    idx = 100 * np.exp(d - d.iloc[0])
    ax.plot(idx.index, idx.values, color=COLORS[m], lw=1.1)
    ax.axvspan(pd.Timestamp("2019-01-01"), pd.Timestamp("2026-06-30"), color="grey", alpha=0.08,
               label="evaluation period 2019-2026")
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"), color="red", alpha=0.08,
               label="2022 inflation shock")
    ax.set_ylabel("index (start = 100)")
    ax.set_title(NAMES[m], loc="left", fontsize=11)
    ax.grid(alpha=0.3)
axes[0].legend(loc="upper left", fontsize=8)
fig.suptitle("The four markets: roll-adjusted daily price, 2010-2026", fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "fig01_markets_overview_4m.png", dpi=150)
plt.close()
print("saved figures/fig01_markets_overview_4m.png")

# ---------------------------------------------------------------- figure 2
ev = pd.read_parquet(EVENT_FILE)
ev["timestamp_utc"] = pd.to_datetime(ev["timestamp_utc"], utc=True)
cpi = ev[ev["event_type"] == "CPI"].dropna(subset=["headline_cpi"])
groups = {"hot CPI (headline z > +0.5)": cpi["headline_cpi"] > 0.5,
          "cool CPI (headline z < -0.5)": cpi["headline_cpi"] < -0.5}
OFFS = np.arange(-30, 61)                       # bar labels relative to the release bar T


def paths(a, ts):
    idx = a.index.as_unit("ns") if hasattr(a.index, "as_unit") else a.index
    t_ns, lp = idx.asi8, a["logp_adj"].to_numpy()
    T = pd.DatetimeIndex(ts)
    T = T.as_unit("ns") if hasattr(T, "as_unit") else T
    out = np.full((len(T), len(OFFS)), np.nan)
    for j, k in enumerate(OFFS):
        x = T.asi8 + int(k) * 60 * 10**9
        pos = np.searchsorted(t_ns, x, side="right") - 1
        ok = (pos >= 0) & ((x - t_ns[np.clip(pos, 0, None)]) <= 5 * 60 * 10**9)
        out[:, j] = np.where(ok, lp[np.clip(pos, 0, None)], np.nan)
    base = out[:, list(OFFS).index(-1)][:, None]          # close of bar T-1 = last pre-release price
    return (out - base) * 1e4


rows = []
fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), sharey=False)
for ax, m in zip(axes, MKTS):
    for (lab, mask), col in zip(groups.items(), ["#c62828", "#1565c0"]):
        p = paths(adj[m], cpi.loc[mask, "timestamp_utc"])
        mean = np.nanmean(p, axis=0)
        ax.plot(OFFS + 1, mean, color=col, lw=2, label=f"{lab}, n={int(mask.sum())}")
        at = lambda k: float(mean[list(OFFS).index(k)])           # close of bar T+k
        rows.append(dict(market=m, group=lab.split(" (")[0], n=int(mask.sum()),
                         jump_T_bps=at(0), level_T29_bps=at(29),
                         drift_T_to_T29_bps=at(29) - at(0), pre_T_minus30_bps=at(-30)))
    ax.axvline(0, color="k", lw=0.8)
    ax.axvspan(1, 30, color="gold", alpha=0.18, label="traded window (T+1 to T+30)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(NAMES[m], fontsize=11)
    ax.set_xlabel("minutes after the release")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("average cumulative return vs pre-release price (bps)")
axes[0].legend(fontsize=7.5, loc="lower left")
fig.suptitle("Average market path around CPI releases (2016-2026): a jump at the release, then a smaller drift",
             fontsize=12)
plt.tight_layout()
plt.savefig(FIG / "fig02_cpi_event_paths_4m.png", dpi=150)
plt.close()
print("saved figures/fig02_cpi_event_paths_4m.png")

# ---------------------------------------------------------------- numbers for the text
paths_df = pd.DataFrame(rows)
print(paths_df.round(1).to_string(index=False))

# CPI family-signal slopes (x = signed family signal): 5-minute reaction and T+1..T+30 drift
slope_rows = []
for m in MKTS:
    f = PROC / ("event_state_features_2016_2026.parquet" if m == "ES"
                else f"event_state_features_2016_2026_{m}.parquet")
    if not f.exists():
        continue
    d = pd.read_parquet(f)
    c = d[d["event_type"] == "CPI"]
    xcol = "x" if "x" in c else None
    if xcol is None:
        print(m, "no x column; columns:", list(c.columns)[:40])
        continue
    r = dict(market=m, n=int(c[xcol].notna().sum()))
    for y in ["ret_5m_bps", "ret_30m_bps", "drift_30m_bps"]:
        if y in c:
            cc = c[[xcol, y]].dropna()
            r[f"slope_{y}"] = float(np.polyfit(cc[xcol], cc[y], 1)[0])
            r[f"corr_{y}"] = float(cc[xcol].corr(cc[y]))
    slope_rows.append(r)
slopes = pd.DataFrame(slope_rows)
print(slopes.round(2).to_string(index=False))

RES.mkdir(exist_ok=True)
paths_df.assign(kind="cpi_path").to_csv(RES / "paper_cpi_numbers_4m.csv", index=False)
slopes.to_csv(RES / "paper_cpi_slopes_4m.csv", index=False)
print("saved results/paper_cpi_numbers_4m.csv and results/paper_cpi_slopes_4m.csv")
