"""
make_readme_figures.py
======================
Builds the two overview figures used at the top of Reza/README.md:

  figures/fig01_markets_overview.png   daily roll-adjusted price of ES, NQ and ZN, 2010-2026
  figures/fig02_cpi_event_paths.png    average price path around CPI releases, hot vs cool
                                        prints, for ES, NQ and ZN (the "jump then drift" picture)

Run from the project folder (the one that contains src/, data/ and figures/):

    python scripts/make_readme_figures.py

Data needed (same files as the notebooks):
    data/raw/ES_1m_2010_2026_full.parquet
    data/processed/macro_event_model_data_2016_2026.parquet            (Notebook 02 output)
    futures_1min_clean_v3_{NQ,ZN}.parquet in the team repo's data/processed/
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.state_conditioning as sc   # noqa: E402
import src.multi_asset as ma          # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
RAW = ROOT / "data" / "raw"
EVENT_FILE = ROOT / "data" / "processed" / "macro_event_model_data_2016_2026.parquet"
TEAM_DATA = next((p for p in [ROOT.parent / "data" / "processed",
                              ROOT.parent / "blackrock-intraday" / "data" / "processed"]
                  if (p / "futures_1min_clean_v3_NQ.parquet").exists()),
                 ROOT.parent / "data" / "processed")
COLORS = {"ES": "#1f4e79", "NQ": "#7a3e9d", "ZN": "#2e7d32"}
NAMES = {"ES": "ES — E-mini S&P 500", "NQ": "NQ — E-mini Nasdaq-100", "ZN": "ZN — 10-year T-note"}

print("loading 1-minute bars ...")
bars = {
    "ES": sc.load_es_minute(RAW / "ES_1m_2010_2026_full.parquet"),
    "NQ": ma.load_bars(TEAM_DATA / "futures_1min_clean_v3_NQ.parquet", "team_v3"),
    "ZN": ma.load_bars(TEAM_DATA / "futures_1min_clean_v3_ZN.parquet", "team_v3"),
}
adj = {m: sc.build_adjusted_log_price(b)[0] for m, b in bars.items()}

# ---------------------------------------------------------------- figure 1
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for ax, m in zip(axes, ["ES", "NQ", "ZN"]):
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
fig.suptitle("The three markets: roll-adjusted daily price, 2010-2026", fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "fig01_markets_overview.png", dpi=150)
plt.close()
print("saved figures/fig01_markets_overview.png")

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


fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
for ax, m in zip(axes, ["ES", "NQ", "ZN"]):
    for (lab, mask), col in zip(groups.items(), ["#c62828", "#1565c0"]):
        p = paths(adj[m], cpi.loc[mask, "timestamp_utc"])
        mean = np.nanmean(p, axis=0)
        ax.plot(OFFS + 1, mean, color=col, lw=2, label=f"{lab}, n={int(mask.sum())}")
    ax.axvline(0, color="k", lw=0.8)
    ax.axvspan(1, 30, color="gold", alpha=0.18, label="traded window (T+1 to T+30)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(NAMES[m], fontsize=11)
    ax.set_xlabel("minutes after the release")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("average cumulative return vs pre-release price (bps)")
axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Average market path around CPI releases (2016-2026): a jump at the release, then a smaller drift",
             fontsize=12)
plt.tight_layout()
plt.savefig(FIG / "fig02_cpi_event_paths.png", dpi=150)
plt.close()
print("saved figures/fig02_cpi_event_paths.png")
