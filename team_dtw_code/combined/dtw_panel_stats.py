"""
dtw_panel_stats.py
==================
Counts for the "DTW event panel" table of the structured DTW paper (Data section).
Reads the with_YM panel, so all four assets come from one run:

    DTW_EXTRA=YM uv run python combined/dtw_panel_stats.py

Writes combined/output/with_YM/dtw_panel_stats.csv and prints the table.
"""
import os, sys
from pathlib import Path
os.environ.setdefault("DTW_EXTRA", "YM")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from config import CLEAN_BARS_FMT, OUT, PANEL, TRAIN_YEARS_SPAN, VALID_YEARS_SPAN, TEST_YEARS_SPAN

p = pd.read_parquet(PANEL)
rows = {}
for s, g in p.groupby("symbol"):
    bars = pd.read_parquet(CLEAN_BARS_FMT.format(symbol=s), columns=["close"])
    adm = g[g.admitted & g.dtw_dir.notna()]
    r = {"clean bars": len(bars),
         "games, all families": len(g),
         "releases (event_ts)": g.event_ts.nunique(),
         "families admitted (some year)": adm.family.nunique(),
         "admitted games": len(adm)}
    for lab, (a, b) in [("burn-in 2010-12", (2010, 2012)), ("train", TRAIN_YEARS_SPAN),
                        ("valid", VALID_YEARS_SPAN), ("test", TEST_YEARS_SPAN)]:
        r[f"admitted games, {lab}"] = int(adm.year.between(a, b).sum())
    r["admitted games at T+1"] = int((adm.anchor == 1).sum())
    r["median dtw_disp, bps"] = adm.dtw_disp.median() * 1e4
    r["sd of 30-min return on admitted games, bps"] = adm.fwd_ret.std() * 1e4
    r["admitted families"] = ", ".join(sorted(adm.family.unique()))
    rows[s] = r
t = pd.DataFrame(rows)[["ES", "NQ", "YM", "ZN"]]
t.to_csv(OUT / "dtw_panel_stats.csv")
pd.set_option("display.width", 250, "display.max_colwidth", 200)
print(t.to_string())
