"""Every parameter for the combined event reaction book, in one place."""
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
# Optional extra assets for the DTW book, e.g. DTW_EXTRA=YM. They are added to BOOK_SYMBOLS and
# every output of that run goes to output/with_<assets>/, so the main results are never touched.
DTW_EXTRA = [x for x in os.environ.get("DTW_EXTRA", "").split(",") if x]
if DTW_EXTRA:
    OUT = OUT / ("with_" + "_".join(DTW_EXTRA))
    OUT.mkdir(exist_ok=True)

NY = ZoneInfo("America/New_York")

# ---------------------------------------------------------------- data
RAW_BARS = RAW / "ES_1m_2010_2026_full_v2.parquet"   # the 2010-2015 re-pull
CLEAN_BARS = PROCESSED / "es_1min_clean_v2.parquet"  # old ES-only file; read by backtest.py and experiments/
BLOOMBERG = RAW / "Bloomberg Economic Releases.xlsx"
# Multi-asset. ES keeps its existing raw file so its results reproduce exactly. Tick sizes
# are in price points. ZT (2-year note) was dropped: before 2019 fewer than 7% of release
# windows have a trade in every minute, and a fresh pull matched tick-level trades exactly,
# so the gaps are genuine no-trade minutes rather than missing data.
# The DTW book (features.py, backtest.py) trades ES, NQ and ZN.
BOOK_SYMBOLS = ["ES", "NQ", "ZN"] + DTW_EXTRA
ASSETS = {
    "ES": dict(databento="ES.c.0", raw=RAW_BARS, tick=0.25, point_value=50),
    "NQ": dict(databento="NQ.c.0", raw=RAW / "NQ_1m_2010_2026.parquet", tick=0.25, point_value=20),
    "ZN": dict(databento="ZN.v.0", raw=RAW / "ZN_1m_2010_2026.parquet", tick=1 / 64, point_value=1000),
}
# Candidate assets: pulled (pull.py) and cleaned (cleaning.py) like the three above, but not
# yet in the books (features.py, backtest.py, surprise_dtw_book.py, team_book.py), so adding
# one does not change any existing result. They are studied first in their own notebook
# (ym_study.py for YM) and moved into ASSETS only once that study says they belong.
# YM: E-mini Dow ($5 x DJIA), CBOT, same 18:00-17:00 ET session, quarterly calendar roll and
# third-Friday expiry as ES and NQ; one tick is 1 index point.
EXTRA_ASSETS = {
    "YM": dict(databento="YM.c.0", raw=RAW / "YM_1m_2010_2026.parquet", tick=1.0, point_value=5),
}
ALL_ASSETS = {**ASSETS, **EXTRA_ASSETS}
# One cleaned file per asset, each under GitHub's 100 MB limit so they can be committed.
CLEAN_BARS_FMT = str(PROCESSED / "futures_1min_clean_v3_{symbol}.parquet")
PANEL = OUT / "features.parquet"
PANEL_LEAKED = OUT / "features_leaked.parquet"

# ---------------------------------------------------------------- sessions
# A session runs 18:00 ET to the next 17:00 ET and is dated by the day it closes
# on; shifting ET forward 6h rolls the evening open onto the closing date.
SESSION_SHIFT = pd.Timedelta(hours=6)
# A session is complete if it has at least this share of the asset's median bars per weekday
# session that year. Relative, because ZN and early NQ genuinely trade fewer minutes than ES;
# 0.85 roughly matches the old ES cut (1,200 of ~1,380) and still drops early-close half-days.
MIN_SESSION_FRAC = 0.85
MODEL_START = pd.Timestamp("2010-06-06", tz="UTC")   # first date GLBX carries ohlcv-1m

# ---------------------------------------------------------------- event games
ENTRY_LAG_MIN = 1              # enter T+1: at T the tape price predates the print
HOLD_MIN = 30                  # exit T+31
# Hoshea's multi-anchor ladder: five back-to-back 30-minute windows after the print.
# Entry offsets from the release, in minutes. ANCHORS = [1] is the single-window book.
ANCHORS = [1, 31, 61, 91, 121]
PATH_PRE_MIN = 30              # path runs [T-30, T+1] and ends exactly at entry
PATH_FREQ_MIN = 1
# Leak test. 0 is the real book: the path ends exactly at entry. Set to 5 to deliberately
# run the path 5 minutes PAST entry, which is the contamination that produced this
# project's worst false positive. Results must change a lot; if they do not, the timing
# plumbing is broken. Writes to PANEL_LEAKED so the real panel is never overwritten.
# Run it without touching the notebook: LEAK_TEST_MINUTES=5 uv run python combined/features.py
LEAK_TEST_MINUTES = int(os.environ.get("LEAK_TEST_MINUTES", 0))

# ---------------------------------------------------------------- dtw
# The book's DTW setting: the original ES book's (commit 9201a51). features.py scores DTW_GRID
# on the training years and uses the best; it is one point. The long-history setting from
# experiments/tune_dtw.py (band 10, lookback 8, no cap, no decay, k 40, uniform) looked better
# on 2013-2020 but scored ES 0.22 against 0.61 on 2021-2026, so it was reverted.
DTW_GRID = {"k": [15], "lookback_years": [3]}
DTW_WARP_MINUTES = 5           # Sakoe-Chiba band
DTW_MAX_POOL = 600             # most recent candidates inside the lookback; None = all
DTW_HALFLIFE_DAYS = 730        # recency half-life; float("inf") = no decay
DTW_WEIGHTS = "inv_dist"       # neighbour weights: "uniform" or "inv_dist" (1 / distance)
MIN_NEIGHBOURS = 20            # below this the factor is not computed
# The old ES book's setting, still read by experiments/anchors.py and anchor_robustness.py.
K_NEIGHBOURS = 15
LOOKBACK_YEARS = 3
MAX_POOL = 600
WARP_MINUTES = 5
RECENCY_HALFLIFE_DAYS = 730

# ---------------------------------------------------------------- universe
FAMILY_OVERLAP = 0.80          # share of timestamps two release lines must share
MIN_RELEASES = 30              # prior observations before a family can be screened
IMPACT_MIN_T = 2.0             # admission: t-stat of the impact statistic
IMPACT_WINDOW_MIN = 30         # window over which release impact is measured

# ---------------------------------------------------------------- backtest
# Costs per side, in ticks of each asset (ASSETS[...]["tick"]), converted to bps at each
# trade's entry price. backtest.py reports every level.
COST_TICKS_PER_SIDE = {"gross": 0.0, "quarter tick": 0.25, "half tick": 0.5}
MAX_POSITION = 2.0             # contracts
# Old ES-only settings, still read by experiments/.
TRAIN_YEARS = 5                # first fit uses 5 years, then expanding walk-forward
TICK_BPS = 0.5                 # one ES tick = 0.25 index pts = $12.50 ~ 0.5 bp
COST_BPS_PER_SIDE = 0.125      # a quarter tick per side -> 0.25 bp round trip
# Gate and sizing. backtest.py picks the best pair on the training years; each grid is one
# point now, the original ES book's rule. The gate trades a
# game only if dtw_dir is below its q quantile or above its 1 - q quantile, so q = 0.3
# trades the outer 60%. Sizing: 1 contract ("flat"), or median(dtw_disp) / dtw_disp capped
# at MAX_POSITION ("inverse_disp").
GATE_GRID = [0.3]             # fixed: trade the outer 60% of dtw_dir, as the original ES book
SIZING_GRID = ["inverse_disp"]  # the original ES book's sizing; see experiments/ for the tune

# ---------------------------------------------------------------- sample split
# 2010-2012 is burn-in: it fills the DTW neighbour library and the impact screen's 30-release
# history, but is never scored. Hyperparameters are chosen on TRAIN, checked on VALID, and
# TEST is run once at the end. Scoring uses admitted games only.
TRAIN_YEARS_SPAN = (2013, 2017)
VALID_YEARS_SPAN = (2018, 2020)
TEST_YEARS_SPAN = (2021, 2026)
