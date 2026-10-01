"""Hyperparameter-search fork of combined/. Do not import from combined/."""
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed"
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

NY = ZoneInfo("America/New_York")

BLOOMBERG = RAW / "Bloomberg Economic Releases.xlsx"
ASSETS = {
    "ES": dict(tick=0.25, point_value=50),
    "NQ": dict(tick=0.25, point_value=20),
    "ZN": dict(tick=1 / 64, point_value=1000),
}
CLEAN_BARS_FMT = str(PROCESSED / "futures_1min_clean_v3_{symbol}.parquet")
PANEL = OUT / "features.parquet"

# ---------------------------------------------------------------- sessions / split (unchanged)
SESSION_SHIFT = pd.Timedelta(hours=6)
TRAIN_YEARS_SPAN = (2013, 2017)
VALID_YEARS_SPAN = (2018, 2020)
TEST_YEARS_SPAN = (2021, 2026)

# ---------------------------------------------------------------- event games
# Waiting time = first-entry lag after the print. The five 30-minute holds stay
# back-to-back, so the ladder is T+w, T+w+30, ..., T+w+120.
WAIT_GRID_MIN = [1, 2, 5]
PATH_PRE_GRID_MIN = [30, 45, 60]   # path length = path_pre + wait, ending at entry
N_ANCHORS = 5
HOLD_MIN = 30
PATH_FREQ_MIN = 1
LEAK_TEST_MINUTES = 0

# ---------------------------------------------------------------- dtw
DTW_GRID = {"k": [10, 15, 25], "lookback_years": [2, 3, 5]}
MIN_NEIGHBOURS = 20
MAX_POOL = 600
WARP_MINUTES = 5
RECENCY_HALFLIFE_DAYS = 730

# Replacements for the neighbour weight that produces dtw_dir / dtw_disp.
# Neighbours are still the k nearest by DTW distance; only the aggregation changes.
#   inv_dist_recency     — combined/ baseline: 1/d × exp(−age / 730)
#   inv_dist             — shape only
#   inv_dist_sq_recency  — more peaked on the nearest neighbour
#   recency              — equal among k except for age
#   uniform              — plain k-NN mean
#   softmax_neg_dist     — softmax(−d / median d), scale-free in d
#   rank_recency         — 1/rank × recency, robust to DTW scale
WEIGHTING_GRID = [
    "inv_dist_recency",
    "inv_dist",
    "inv_dist_sq_recency",
    "recency",
    "uniform",
    "softmax_neg_dist",
    "rank_recency",
]

# ---------------------------------------------------------------- universe (unchanged)
FAMILY_OVERLAP = 0.80
MIN_RELEASES = 30
IMPACT_MIN_T = 2.0
IMPACT_WINDOW_MIN = 30

# ---------------------------------------------------------------- backtest
COST_TICKS_PER_SIDE = {"gross": 0.0, "quarter tick": 0.25, "half tick": 0.5}
MAX_POSITION = 2.0
GATE_GRID = [0.2, 0.3, 0.4]
SIZING_GRID = ["flat", "inverse_disp"]


def anchors_for_wait(wait_min: int) -> list[int]:
    return [wait_min + HOLD_MIN * i for i in range(N_ANCHORS)]


def path_len_minutes(path_pre_min: int, wait_min: int) -> int:
    """Number of 1-minute path points ending at entry. 30+1 → 31, matching combined/."""
    return path_pre_min + wait_min
