"""ES-only event book: Jack's traded rule, plus the three construction axes."""
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed"
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
FIG = HERE / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

NY = ZoneInfo("America/New_York")
BLOOMBERG = RAW / "Bloomberg Economic Releases.xlsx"
ASSETS = {"ES": dict(tick=0.25, point_value=50)}
CLEAN_BARS_FMT = str(PROCESSED / "futures_1min_clean_v3_{symbol}.parquet")

SESSION_SHIFT = pd.Timedelta(hours=6)
TRAIN_YEARS_SPAN = (2013, 2017)
VALID_YEARS_SPAN = (2018, 2020)
TEST_YEARS_SPAN = (2021, 2026)
OOS_START_YEAR = 2018

WAIT_GRID_MIN = [1, 2, 5]
PATH_PRE_GRID_MIN = [30, 45, 60]
N_ANCHORS = 5
HOLD_MIN = 30
PATH_FREQ_MIN = 1
LEAK_TEST_MINUTES = 0

# Jack's ES book used these for the traded rule. The search varies only the three axes.
K_NEIGHBOURS = 15
LOOKBACK_YEARS = 3
MIN_NEIGHBOURS = 20
MAX_POOL = 600
WARP_MINUTES = 5
RECENCY_HALFLIFE_DAYS = 730
TAIL_Q = 0.30                 # do not trade the middle 40%
SIZING = "inverse_disp"
MAX_POSITION = 2.0
COST_TICKS = 0.25             # PPT / design.md net Sharpe is at a quarter tick
TICK = 0.25

WEIGHTING_GRID = [
    "inv_dist_recency",
    "inv_dist",
    "inv_dist_sq_recency",
    "recency",
    "uniform",
    "softmax_neg_dist",
    "rank_recency",
]

FAMILY_OVERLAP = 0.80
MIN_RELEASES = 30
IMPACT_MIN_T = 2.0
IMPACT_WINDOW_MIN = 30


def anchors_for_wait(wait_min: int) -> list[int]:
    return [wait_min + HOLD_MIN * i for i in range(N_ANCHORS)]


def path_len_minutes(path_pre_min: int, wait_min: int) -> int:
    return path_pre_min + wait_min
