# combined_modified_hoshea

An editable copy of `combined/`. Nothing in that folder was changed.

The extra work here is a grid over three construction choices. Every combo is scored on train and validation and written to CSV.

## What is searched (vs `combined/`)

| Axis | Fixed in `combined/` | Grid here |
|---|---|---|
| Waiting time `w` | 1 min; anchors `T+1, T+31, …, T+121` | `{1, 2, 5}`; anchors `T+w, T+w+30, …, T+w+120` |
| DTW path | 30 min + 1 min wait = 31 points, ending at entry | `path_pre ∈ {30, 45, 60}`; length = `path_pre + w` |
| `dtw_dir` weights | `1/d × exp(−age / 730)` | the seven schemes below |

`k` and lookback stay `{10, 15, 25} × {2, 3, 5}`. Family merge, the impact screen, and the 30-minute hold are unchanged.

### Anchors and paths

`w = 2` → entries at `T+2, T+32, T+62, T+92, T+122`, still holding 30 minutes each.

The path always ends on the entry minute and is z-scored before DTW:

- `path_pre=30, w=1` → 31 points (same as `combined/`)
- `path_pre=30, w=2` → 32 points
- `path_pre=45, w=1` → 46 points
- `path_pre=60, w=5` → 65 points

### Neighbour weightings

The k nearest neighbours are still chosen by DTW distance. Only the weights that turn their forward returns into `dtw_dir` / `dtw_disp` change.

| Name | Weights |
|---|---|
| `inv_dist_recency` | `1/d × exp(−age/730)`, the `combined/` formula |
| `inv_dist` | `1/d` only |
| `inv_dist_sq_recency` | `1/d² × recency`, more mass on the nearest neighbour |
| `recency` | age only; equal shape weight among the k |
| `uniform` | plain k-NN mean |
| `softmax_neg_dist` | `softmax(−d / median d)` |
| `rank_recency` | `1/rank × recency`, invariant to the scale of DTW distance |

## How to run

Uses the same cleaned bars and Bloomberg file as `combined/`:
`data/processed/futures_1min_clean_v3_{ES,NQ,ZN}.parquet` and
`data/raw/Bloomberg Economic Releases.xlsx`.

```bash
# Full grid (9 path constructions × 3 assets). Safe to interrupt and rerun.
.venv\Scripts\python.exe combined_modified_hoshea/hyperparam_search.py

# Smoke test: ES only, baseline path
.venv\Scripts\python.exe combined_modified_hoshea/hyperparam_search.py --assets ES --waits 1 --path-pres 30
```

Outputs:

- `output/hp_search_by_asset.csv` — one row per asset and combo
- `output/hp_search_pooled.csv` — equal-weighted mean across the three assets, sorted by train dir IC
- `output/hp_search_chosen.csv` — the combo with the highest train pooled dir IC

Columns: `train_dir_ic` / `valid_dir_ic`, `disp_ic`, `hit` (sign hit rate), `q51` (top minus bottom dir quintile of the forward return). **Test (2021–2026) is not in these tables.**

To materialise a panel and backtest after picking a combo:

```bash
.venv\Scripts\python.exe combined_modified_hoshea/build_panel.py
# or pass a combo explicitly
.venv\Scripts\python.exe combined_modified_hoshea/build_panel.py --wait 2 --path-pre 45 --weighting uniform --k 15 --lookback 3
.venv\Scripts\python.exe combined_modified_hoshea/backtest.py
```

`hyperparam_search.py` and `backtest.py` are jupytext pairs. After editing the `.py`:

```bash
uv run jupytext --sync combined_modified_hoshea/hyperparam_search.py
uv run jupytext --sync combined_modified_hoshea/backtest.py
```

## Files

```
combined_modified_hoshea/
  config.py              grids and paths
  engine.py              families, games, DTW, weights
  hyperparam_search.py   runs the grid, writes CSVs
  build_panel.py         writes features.parquet for one combo
  backtest.py            copy of the combined/ backtest
  output/                search tables and panels
```
