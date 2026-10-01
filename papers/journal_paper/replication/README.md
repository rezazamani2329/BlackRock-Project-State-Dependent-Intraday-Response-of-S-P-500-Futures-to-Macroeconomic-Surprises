# Replication package (AEA Data and Code Availability Policy) — skeleton

Follow the AEA Data Editor's README template when the paper is accepted.

## Data availability statement
- **Futures prices:** Databento, GLBX.MDP3 one-minute OHLCV for ES, NQ, YM, ZN (licensed; describe how
  a replicator obtains access and the cost).
- **Release calendar and consensus forecasts:** Bloomberg Economic Releases (licensed terminal
  data; cannot be redistributed; list the fields and the export procedure).
- Cleaned derived files that may be shared: `data/processed/futures_1min_clean_v3_{ES,NQ,YM,ZN}.parquet`
  (check the licence before including).

## Computational requirements
- Python 3.12, `uv` environment (`pyproject.toml`, `uv.lock`); runtime per notebook.

## Instructions and table/figure map
| Output | Program |
|---|---|
| Figures on price discovery | `Reza/notebooks/02*`, `03*` (YM: `02b_*_YM`, `03b_*_YM`); `scripts/make_paper_figures_ym.py` for Figures 1-2 |
| Table (entry assumption) | `Reza/notebooks/08` |
| Combined surprise/DTW figure | `combined/surprise_dtw_book.py` |
| Strategy and portfolio tables | `combined/team_book.py` |
