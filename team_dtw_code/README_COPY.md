# Team DTW code (copy)

A copy of the team's code and results from the BlackRock industry project repository
(`jack-duncan/blackrock-intraday`), without the papers (see `../papers/`), the market data and the API key.

* `combined/` — the DTW event book used in both papers: `pull.py`, `cleaning.py`, `features.py`,
  `backtest.py`, `surprise_dtw_book.py`, `team_book.py`, plus `dtw_book_with_ym.py` and
  `dtw_panel_stats.py` for the E-mini Dow. `DTW_EXTRA=YM` adds YM to the book and writes every
  output to `combined/output/with_YM/`. Results are in `combined/output/`.
* `combined_modified_hoshea/`, `hoshea_zeng/`, `jack/`, `notebooks/` — teammates' versions,
  experiments and notebooks.

Not included: one-minute futures bars (Databento, licensed), the Bloomberg release calendar,
`.parquet` panels and `.env`. To rerun: put a Databento key in `.env` (see `.env.example`), the
Bloomberg file in `data/raw/`, then `uv sync` and run `combined/pull.py --pull`, `cleaning.py`,
`features.py` and `backtest.py` in order.

Code by Jack Duncan, Hoshea Zeng, Aaryen Mehta and Reza Zamani.
