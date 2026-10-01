# notebooks/

Every notebook here is paired with a companion `.py` file of the same name via [jupytext](https://jupytext.readthedocs.io/) (`py:percent` format, configured in the root `pyproject.toml`).

**The `.py` file is the source of truth.** Write and edit code there; the `.ipynb` is a synced, runnable view of it. See `CLAUDE.md` at the repo root for the exact workflow — that file governs how notebooks get created and updated in this project, so it isn't repeated here.

## Layout
Plain descriptive `snake_case` names, no numeric prefix — e.g. `data_exploration.py`, `dtw_baseline.py`. Name each notebook after what it does, not its position in a sequence.

- `data_cleaning.py` — writes the cleaned ES OHLCV parquet to `data/processed/` (raw -> processed).
- `macro_event_calendar.py` — a second, clearly-named cleaning notebook; writes the macro event calendar CSV to `data/processed/`. It builds an independent derived dataset (event timestamps, not price data), so it doesn't belong in `data_cleaning.py`.
- Every other notebook only reads from `data/processed/`; it never writes/derives data files itself. See `CLAUDE.md` for the full rule.
