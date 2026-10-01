# blackrock-intraday

Research into an intraday S&P futures trading strategy built around dynamic time warping (DTW) — using DTW-based similarity between historical intraday price paths and the current session to find analog days/patterns and inform trade signals.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This creates `.venv/` and installs everything from `pyproject.toml` (pinned in `uv.lock`).

To run/execute notebooks from the command line (not just open them in Jupyter), register the venv as a Jupyter kernel once:

```bash
uv run python -m ipykernel install --user --name blackrock-intraday --display-name "blackrock-intraday"
```

To pull new data via `notebooks/getting_futures_data.py`, copy `.env.example` to `.env` and fill in your Databento API key:

```bash
cp .env.example .env
```

## Structure

## Contributors

- **Reza** — Macro-surprise strategy on ES, NQ and ZN: event study, pre-event state conditioning, walk-forward trading, multi-release scaling, surprise + DTW, and a team comparison at equal costs.  
  Research, notebooks and results: [`Reza/`](./Reza/) (see [`Reza/README.md`](./Reza/README.md))


```
data/          raw and processed S&P futures data (gitignored, see data/README.md)
notebooks/     analysis notebooks, each paired with a companion .py script (see notebooks/README.md)
CLAUDE.md      project context + the notebook authoring workflow, for Claude Code
pyproject.toml uv-managed dependencies, incl. jupytext pairing config
```

## Working in notebooks

Notebooks are edited as paired `.py` files (jupytext, `py:percent` format) and synced to `.ipynb`. Run everything through `uv run`:

```bash
uv run jupytext --sync notebooks/<name>.py     # sync an existing pair after editing the .py
uv run jupyter lab                             # open notebooks/ in Jupyter
```

See `CLAUDE.md` for the full authoring workflow.

## Dependencies of note
- `dtaidistance` — DTW implementations (incl. fast C bindings) used for time-series alignment/similarity.
- `jupytext` — keeps each notebook's `.py` and `.ipynb` in sync.
