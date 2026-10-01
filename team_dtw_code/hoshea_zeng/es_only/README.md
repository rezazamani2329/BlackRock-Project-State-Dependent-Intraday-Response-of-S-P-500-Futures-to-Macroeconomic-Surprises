# hoshea_zeng / es_only

Jack's **ES-only** event book, with the three construction axes that were not
swept on the Sharpe-0.61 PPT result.

The traded rule is held fixed (so the grid is those three knobs, not a second
gate/sizing search):

- sign of `dtw_dir`, skip the middle 40% (`TAIL_Q = 0.30`)
- size inversely with `dtw_disp`, cap 2 contracts
- k = 15, 3-year lookback, Sakoe-Chiba 5 min
- expanding walk-forward from 2018, **a quarter tick** per side

| Axis | Jack's ES book | Grid here |
|---|---|---|
| Waiting time | 1 min; `T+1 … T+121` | `{1, 2, 5}` |
| DTW path | 31 points ending at entry | `path_pre ∈ {30, 45, 60}`, length = `path_pre + wait` |
| Neighbour weights | `1/d × exp(−age/730)` | seven schemes, same as `combined_modified_hoshea` |

```bash
.venv\Scripts\python.exe hoshea_zeng/es_only/run_es_grid.py
uv run jupytext --sync hoshea_zeng/es_only/run_es_grid.py
```

Outputs: `output/es_only_grid.csv`, `figures/*.png`.
