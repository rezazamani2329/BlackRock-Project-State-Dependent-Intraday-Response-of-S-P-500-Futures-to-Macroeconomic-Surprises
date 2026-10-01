# Papers

Two drafts written from the results in this repository and the team's DTW pipeline.

| Folder | PDF | What it is |
|---|---|---|
| `journal_paper/` | `main_journal_draft.pdf` (~59 pp) | *Surprises, Shapes or Machines? Forecasting the Market's Response to Macroeconomic Releases with Dynamic Time Warping (DTW)*. Full paper: price discovery after releases, the surprise, DTW, game and ML approaches on ES, NQ, YM and ZN, robustness, directions for future research, and an appendix with figures and tables from notebooks 02-07. |
| `dtw_structured_paper/` | `main_structured_second_draft.pdf` (13 pp) | Short paper on the DTW event book in the team outline: Introduction, Sensibility, Data (incl. the DTW event panel), Model, Backtest (incl. comparison with the surprise strategy), Conclusion. |

Authors: Benjamin Steel (BlackRock), Reza Zamani, Jack Duncan, Aaryen Mehta and Hoshea Zeng.

Build either paper from its folder with `latexmk -pdf main.tex`. Figures are in each folder's `figures/`.
Numbers come from the notebooks in `notebooks/` (surprise books, all four markets) and from the
team DTW pipeline (`combined/` in the team repository, run with `DTW_EXTRA=YM` for the E-mini Dow).
