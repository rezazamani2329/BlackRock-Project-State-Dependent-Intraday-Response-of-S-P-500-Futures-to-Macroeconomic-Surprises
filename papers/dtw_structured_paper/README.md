# Paper: DTW event book (AER format)

*Surprises, Shapes or Machines? Forecasting the Market's Response to Macroeconomic Releases with Dynamic Time Warping (DTW)*

Authors: Benjamin Steel (BlackRock, first author), Jack Duncan, Aaryen Mehta, Reza Zamani,
Hoshea Zeng (Haas School of Business, UC Berkeley). The views are the authors' own, not those of
BlackRock, Inc. or UC Berkeley.

Structure: Abstract; I Sensibility (DTW, events, assets); II Data (Bloomberg events, futures);
III Model (event universe, signal, strategy: gating and weighting); IV Backtest; V Conclusion.

Build: `latexmk -pdf main.tex`. Figures: `python3 make_figs.py` (the DTW illustration uses
stylized paths; the result charts are drawn from the numbers in `combined/README.md` and the
`combined-iteration` notebooks). `fig01_markets_overview.png` and
`surprise_dtw_book_cumulative.png` come from Reza's and the combined notebooks.

Numbers come from `combined/` on the `combined-iteration` branch (config, features.py,
backtest.py) and `surprise_dtw_book.py`.
