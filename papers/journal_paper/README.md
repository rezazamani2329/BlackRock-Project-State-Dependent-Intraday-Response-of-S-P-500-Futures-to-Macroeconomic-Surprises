# Paper: submission draft in AER format

*Surprises, Shapes or Machines? Forecasting the Market's Response to Macroeconomic Releases with Dynamic Time Warping (DTW)*

Authors: Benjamin Steel (BlackRock, first author), Jack Duncan, Aaryen Mehta, Reza Zamani,
Hoshea Zeng. The paper represents the authors' views only, not those of BlackRock or UC Berkeley.

## AER requirements built into this draft
- PDF; 12-pt font, 1.5 spacing, 1-inch margins; aim for about 40-45 pages.
- Title page with authors, affiliations, acknowledgement footnote and an abstract of 100 words or
  fewer, plus JEL codes and keywords.
- Roman-numeral sections (I, II, ...) with lettered subsections, literature review inside the
  introduction, Chicago author-date references (uses `aea.bst` if installed, else `chicago.bst`,
  else `apalike`).
- In-paper appendix for essential technical material only; everything else in
  `online_appendix/online_appendix.tex` (separate PDF; the paper must stand alone without it).
- **A separate Disclosure Statement PDF from each of the five authors** (`disclosure/`).
- **AEA Data and Code Availability Policy**: replication package skeleton in `replication/`.
  Bloomberg and Databento data are licensed; the data availability statement must say how a
  replicator obtains them.

Check the current AER submission guidelines before submitting:
https://www.aeaweb.org/journals/aer/submissions/guidelines

## Build
```bash
cd paper
latexmk -pdf main.tex
cd online_appendix && latexmk -pdf online_appendix.tex
```
Red `[TODO]` and blue `[Source]` notes disappear when `\drafttrue` is set to `\draftfalse`.

## Structure
```
main.tex                    title page, abstract, section includes
sections/00_abstract.tex    abstract (<=100 words), JEL, keywords
sections/01_introduction    question, findings, related literature, roadmap
sections/02_setting_data    releases and consensus, futures data, surprise construction, sample
sections/03_price_discovery jump vs drift, surprises predict the drift, pre-release state
sections/04_exploitability  common protocol and the four forecasting approaches
sections/05_results         performance, costs, combination, entry assumption, diversification
sections/06_mechanisms      tick size, analogue instability, speed of price discovery
sections/07_robustness      2022, construction choices, windows, multiple testing
sections/08_conclusion
sections/A_appendix         surprise construction, DTW forecast (in-paper, essential only)
online_appendix/            supplemental appendix (separate PDF)
disclosure/                 disclosure-statement template (one PDF per author)
replication/                replication-package README skeleton
tables/, figures/, references.bib
```

## Other journals with the same structure
AER: Insights (max 6,000-7,000 words and 5 exhibits), Journal of Finance, Journal of Financial
Economics, Review of Financial Studies, Journal of Financial and Quantitative Analysis. The
section structure transfers with minor changes to each journal's length and style rules.
