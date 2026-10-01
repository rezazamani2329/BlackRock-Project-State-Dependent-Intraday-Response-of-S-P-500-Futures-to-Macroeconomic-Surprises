"""Item 3: the state interaction Reza's title promises.

`Reza/` is titled "State-Dependent Intraday Response of S&P 500 Futures to
Macroeconomic Surprises" and measures the response but never the state dependence.
This adds it, and fixes the entry timing at the same time: Reza's returns run from
T-1, so they span the announcement jump and are not a forecast of anything. Here the
jump and the drift are measured separately.

  jump  = log open(T+1)  / open(T)     -- the reaction. Not executable: open(T) is the
                                          pre-release price.
  drift = log open(T+31) / open(T+1)   -- a 30-minute hold entered one minute after the
                                          print, which is executable.

Prices are bar *opens*, not closes. Bars are left-labelled, so close at index T already
covers [T, T+1) and would silently shift every window by a minute.

Four candidate states, all measured strictly before the release:

  1. VIX      -- prior settlement close, merged as-of its 16:15 ET publication instant.
  2. rv_4h    -- realised volatility of ES over the four hours before the release.
  3. slope    -- VIX3M / VIX, the term structure, as a risk-appetite proxy.
  4. reaction -- the residual of the jump on the surprise, signed so that positive
                 means the initial move already went *further* in the trade's direction
                 than the surprise implied (an over-reaction) and negative means it
                 under-shot. Measured at T+1 rather than pre-release, which is still
                 executable because the drift window opens at T+1.

For each release and each state, the drift is regressed on the surprise, the state,
and their interaction, and separately split into terciles of the state.

Run: uv run python jack/state_dependence.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
ES_FILE = ROOT / "data" / "processed" / "es_1min_clean.parquet"
VIX_FILE = ROOT / "data" / "raw" / "vix_daily.parquet"
BLOOMBERG_FILE = ROOT / "data" / "raw" / "Bloomberg Economic Releases.xlsx"
WINSOR_SIGMA = 5.0
TICK_BPS = 0.5

RELEASES = {
    "CPI": [("CPI MoM", "08:30:00"), ("Core CPI MoM", "08:30:00")],
    "NFP": [("Change in Nonfarm Payrolls", "08:30:00")],
    "PCE": [("Core PCE Price Index MoM", "08:30:00")],
}
# sign that turns the surprise into a trade direction (hot inflation -> short)
TRADE_SIGN = {"CPI": -1, "NFP": +1, "PCE": -1}

es = pd.read_parquet(ES_FILE, columns=["open", "close"])
es = es[~es.index.duplicated(keep="first")].sort_index()
ES_IDX, ES_OPEN, ES_CLOSE = es.index, es["open"].to_numpy(), es["close"].to_numpy()


def px(ts: pd.Timestamp, tol=pd.Timedelta(minutes=5)) -> float:
    """Price at the instant `ts` — the open of the bar labelled `ts`."""
    pos = ES_IDX.searchsorted(ts)
    if pos >= len(ES_IDX) or abs(ES_IDX[pos] - ts) > tol:
        return np.nan
    return ES_OPEN[pos]


def rv_4h(ts: pd.Timestamp) -> float:
    lo = ES_IDX.searchsorted(ts - pd.Timedelta(hours=4), "left")
    hi = ES_IDX.searchsorted(ts, "left")
    if hi - lo < 60:
        return np.nan
    return float(np.std(np.diff(np.log(ES_CLOSE[lo:hi]))) * 1e4)


def load_bloomberg() -> pd.DataFrame:
    sheets = pd.ExcelFile(BLOOMBERG_FILE)
    frames = []
    for sheet in sheets.sheet_names:
        part = pd.read_excel(BLOOMBERG_FILE, sheet_name=sheet)
        part.columns = [c.split(".")[-1] if "DROPNA" in str(c) else c for c in part.columns]
        frames.append(part)
    b = pd.concat(frames, ignore_index=True)
    b["RELEASE_DATE"] = pd.to_datetime(b.RELEASE_DATE, errors="coerce")
    b["RELEASE_TIME"] = b.RELEASE_TIME.astype(str)
    b = b.dropna(subset=["RELEASE_DATE"])
    return b[b.SURVEY_MEDIAN.notna() & b.ACTUAL.notna()]


def surprise_series(b: pd.DataFrame, name: str, time: str) -> pd.DataFrame:
    g = b[(b.EVENT_NAME == name) & (b.RELEASE_TIME == time)].sort_values("RELEASE_DATE").copy()
    g["surprise"] = g.ACTUAL - g.SURVEY_MEDIAN
    scale = g.surprise.expanding().std().shift(1)
    g["z"] = (g.surprise / scale).clip(-WINSOR_SIGMA, WINSOR_SIGMA)
    g = g[np.isfinite(g.z) & (scale > 0)]
    stamp = pd.to_datetime(g.RELEASE_DATE.dt.strftime("%Y-%m-%d") + " " + time)
    g["ts"] = (stamp.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
               .dt.tz_convert("UTC"))
    return g.dropna(subset=["ts"])[["ts", "z"]].set_index("ts")


def build_panel(family: str, b: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    parts = [surprise_series(b, name, time).rename(columns={"z": f"z{i}"})
             for i, (name, time) in enumerate(RELEASES[family])]
    z = pd.concat(parts, axis=1).dropna()
    z["z"] = z.mean(axis=1)  # headline and core publish together; average rather than double-count
    rows = []
    for ts, r in z.iterrows():
        p0 = px(ts)                                   # pre-release
        p1 = px(ts + pd.Timedelta(minutes=1))          # one minute after the print
        p31 = px(ts + pd.Timedelta(minutes=31))        # 30-minute hold from there
        if not np.isfinite([p0, p1, p31]).all():
            continue
        prior = vix[vix.published_utc < ts]
        if prior.empty:
            continue
        v = prior.iloc[-1]
        rows.append({"ts": ts, "z": r.z,
                     "jump": 1e4 * np.log(p1 / p0),
                     "drift": 1e4 * np.log(p31 / p1),
                     "vix": v.vix, "slope": v.vix3m / v.vix if v.vix3m > 0 else np.nan,
                     "rv_4h": rv_4h(ts)})
    d = pd.DataFrame(rows)
    d = d[d.z != 0].dropna(subset=["vix", "rv_4h", "jump", "drift"]).reset_index(drop=True)
    # Reaction residual, signed along the trade direction: positive = the initial move
    # already over-shot what the surprise implied, negative = it under-shot.
    fit = sm.OLS(d.jump, sm.add_constant(d[["z"]])).fit()
    direction = TRADE_SIGN[family] * np.sign(d.z)
    d["reaction"] = direction * fit.resid
    d["trade_bps"] = direction * d.drift
    return d


def interaction_table(d: pd.DataFrame, family: str) -> None:
    print(f"\n{family}  n={len(d)}   drift = a + b*z + c*state + d*(z x state),  HC1 errors")
    print(f"  {'state':<10} {'b (bps/sig)':>12} {'t':>7} {'d (interaction)':>17} {'t':>7} {'R2':>7}")
    for state in ("vix", "rv_4h", "slope", "reaction"):
        sub = d.dropna(subset=[state, "drift", "z"])
        s = (sub[state] - sub[state].mean()) / sub[state].std()
        X = sm.add_constant(pd.DataFrame({"z": sub.z.to_numpy(), "state": s.to_numpy(),
                                          "z_x_state": (sub.z * s).to_numpy()}))
        m = sm.OLS(sub.drift.to_numpy(), X).fit(cov_type="HC1")
        print(f"  {state:<10} {m.params['z']:>12.2f} {m.tvalues['z']:>+7.2f} "
              f"{m.params['z_x_state']:>17.2f} {m.tvalues['z_x_state']:>+7.2f} {m.rsquared:>7.3f}")


def tercile_table(d: pd.DataFrame, family: str) -> None:
    print(f"\n{family}  sign trade on the drift window (T+1 to T+30), by tercile of each state")
    print(f"  {'state':<10} {'tercile':<8} {'n':>4} {'gross bps':>10} {'net':>7} {'hit':>7} {'t':>7}")
    for state in ("vix", "rv_4h", "slope", "reaction"):
        g = d.dropna(subset=[state]).copy()
        g["_t"] = pd.qcut(g[state], 3, labels=["low", "mid", "high"])
        for label in ("low", "mid", "high"):
            sub = g[g._t == label].trade_bps.to_numpy()
            if len(sub) < 10:
                continue
            t = sub.mean() / (sub.std(ddof=1) / np.sqrt(len(sub)))
            print(f"  {state:<10} {label:<8} {len(sub):>4} {sub.mean():>10.2f} "
                  f"{sub.mean()-TICK_BPS:>7.2f} {(sub > 0).mean():>6.1%} {t:>+7.2f}")
        print()


def robustness(d: pd.DataFrame, family: str, state: str) -> None:
    """Anything clearing |t| > 3 gets checked against outliers and a 2020 split."""
    sub = d.dropna(subset=[state]).copy()
    def fit(x):
        s = (x[state] - x[state].mean()) / x[state].std()
        X = sm.add_constant(pd.DataFrame({"z": x.z.to_numpy(), "state": s.to_numpy(),
                                          "z_x_state": (x.z * s).to_numpy()}))
        m = sm.OLS(x.drift.to_numpy(), X).fit(cov_type="HC1")
        return m.params["z_x_state"], m.tvalues["z_x_state"], len(x)
    trimmed = sub.loc[sub.drift.abs().sort_values().index[:-3]]
    early, late = sub[sub.ts < "2021-01-01"], sub[sub.ts >= "2021-01-01"]
    print(f"  {family} x {state}:")
    for label, x in (("all", sub), ("drop 3 largest |drift|", trimmed),
                     ("<=2020", early), ("2021+", late)):
        if len(x) < 25:
            print(f"    {label:<24} n={len(x):>4}  (too few)")
            continue
        c, t, n = fit(x)
        print(f"    {label:<24} n={n:>4}  interaction {c:>+7.2f}  t {t:>+6.2f}")


def main() -> None:
    b = load_bloomberg()
    vix = pd.read_parquet(VIX_FILE).dropna(subset=["vix"])
    panels = {}
    print("=" * 80)
    print("Jump vs drift — separating the reaction from the forecast")
    print("=" * 80)
    print(f"  {'family':<6} {'n':>4} {'jump b':>9} {'t':>7} {'R2':>6} | {'drift b':>9} {'t':>7} {'R2':>6}")
    for family in RELEASES:
        d = build_panel(family, b, vix)
        panels[family] = d
        X = sm.add_constant(d[["z"]])
        mj = sm.OLS(d.jump, X).fit(cov_type="HC1")
        md = sm.OLS(d.drift, X).fit(cov_type="HC1")
        print(f"  {family:<6} {len(d):>4} {mj.params.z:>9.2f} {mj.tvalues.z:>+7.2f} {mj.rsquared:>6.3f} | "
              f"{md.params.z:>9.2f} {md.tvalues.z:>+7.2f} {md.rsquared:>6.3f}")

    print("\n" + "=" * 80)
    print("STATE INTERACTIONS on the drift window")
    print("=" * 80)
    for family, d in panels.items():
        interaction_table(d, family)

    print("\n" + "=" * 80)
    print("TERCILE SPLITS — the tradeable version")
    print("=" * 80)
    for family, d in panels.items():
        tercile_table(d, family)

    print("\n" + "=" * 80)
    print("ROBUSTNESS on every interaction clearing |t| > 3")
    print("=" * 80)
    for family, d in panels.items():
        for state in ("vix", "rv_4h", "slope", "reaction"):
            x = d.dropna(subset=[state])
            s_ = (x[state] - x[state].mean()) / x[state].std()
            X = sm.add_constant(pd.DataFrame({"z": x.z.to_numpy(), "state": s_.to_numpy(),
                                              "z_x_state": (x.z * s_).to_numpy()}))
            m = sm.OLS(x.drift.to_numpy(), X).fit(cov_type="HC1")
            if abs(m.tvalues["z_x_state"]) > 3:
                robustness(d, family, state)

    out = pd.concat([d.assign(family=f) for f, d in panels.items()], ignore_index=True)
    out.to_parquet(ROOT / "jack" / "data" / "state_dependence_panel.parquet", index=False)
    print(f"saved panel -> {ROOT/'jack'/'data'/'state_dependence_panel.parquet'}")


if __name__ == "__main__":
    main()
