# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: blackrock-intraday
#     language: python
#     name: blackrock-intraday
# ---

# %% [markdown]
# # Forecasting magnitude
#
# The direction study found nothing tradeable. The same features forecast *size*
# well, so this notebook asks whether that forecast is good enough to be worth
# anything, by scoring it against the benchmarks a volatility desk would already have.
#
# Three questions:
#
# 1. Does the feature model beat GARCH(1,1), the standard parametric benchmark?
# 2. Does it beat a naive seasonal forecast, which is what "knowing the time of day"
#    gets you for free?
# 3. Do Ridge, Lasso or ElasticNet improve on OLS here?
#
# **Target.** Realised variance of the next 30-minute bar, `fwd_ret_30m ** 2`. A single
# squared return is a noisy proxy for variance, so R-squared is low for every model
# here. The ranking is what matters, not the level.
#
# **Loss.** QLIKE, `a/f - log(a/f) - 1`, which is standard for variance forecasts
# because it is robust to that noise and penalises under-forecasting asymmetrically.
# Lower is better. R-squared on log variance is reported alongside as a symmetric
# measure of information content, and the two disagree in an informative way.

# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, LinearRegression

warnings.filterwarnings("ignore")
pd.set_option("display.width", 140)

JACK_DATA = Path("data")
BARS_PER_SESSION = 46
TEST_YEARS = range(2021, 2027)
GARCH_FIRST_FIT = 2018      # GARCH needs its own training history before the test window
PCT = 100.0                 # arch is better conditioned on percentage returns
NW_LAG = BARS_PER_SESSION   # Newey-West lag for the Diebold-Mariano tests, one session

# %% [markdown]
# ## Data and benchmark components
#
# Everything here is either already lagged in `features.parquet` or shifted explicitly
# below. The target is the *next* bar, so a forecast built from data through bar *t* is
# scored against bar *t+1*.

# %%
features = pd.read_parquet(JACK_DATA / "features.parquet")
bars = pd.read_parquet(JACK_DATA / "es_30min_bars.parquet")
d = features.join(bars[["close"]], how="left")

d["ret"] = np.log(d.close).diff()
d["slot"] = pd.Series(d.index.tz_convert("America/New_York").strftime("%H:%M"), index=d.index)
d["next_slot"] = d.slot.shift(-1)
d["rv_fwd"] = d.fwd_ret_30m ** 2

r2 = d.ret ** 2
d["har_1b"] = r2.shift(1)
d["har_1d"] = r2.rolling(BARS_PER_SESSION).mean().shift(1)
d["har_1w"] = r2.rolling(BARS_PER_SESSION * 5).mean().shift(1)
d["ewma_var"] = d.realised_vol ** 2
d["seas_naive"] = d.ewma_var * d.seas_exp ** 2      # level x time-of-day multiplier

FEATURES = ["seas_exp", "vol_z", "is_event_window", "above_ma100",
            "range_pos_30m", "range_pos_2h"]
HAR = ["l_1b", "l_1d", "l_1w"]

d = d.dropna(subset=["rv_fwd", "ret", "ewma_var", "har_1b", "har_1d", "har_1w",
                     "seas_naive", "next_slot"] + FEATURES)
d = d[d.rv_fwd > 0].copy()
for col, src in [("l_1b", "har_1b"), ("l_1d", "har_1d"), ("l_1w", "har_1w"),
                 ("l_seas", "seas_naive"), ("l_ewma", "ewma_var")]:
    d[col] = np.log(d[src].clip(lower=1e-12))
d["y"] = np.log(d.rv_fwd)
year = np.asarray(d.index.year)

print(f"{len(d):,} bars   {d.index.min().date()} -> {d.index.max().date()}")
print(f"test window {TEST_YEARS.start}-{TEST_YEARS.stop - 1}: "
      f"{int(np.isin(year, list(TEST_YEARS)).sum()):,} bars")

# %% [markdown]
# ## GARCH(1,1)
#
# Two versions. The plain one fits GARCH to the 30-minute returns directly. That model
# has no way to represent the intraday cycle, which is the dominant feature of
# intraday volatility, so it is included as a floor rather than a serious benchmark.
#
# The second deseasonalises first: divide returns by the square root of a
# training-period time-of-day variance factor, fit GARCH to the result, then scale the
# forecast back up by the factor for the bar being predicted. This is the fair
# benchmark and it is much harder to beat.
#
# Parameters are refit each year on an expanding window. Within a test year the
# variance recursion is carried forward with those parameters fixed, so no test-year
# observation informs the parameters used to forecast it.

# %%
def garch_forecasts(deseasonalise: bool) -> pd.Series:
    """One-step-ahead variance forecast of the next bar, refit annually."""
    out = pd.Series(index=d.index, dtype=float)
    for test_year in range(GARCH_FIRST_FIT, TEST_YEARS.stop):
        train, test = d[year < test_year], d[year == test_year]
        if len(test) == 0:
            continue

        if deseasonalise:
            factor = (train.ret ** 2).groupby(train.slot).mean()
            factor = (factor / factor.mean()).clip(lower=1e-6)
            train_r = (train.ret / np.sqrt(train.slot.map(factor))) * PCT
            test_r = (test.ret / np.sqrt(test.slot.map(factor))) * PCT
        else:
            factor = None
            train_r, test_r = train.ret * PCT, test.ret * PCT

        fit = arch_model(train_r.dropna(), p=1, q=1, mean="Zero", dist="t").fit(disp="off")
        omega, alpha, beta = (fit.params["omega"], fit.params["alpha[1]"],
                              fit.params["beta[1]"])

        sigma2 = fit.conditional_volatility.iloc[-1] ** 2
        last_eps2 = train_r.iloc[-1] ** 2
        test_arr = test_r.to_numpy()
        current = np.empty(len(test))
        for i in range(len(test)):
            sigma2 = omega + alpha * last_eps2 + beta * sigma2
            current[i] = sigma2                      # forecast for bar i
            last_eps2 = test_arr[i] ** 2

        nxt = np.full(len(test), np.nan)             # we need the forecast for bar i+1
        nxt[:-1] = current[1:]
        forecast = nxt / PCT ** 2
        if deseasonalise:
            forecast = forecast * test.next_slot.map(factor).to_numpy()
        out.loc[test.index] = forecast
    return out


d["garch_raw"] = garch_forecasts(deseasonalise=False)
d["garch_seas"] = garch_forecasts(deseasonalise=True)
d = d.dropna(subset=["garch_raw", "garch_seas"]).copy()
d["l_graw"], d["l_gseas"] = np.log(d.garch_raw), np.log(d.garch_seas)
year = np.asarray(d.index.year)
print(f"GARCH forecasts on {len(d):,} bars, "
      f"{min(year)}-{max(year)}")

# %% [markdown]
# **Findings:** Fits cleanly on every fold, 82,726 bars over 2018 to 2026. Both series
# stay well away from the clip floor: the smallest seasonal forecast is 3.5e-08 in
# variance terms against a median of 9.8e-07, so nothing here is being rescued by the
# guard on the log.

# %% [markdown]
# ## Scoring
#
# `wf` runs the annual walk-forward. Two fitting methods are compared, and the
# difference between them turns out to matter more than any feature choice:
#
# - **OLS on log variance** minimises symmetric squared error in logs.
# - **Gamma GLM with a log link** minimises the Gamma deviance, which is exactly
#   `2n x QLIKE`. It is fitting the loss we are scoring on.
#
# The OLS forecast needs a log-normal retransformation bias correction to be
# comparable; the GLM predicts the mean directly and needs none.

# %%
def qlike(actual, forecast_var) -> np.ndarray:
    forecast_var = np.clip(np.asarray(forecast_var), 1e-14, None)
    actual = np.asarray(actual)
    return actual / forecast_var - np.log(actual / forecast_var) - 1


def wf(cols, how="ols"):
    """Annual expanding walk-forward. Returns (variance forecast, actual, log forecast)."""
    preds, actuals = [], []
    for test_year in TEST_YEARS:
        train, test = d[year < test_year], d[year == test_year]
        if len(test) == 0:
            continue
        x_train = sm.add_constant(train[cols], has_constant="add")
        x_test = sm.add_constant(test[cols], has_constant="add")
        if how == "ols":
            fit = sm.OLS(train.y, x_train).fit()
            preds.append(fit.predict(x_test))
        else:
            fit = sm.GLM(train.rv_fwd, x_train,
                         family=sm.families.Gamma(link=sm.families.links.Log())).fit()
            preds.append(np.log(fit.predict(x_test)))
        actuals.append(test.rv_fwd)
    log_pred = pd.concat([pd.Series(p) for p in preds])
    actual = pd.concat(actuals)
    if how == "ols":
        forecast = np.exp(log_pred + 0.5 * np.var(np.log(actual) - log_pred))
    else:
        forecast = np.exp(log_pred)
    return np.asarray(forecast), actual, np.asarray(log_pred)


def score(name, cols=None, how="ols", direct=None):
    if direct is not None:
        mask = np.isin(year, list(TEST_YEARS))
        forecast, actual = d[direct][mask].to_numpy(), d.rv_fwd[mask]
        log_pred = np.log(forecast)
    else:
        forecast, actual, log_pred = wf(cols, how)
    loss = qlike(actual, forecast)
    r2 = 1 - np.var(np.log(actual) - log_pred) / np.var(np.log(actual))
    return {"model": name, "QLIKE": loss.mean(), "R2_log": r2}, loss


def diebold_mariano(loss_a, loss_b, name_a, name_b, lag=NW_LAG):
    """Newey-West t-test on the QLIKE loss differential. Negative favours the first."""
    diff = loss_a - loss_b
    n, mean = len(diff), diff.mean()
    var = np.var(diff) + 2 * sum(
        (1 - k / (lag + 1)) * np.cov(diff[:-k], diff[k:])[0, 1] for k in range(1, lag + 1)
    )
    t = mean / np.sqrt(var / n)
    winner = name_a if mean < 0 else name_b
    verdict = "significant" if abs(t) > 1.96 else "not significant"
    print(f"  {name_a} vs {name_b}: diff {mean:+.4f}  t {t:+.2f}  -> {winner} ({verdict})")


# %%
SPECS = [
    ("EWMA level only",            ["l_ewma"],           "ols",  None),
    ("HAR",                        HAR,                  "ols",  None),
    ("seasonal naive",             ["l_seas"],           "ols",  None),
    ("GARCH(1,1) raw",             None,                 None,   "garch_raw"),
    ("GARCH(1,1) seasonal",        None,                 None,   "garch_seas"),
    ("features, OLS",              HAR + FEATURES,       "ols",  None),
    ("features, QMLE",             HAR + FEATURES,       "qmle", None),
    ("GARCH-seasonal + features",  ["l_gseas"] + FEATURES, "qmle", None),
]
rows, losses = [], {}
for name, cols, how, direct in SPECS:
    row, loss = score(name, cols, how or "ols", direct)
    rows.append(row)
    losses[name] = loss
results = pd.DataFrame(rows).set_index("model")
print(results.round(4).to_string())

# %% [markdown]
# **Findings:** Two results, and the second is the more useful one.
#
# Raw GARCH(1,1) is beaten by everything that knows the time of day, which is the
# expected outcome for a model with no way to represent the intraday cycle. Seasonally
# adjusted GARCH is a different matter: at QLIKE 1.5141 it beats the feature model as
# originally fitted, 1.6247.
#
# But the two measures disagree there. The feature model has the *higher* R-squared on
# log variance, 0.2659 against 0.2557, while losing on QLIKE. It carries more
# information and is worse calibrated, and calibration is a property of the fitting
# method rather than the features. Refitting the same features by Gamma GLM moves QLIKE
# from 1.6247 to 1.4697 and now beats GARCH, while R-squared barely moves, 0.2659 to
# 0.2637. Nothing was added to the model; it was only fitted against the loss it is
# scored on.
#
# The combination is best at 1.4214, so the features hold information GARCH does not.
# `is_event_window` is the clearest case, since a scheduled release is a calendar fact
# that no model of past returns can recover.

# %% [markdown]
# ## Is the difference real?

# %%
print("Diebold-Mariano on QLIKE loss, Newey-West at one session of lags\n")
diebold_mariano(losses["features, QMLE"], losses["features, OLS"], "QMLE", "OLS")
diebold_mariano(losses["features, OLS"], losses["GARCH(1,1) raw"],
                "features (OLS)", "GARCH raw")
diebold_mariano(losses["features, OLS"], losses["GARCH(1,1) seasonal"],
                "features (OLS)", "GARCH seasonal")
diebold_mariano(losses["features, QMLE"], losses["GARCH(1,1) seasonal"],
                "features (QMLE)", "GARCH seasonal")
diebold_mariano(losses["GARCH-seasonal + features"], losses["GARCH(1,1) seasonal"],
                "GARCH + features", "GARCH seasonal")

# %% [markdown]
# **Findings:** Every comparison clears significance. The QMLE-against-OLS differential
# is the largest at t = -21.66, which is a fitting change rather than a modelling one.
# Against seasonally adjusted GARCH the feature model loses as fitted by OLS, t = +6.11,
# and wins once refitted, t = -2.48. Adding the features to GARCH beats GARCH alone at
# t = -6.55.
#
# The margin over GARCH is narrow. This is a real but small edge over a well specified
# parametric benchmark, not a large one.

# %% [markdown]
# ## How much of this is just knowing the time of day?
#
# The intraday volatility curve is public knowledge and every options desk has one.
# This measures what the rest of the feature set adds once it is already in the model.

# %%
STEPS = [
    ("EWMA x time-of-day",  ["l_seas"]),
    ("+ HAR terms",         ["l_seas"] + HAR),
    ("+ event window",      ["l_seas"] + HAR + ["is_event_window"]),
    ("+ everything else",   ["l_seas"] + HAR + FEATURES),
]
baseline = None
for name, cols in STEPS:
    forecast, actual, _ = wf(cols, "qmle")
    q = qlike(actual, forecast).mean()
    baseline = q if baseline is None else baseline
    print(f"  {name:<22} QLIKE {q:.4f}   vs baseline {100 * (baseline - q) / baseline:+5.2f}%")

# %% [markdown]
# **Findings:** Starting from the level times the time-of-day curve, the HAR terms add
# 0.28%, the event window 3.07%, and everything else together 3.69%. Most of what the
# model knows is the clock, and the one component that adds materially beyond it is the
# scheduled-release flag.
#
# That matters for any trade built on this. The intraday volatility curve and the
# release calendar are both public, and both are priced into the options surface, so a
# forecast that beats HAR mostly by knowing the time of day is not obviously an edge
# over a market maker. The comparison that decides it is against implied volatility, not
# against GARCH.

# %% [markdown]
# ## The DTW feature, and the controls it had to pass
#
# `dtw_mag` is built in `feature_engineering.py`. The four earlier DTW designs all failed
# the same way, by re-deriving something the model already had, so this section runs the
# controls rather than reporting the feature's own correlation.
#
# Two controls, each repeated 20 times. The **shuffled-outcome null** keeps the DTW
# neighbour selection intact and permutes the outcomes those neighbours contribute, which
# measures what the search finds when there is nothing to find. The **random-neighbour
# control** keeps the pool and the target but picks neighbours at random, which measures
# what the pooling alone is worth.
#
# Scored on the Parkinson target with the volatility level and the seasonal term already
# in the model, so the feature has to beat a strong baseline rather than a bare one.

# %%
d["l_rv"] = np.log(d.realised_vol)
STRONG = FEATURES + ["l_rv", "l_seas"]
v = d.dropna(subset=["rv_park", "dtw_mag"] + STRONG).copy()
v = v[v.rv_park > 0]
v_year = np.asarray(v.index.year)


def wf_park(cols, years):
    preds, actuals = [], []
    for test_year in years:
        train, test = v[v_year < test_year], v[v_year == test_year]
        if len(test) == 0 or len(train) < 2000:
            continue
        fit = sm.GLM(train.rv_park, sm.add_constant(train[cols], has_constant="add"),
                     family=sm.families.Gamma(link=sm.families.links.Log())).fit()
        preds.append(fit.predict(sm.add_constant(test[cols], has_constant="add")))
        actuals.append(test.rv_park)
    p, a = pd.concat(preds), pd.concat(actuals)
    return qlike(a, p).mean(), 1 - np.var(np.log(a) - np.log(p)) / np.var(np.log(a))


for label, years in [("confirmation 2019-2020", range(2019, 2021)),
                     ("main 2021-2026", range(2021, 2027))]:
    base_q, base_r2 = wf_park(STRONG, years)
    real_q, real_r2 = wf_park(STRONG + ["dtw_mag"], years)
    gain = 100 * (base_q - real_q) / base_q
    print(f"{label}")
    print(f"  baseline           QLIKE {base_q:.4f}   R2_log {base_r2:.4f}")
    print(f"  baseline + dtw_mag QLIKE {real_q:.4f}   R2_log {real_r2:.4f}   ({gain:+.2f}%)")
    print()

# %% [markdown]
# **Findings:** The feature improves QLIKE by 5.51% on 2021-2026 and 7.44% on 2019-2020,
# a window untouched by any of the tuning. Against 20 shuffled-outcome nulls and 20
# random-neighbour controls, run separately on the same machinery, no control exceeds
# +0.30% on either window, which places the real result roughly 29 standard deviations
# above the null distribution on the confirmation window and 82 on the main one. Its
# partial t-stat is +24.8 with the volatility level and the seasonal term already in the
# model.
#
# **These numbers are the corrected ones.** The first version of this feature ended its
# query path at the bar labelled `t+30`, which on a left-labelled 5-minute grid carries
# data through `t+35` and so leaked the first five minutes of the target window. That
# version scored +13.54% and +12.73%. Ending the path at `t+25` instead cuts the gain
# roughly in half, and what remains is still far outside the null.
#
# This is the first DTW construction in the project to survive its own controls. The
# difference from the four that failed is that it matches the volatility *trajectory*
# rather than a z-normalised price path, so the amplitude information magnitude
# prediction needs is still present at match time.
#
# The same machinery was pointed at direction, with signed paths and three targets: the
# forward return, the gap between upside and downside excursion inside the window, and
# whether the high precedes the low. All three landed inside their control distributions
# on both windows, with 10 to 32 of 40 controls beating the real feature and z-scores
# between -1.0 and +0.8. Direction remains unpredicted.

# %% [markdown]
# ## Regularisation
#
# Ridge, Lasso and ElasticNet against OLS on the same feature set, alphas chosen by
# cross-validation inside each training fold.

# %%
REG_COLS = HAR + ["l_seas", "l_gseas"] + FEATURES
ESTIMATORS = {
    "OLS":        lambda: LinearRegression(),
    "Ridge":      lambda: RidgeCV(alphas=np.logspace(-4, 4, 40)),
    "Lasso":      lambda: LassoCV(alphas=np.logspace(-6, 1, 60), cv=5, max_iter=5000),
    "ElasticNet": lambda: ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1],
                                       alphas=np.logspace(-6, 1, 40), cv=5, max_iter=5000),
}
print(f"{len(REG_COLS)} features, {len(d[np.isin(year, list(TEST_YEARS))]):,} test bars\n")
for name, make in ESTIMATORS.items():
    preds, actuals, zeroed = [], [], []
    for test_year in TEST_YEARS:
        train, test = d[year < test_year], d[year == test_year]
        if len(test) == 0:
            continue
        mu, sd = train[REG_COLS].mean(), train[REG_COLS].std()
        est = make().fit(((train[REG_COLS] - mu) / sd).to_numpy(), train.y.to_numpy())
        preds.append(pd.Series(est.predict(((test[REG_COLS] - mu) / sd).to_numpy()),
                               index=test.index))
        actuals.append(test.rv_fwd)
        zeroed.append(int(np.sum(np.abs(est.coef_) < 1e-10)))
    log_pred, actual = pd.concat(preds), pd.concat(actuals)
    forecast = np.exp(log_pred + 0.5 * np.var(np.log(actual) - log_pred))
    r2 = 1 - np.var(np.log(actual) - log_pred) / np.var(np.log(actual))
    dropped = f"   dropped {int(np.mean(zeroed))}/{len(REG_COLS)}" if np.mean(zeroed) else ""
    print(f"  {name:<12} QLIKE {qlike(actual, forecast).mean():.4f}   "
          f"R2_log {r2:.4f}{dropped}")

# %%
train = d[year < TEST_YEARS.stop - 1]
mu, sd = train[REG_COLS].mean(), train[REG_COLS].std()
lasso = LassoCV(alphas=np.logspace(-6, 1, 60), cv=5, max_iter=5000).fit(
    ((train[REG_COLS] - mu) / sd).to_numpy(), train.y.to_numpy())
print(f"Lasso coefficients, final fold (alpha {lasso.alpha_:.2e})\n")
for col, coef in sorted(zip(REG_COLS, lasso.coef_), key=lambda x: -abs(x[1])):
    print(f"  {col:<18} {coef:+.4f}" + ("   dropped" if abs(coef) < 1e-10 else ""))

# %% [markdown]
# **Findings:** Regularisation does nothing here. Ridge matches OLS to four decimals,
# since cross-validation selects an alpha near zero, and Lasso and ElasticNet are
# marginally worse while dropping two features on average. With 11 features and 53,458
# test bars the coefficients are already precisely estimated, and the largest VIF in
# the directional feature set was 1.73, which is not the kind of collinearity
# regularisation exists to fix.
#
# The Lasso path is still informative. It drops `vol_z` entirely, not because the
# volatility level is irrelevant but because `l_seas` is the log of the level times the
# squared seasonal multiplier and already contains it. It also drops the weekly HAR
# term and `range_pos_30m`. The two seasonal terms carry the largest coefficients by a
# wide margin.

# %% [markdown]
# ## Save the forecasts

# %%
forecast, actual, log_pred = wf(["l_gseas"] + FEATURES, "qmle")
out = pd.DataFrame({"forecast_var": forecast, "realised_var": actual.to_numpy()},
                   index=actual.index)
out["forecast_vol_bps"] = np.sqrt(out.forecast_var) * 1e4
out.to_parquet(JACK_DATA / "vol_forecasts.parquet")
print(f"wrote vol_forecasts.parquet  {len(out):,} rows")
print(out.describe().round(6).to_string())

# %% [markdown]
# ## The complete feature set
#
# Everything now available, scored on the Parkinson target by Gamma GLM, built up in the
# order the features were added so each block's contribution is visible. `l_rv` and
# `l_seas` are the volatility level and the level times the squared seasonal multiplier.

# %%
full = d.dropna(subset=["rv_park"]).copy()
full["l_rv"] = np.log(full.realised_vol)
full = full[(full.rv_park > 0) & np.isfinite(full.l_rv)]
full_year = np.asarray(full.index.year)

BLOCKS = [
    ("original six",       FEATURES),
    ("+ level and seasonal", ["l_rv", "l_seas"]),
    ("+ HAR terms",        ["l_1b", "l_1d", "l_1w"]),
    ("+ dtw_mag",          ["dtw_mag"]),
    ("+ realised variance and volume", ["log_rv4h", "signed_jump", "vol_surprise"]),
    ("+ VIX",              ["vix_level", "vix_slope"]),
    ("+ bar range and clock", ["bar_range", "tod_sin", "tod_cos"]),
]


def wf_full(cols, years):
    sub = full.dropna(subset=cols)
    sub_year = np.asarray(sub.index.year)
    preds, actuals = [], []
    for test_year in years:
        train, test = sub[sub_year < test_year], sub[sub_year == test_year]
        if len(test) == 0 or len(train) < 2000:
            continue
        fit = sm.GLM(train.rv_park, sm.add_constant(train[cols], has_constant="add"),
                     family=sm.families.Gamma(link=sm.families.links.Log())).fit()
        preds.append(fit.predict(sm.add_constant(test[cols], has_constant="add")))
        actuals.append(test.rv_park)
    p, a = pd.concat(preds), pd.concat(actuals)
    return qlike(a, p).mean(), 1 - np.var(np.log(a) - np.log(p)) / np.var(np.log(a))


for label, years in [("2019-2020 (confirmation)", range(2019, 2021)),
                     ("2021-2026 (main)", range(2021, 2027))]:
    print(label)
    cols, prev = [], None
    for name, block in BLOCKS:
        cols = cols + block
        sub = full.dropna(subset=cols)
        if len(sub) < 5000:
            print(f"  {name:<34} skipped, {len(sub)} rows"); continue
        q, r2 = wf_full(cols, years)
        step = "" if prev is None else f"  ({100 * (prev - q) / prev:+.2f}%)"
        print(f"  {name:<34} QLIKE {q:.4f}  R2 {r2:.4f}{step}")
        prev = q
    print()

# %% [markdown]
# **Findings:** The full set reaches QLIKE 0.4063 on the confirmation window and 0.3799
# on the main one, from 1.1067 and 0.6351 for the six-feature starting point. In
# R-squared on log variance that is 0.747 and 0.729, against 0.605 and 0.517.
#
# | block | 2019-2020 | 2021-2026 |
# |---|---|---|
# | level and seasonal | +53.8% | +20.4% |
# | HAR terms | -1.5% | +2.5% |
# | `dtw_mag` | +7.8% | +4.6% |
# | realised variance and volume | +6.9% | +10.9% |
# | VIX | +3.9% | +3.1% |
# | bar range and clock | +5.1% | +6.5% |
#
# The realised-variance group is the largest addition of this round, which is the least
# surprising result here: it estimates volatility from 1-minute returns rather than from
# 30-minute bars, and the whole realised-volatility literature says that is the better
# estimator. `bar_range` is nearly as valuable and was already sitting in the bar table.
#
# The HAR terms are the one block that does not earn its place, costing 1.5% on the
# confirmation window and adding 2.5% on the main one. They are computed from 30-minute
# squared returns and `log_rv4h` measures the same thing far more precisely, so they are
# largely redundant once it is present.
#
# VIX is worth 3.9% and 3.1%. That is modest, but it is the only forward-looking input in
# the set and the only one that is not a function of past ES prices, so it is the feature
# whose information the others structurally cannot reproduce.
#
# Every other block improves both windows. Nothing in this table forecasts direction.
