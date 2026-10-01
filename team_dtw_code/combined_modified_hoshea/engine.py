"""Reusable building blocks for the combined_modified_hoshea search.

Family merge, impact screen and DTW matching follow combined/features.py.
Waiting time, path length and neighbour weighting are parameters, not constants.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from dtaidistance import dtw
from scipy import stats

from config import (
    ASSETS,
    BLOOMBERG,
    CLEAN_BARS_FMT,
    FAMILY_OVERLAP,
    HOLD_MIN,
    IMPACT_MIN_T,
    IMPACT_WINDOW_MIN,
    LEAK_TEST_MINUTES,
    MAX_POOL,
    MIN_NEIGHBOURS,
    MIN_RELEASES,
    NY,
    PATH_FREQ_MIN,
    RECENCY_HALFLIFE_DAYS,
    WARP_MINUTES,
    anchors_for_wait,
    path_len_minutes,
)


class Px:
    """Exact-timestamp lookup. NaN means the minute is outside a kept session."""

    def __init__(self, s: pd.Series):
        self.ts = s.index.values.astype("datetime64[ns]").astype(np.int64)
        self.v = s.to_numpy(float)

    def at(self, stamps) -> np.ndarray:
        q = pd.DatetimeIndex(stamps).tz_convert("UTC").values.astype("datetime64[ns]").astype(np.int64)
        pos = np.searchsorted(self.ts, q)
        pc = np.clip(pos, 0, len(self.ts) - 1)
        out = np.full(len(q), np.nan)
        hit = (pos < len(self.ts)) & (self.ts[pc] == q)
        out[hit] = self.v[pc[hit]]
        return out


def load_prices() -> tuple[dict[str, Px], dict[str, Px]]:
    pxs, contracts = {}, {}
    for s in ASSETS:
        b = pd.read_parquet(CLEAN_BARS_FMT.format(symbol=s), columns=["close", "contract"]).sort_index()
        b.index = pd.to_datetime(b.index, utc=True)
        pxs[s], contracts[s] = Px(b.close), Px(b.contract.astype(float))
        print(f"{s}: {len(b):,} clean bars  {b.index.min().date()} -> {b.index.max().date()}")
    return pxs, contracts


def load_lines() -> pd.DataFrame:
    b = pd.concat([pd.read_excel(BLOOMBERG, sheet_name=s)
                   for s in pd.ExcelFile(BLOOMBERG).sheet_names], ignore_index=True)
    b.columns = [c.split(".")[-1] if "DROPNA" in str(c) else c for c in b.columns]
    b = b[b.COUNTRY_NAME.astype(str).str.contains("United States", na=False)]
    b["RELEASE_DATE"] = pd.to_datetime(b.RELEASE_DATE, errors="coerce")
    b = b.dropna(subset=["RELEASE_DATE", "RELEASE_TIME", "EVENT_NAME"])
    t = pd.to_datetime(b.RELEASE_TIME.astype(str), format="%H:%M:%S", errors="coerce")
    b, t = b[t.notna()].copy(), t[t.notna()]
    local = (b.RELEASE_DATE.dt.normalize() + pd.to_timedelta(t.dt.hour, "h")
             + pd.to_timedelta(t.dt.minute, "m"))
    b["ts"] = local.dt.tz_localize(NY, ambiguous=True, nonexistent="shift_forward").dt.tz_convert("UTC")
    b["clock_et"] = b.ts.dt.tz_convert(NY).dt.strftime("%H:%M")
    return b[["ts", "clock_et", "EVENT_NAME"]].rename(columns={"EVENT_NAME": "line"})


def assign_families(lines: pd.DataFrame) -> dict[str, str]:
    stamps = {ln: set(g.ts) for ln, g in lines.groupby("line") if len(g) >= 12}
    names = sorted(stamps, key=lambda n: -len(stamps[n]))
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(names):
        for c in names[i + 1:]:
            if find(a) == find(c):
                continue
            shared = len(stamps[a] & stamps[c])
            if shared and shared / min(len(stamps[a]), len(stamps[c])) >= FAMILY_OVERLAP:
                parent[find(c)] = find(a)
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    out = {}
    for members in groups.values():
        best = max(members, key=lambda m: len(stamps[m]))
        clock = lines.loc[lines.line == best, "clock_et"].mode().iloc[0]
        for m in members:
            out[m] = f"{best} @ {clock}"
    return out


def load_families() -> dict[str, list]:
    lines = load_lines()
    fam = assign_families(lines)
    lines["family"] = lines.line.map(fam)
    lines = lines.dropna(subset=["family"])
    fam_stamps = {f: sorted(set(g.ts)) for f, g in lines.groupby("family")}
    print(f"{lines.line.nunique()} release lines -> {len(fam_stamps)} families")
    return fam_stamps


def path_step_offsets(path_pre_min: int, wait_min: int, leak_min: int = LEAK_TEST_MINUTES) -> np.ndarray:
    n = path_len_minutes(path_pre_min, wait_min)
    return np.arange(-(n - 1), leak_min + 1)


def build_games(
    px: Px,
    contract: Px,
    fam_stamps: dict,
    wait_min: int,
    path_pre_min: int,
    hold_min: int = HOLD_MIN,
) -> pd.DataFrame:
    anchors = anchors_for_wait(wait_min)
    steps = path_step_offsets(path_pre_min, wait_min)
    recs = []
    for f, ts in fam_stamps.items():
        T = pd.DatetimeIndex(sorted(ts))
        has_print = np.isfinite(px.at(T)) & np.isfinite(px.at(T + pd.Timedelta(minutes=wait_min)))
        for a in anchors:
            entry = T + pd.Timedelta(minutes=a)
            grid = entry.values[:, None] + (steps * 60_000_000_000).astype("timedelta64[ns]")
            path = px.at(pd.DatetimeIndex(grid.ravel()).tz_localize("UTC")).reshape(len(T), len(steps))
            p0 = px.at(entry)
            p1 = px.at(entry + pd.Timedelta(minutes=hold_min))
            same_contract = (
                contract.at(entry + pd.Timedelta(minutes=int(steps[0])))
                == contract.at(entry + pd.Timedelta(minutes=hold_min))
            )
            ok = (np.isfinite(path).all(1) & np.isfinite(p0) & np.isfinite(p1) & has_print
                  & same_contract)
            sd = path.std(1, keepdims=True)
            sd[sd < 1e-12] = 1.0
            z = (path - path.mean(1, keepdims=True)) / sd
            for i in np.where(ok)[0]:
                recs.append(dict(
                    family=f, event_ts=T[i], anchor=a,
                    clock_et=T[i].tz_convert(NY).strftime("%H:%M"),
                    entry_ts=entry[i],
                    exit_ts=entry[i] + pd.Timedelta(minutes=hold_min),
                    year=int(T[i].year),
                    fwd_ret=float(np.log(p1[i] / p0[i])),
                    path=z[i],
                ))
    return pd.DataFrame(recs).sort_values("entry_ts").reset_index(drop=True)


def impact_by_year(px: Px, fam_stamps: dict, years: list[int]) -> pd.DataFrame:
    by_clock: dict[str, set] = {}
    for f, ts in fam_stamps.items():
        for t in ts:
            by_clock.setdefault(t.tz_convert(NY).strftime("%H:%M"), set()).add(t)

    def moves(stamps, before: int) -> np.ndarray:
        s = pd.DatetimeIndex(sorted(t for t in stamps if t.year < before))
        if len(s) == 0:
            return np.array([])
        a, b = px.at(s), px.at(s + pd.Timedelta(minutes=IMPACT_WINDOW_MIN))
        v = np.abs(np.log(b / a)) * 1e4
        return v[np.isfinite(v) & (v > 0)]

    rows = []
    for year in years:
        for f, ts in fam_stamps.items():
            own = moves(set(ts), year)
            if len(own) < MIN_RELEASES:
                continue
            clock = pd.DatetimeIndex(sorted(ts))[0].tz_convert(NY).strftime("%H:%M")
            ctrl = moves(by_clock[clock] - set(ts), year)
            if len(ctrl) < 20:
                continue
            t = stats.ttest_ind(np.log(own), np.log(ctrl), equal_var=False).statistic
            rows.append(dict(year=year, family=f, impact=own.mean() / ctrl.mean(),
                             impact_t=float(t)))
    return pd.DataFrame(rows)


def admit_games(games: pd.DataFrame, impact: pd.DataFrame) -> pd.DataFrame:
    out = games.merge(impact, on=["year", "family"], how="left")
    out["admitted"] = (out.impact_t >= IMPACT_MIN_T).fillna(False)
    return out


def neighbor_weights(dsel: np.ndarray, age_sel: np.ndarray, scheme: str) -> np.ndarray:
    """Normalised weights over the k selected neighbours."""
    eps = 1e-6
    if scheme == "inv_dist_recency":
        w = (1.0 / (dsel + eps)) * np.exp(-age_sel / RECENCY_HALFLIFE_DAYS)
    elif scheme == "inv_dist":
        w = 1.0 / (dsel + eps)
    elif scheme == "inv_dist_sq_recency":
        w = (1.0 / (dsel * dsel + eps)) * np.exp(-age_sel / RECENCY_HALFLIFE_DAYS)
    elif scheme == "recency":
        w = np.exp(-age_sel / RECENCY_HALFLIFE_DAYS)
    elif scheme == "uniform":
        w = np.ones_like(dsel, dtype=float)
    elif scheme == "softmax_neg_dist":
        scale = max(float(np.median(dsel)), eps)
        z = -(dsel - dsel.min()) / scale
        z = z - z.max()
        w = np.exp(z)
    elif scheme == "rank_recency":
        ranks = np.argsort(np.argsort(dsel)).astype(float) + 1.0
        w = (1.0 / ranks) * np.exp(-age_sel / RECENCY_HALFLIFE_DAYS)
    else:
        raise ValueError(f"unknown weighting {scheme!r}")
    s = float(w.sum())
    return w / s if s > 0 else np.full_like(w, 1.0 / max(len(w), 1))


def dtw_block(
    games: pd.DataFrame,
    query: np.ndarray,
    ks: Iterable[int],
    lookbacks: Iterable[int],
    weightings: Iterable[str],
) -> dict[tuple, pd.DataFrame]:
    """dtw_dir / dtw_disp for every (k, lookback, weighting). Distances computed once."""
    ks, lookbacks, weightings = list(ks), list(lookbacks), list(weightings)
    win = max(1, round(WARP_MINUTES / PATH_FREQ_MIN))
    settings = [(k, lb, w) for k in ks for lb in lookbacks for w in weightings]
    out = {st: {c: np.full(len(games), np.nan) for c in ("dtw_dir", "dtw_disp")} for st in settings}
    for _, grp in games.groupby(["clock_et", "anchor"], sort=False):
        idx = grp.index.to_numpy()
        qpos = np.where(query[idx])[0]
        if len(qpos) == 0:
            continue
        t0 = grp.entry_ts.values.astype("datetime64[ns]").astype(np.int64)
        rets = grp.fwd_ret.to_numpy(float)
        paths = [np.asarray(p, dtype=np.double) for p in grp.path]
        hi = int(qpos.max()) + 1
        D = dtw.distance_matrix_fast(
            paths[:hi], window=win, use_pruning=True,
            block=((0, hi), (int(qpos.min()), hi)), compact=False, parallel=True,
        )
        for pos in qpos:
            age = (t0[pos] - t0[:pos]) / 86_400_000_000_000.0
            for lb in lookbacks:
                el = np.where((age > 0) & (age <= 365 * lb))[0]
                if len(el) < MIN_NEIGHBOURS:
                    continue
                el = el[-MAX_POOL:]
                d = D[pos, el]
                m = np.isfinite(d)
                el, d = el[m], d[m]
                if len(el) < MIN_NEIGHBOURS:
                    continue
                order = np.argsort(d)
                for k in ks:
                    sel, dsel = el[order[:k]], d[order[:k]]
                    r = rets[sel]
                    age_sel = age[sel]
                    for scheme in weightings:
                        w = neighbor_weights(dsel, age_sel, scheme)
                        f = float(np.dot(w, r))
                        out[k, lb, scheme]["dtw_dir"][idx[pos]] = f
                        out[k, lb, scheme]["dtw_disp"][idx[pos]] = float(np.sqrt(np.dot(w, (r - f) ** 2)))
    return {st: pd.DataFrame(v, index=games.index) for st, v in out.items()}


def rank_ic(x: pd.Series, y: pd.Series) -> float:
    m = x.notna() & y.notna()
    return float(stats.spearmanr(x[m], y[m]).correlation) if m.sum() > 10 else np.nan


def hit_rate(signal: pd.Series, y: pd.Series) -> float:
    m = signal.notna() & y.notna() & (signal != 0) & (y != 0)
    if m.sum() < 10:
        return np.nan
    return float((np.sign(signal[m]) == np.sign(y[m])).mean())


def quintile_spread(signal: pd.Series, y: pd.Series) -> float:
    m = signal.notna() & y.notna()
    if m.sum() < 50:
        return np.nan
    q = pd.qcut(signal[m].to_numpy(), 5, labels=False, duplicates="drop")
    if len(set(q)) < 5:
        return np.nan
    yy = y[m].to_numpy()
    return float(yy[q == 4].mean() - yy[q == 0].mean())


def score_span(games: pd.DataFrame, feats: pd.DataFrame, span: tuple[int, int]) -> dict:
    m = games.admitted & games.year.between(*span) & feats.dtw_dir.notna()
    return {
        "games": int(m.sum()),
        "dir_ic": rank_ic(feats.dtw_dir[m], games.fwd_ret[m]),
        "disp_ic": rank_ic(feats.dtw_disp[m], games.fwd_ret[m].abs()),
        "hit": hit_rate(feats.dtw_dir[m], games.fwd_ret[m]),
        "q51": quintile_spread(feats.dtw_dir[m], games.fwd_ret[m]),
    }
