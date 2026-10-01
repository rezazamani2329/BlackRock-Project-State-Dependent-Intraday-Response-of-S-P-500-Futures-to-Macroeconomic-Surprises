from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
FIG_DIR = BASE_DIR / "figures"
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_overall_metrics() -> None:
    experiments = ["Mixed major-event pool", "Single-event expanded pool"]
    metrics = {
        "Pearson IC": [0.0699, 0.0111],
        "Spearman IC": [0.0355, 0.0006],
        "Hit Rate": [0.506888, 0.491929],
        "Q5-Q1 Spread": [0.000477, 0.000092],
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()

    for ax, (name, values) in zip(axes, metrics.items()):
        bars = ax.bar(experiments, values, color=["#4C78A8", "#F58518"])
        ax.set_title(name)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=12)
        for b, v in zip(bars, values):
            if "Hit Rate" in name:
                label = f"{v:.2%}"
            elif "Spread" in name:
                label = f"{v:.6f}"
            else:
                label = f"{v:.4f}"
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), label, ha="center", va="bottom", fontsize=9)

    fig.suptitle("DTW experiments: overall metric comparison", fontsize=13)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "dtw_overall_metrics_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_event_level_ic() -> None:
    mixed_major_ic = {
        "CPI": 0.023399,
        "FOMC": 0.148634,
        "NFP": -0.041889,
        "PCE": 0.088909,
    }
    single_expanded_ic = {
        "CPI": -0.012966,
        "Employment Cost Index": -0.076023,
        "FOMC": 0.122885,
        "GDP": -0.036503,
        "Import/Export Prices": -0.026238,
        "Industrial Production": -0.007220,
        "Initial Jobless Claims": 0.008036,
        "JOLTS": 0.036794,
        "NFP": -0.037204,
        "PCE": 0.024010,
        "PPI": -0.004454,
        "Retail Sales": -0.015733,
        "Wholesale Trade": -0.013881,
    }

    common_events = ["CPI", "NFP", "PCE", "FOMC"]
    x = np.arange(len(common_events))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), gridspec_kw={"width_ratios": [1.0, 1.25]})

    # Left: major events comparison across two experiments
    mixed_vals = [mixed_major_ic[e] for e in common_events]
    single_vals = [single_expanded_ic[e] for e in common_events]
    axes[0].bar(x - width / 2, mixed_vals, width=width, label="Mixed major-event", color="#4C78A8")
    axes[0].bar(x + width / 2, single_vals, width=width, label="Single-event expanded", color="#F58518")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(common_events)
    axes[0].set_ylabel("Pearson IC")
    axes[0].set_title("Major-event IC: mixed vs single-event")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    # Right: full expanded single-event IC
    single_df = pd.DataFrame({"event_type": list(single_expanded_ic.keys()), "pearson_ic": list(single_expanded_ic.values())})
    single_df = single_df.sort_values("pearson_ic")
    axes[1].barh(single_df["event_type"], single_df["pearson_ic"], color="#72B7B2")
    axes[1].axvline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("Pearson IC")
    axes[1].set_title("Single-event expanded pool: IC by event type")
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle("Event-level IC decomposition", fontsize=13)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "dtw_event_level_ic.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_t0_ic_decomposition() -> None:
    t0_rows = [
        ("08:00", -30, 458, 0.024860),
        ("08:30", 0, 458, 0.020381),
        ("09:00", 30, 458, 0.046657),
        ("09:30", 60, 458, -0.023506),
        ("13:30", -30, 73, 0.117478),
        ("14:00", 0, 73, 0.060479),
        ("14:30", 30, 73, 0.100612),
        ("15:00", 60, 54, -0.275096),
    ]
    t0_df = pd.DataFrame(t0_rows, columns=["clock_et", "offset_min", "n", "pearson_ic"])

    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = ["#E45756" if v < 0 else "#4C78A8" for v in t0_df["pearson_ic"]]
    bars = ax.bar(t0_df["clock_et"], t0_df["pearson_ic"], color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Pearson IC")
    ax.set_xlabel("T0 (ET)")
    ax.set_title("Mixed major-event DTW: Pearson IC by T0-specific factor")
    ax.grid(axis="y", alpha=0.3)

    for bar, n in zip(bars, t0_df["n"]):
        y = bar.get_height()
        dy = 0.006 if y >= 0 else -0.018
        va = "bottom" if y >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y + dy, f"n={n}", ha="center", va=va, fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "dtw_t0_ic_decomposition.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _load_generalized_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    grid_file = OUTPUT_DIR / "dtw_simple_generalized_grid_results.parquet"
    best_file = OUTPUT_DIR / "dtw_simple_generalized_best_factor.parquet"
    params_file = OUTPUT_DIR / "dtw_simple_generalized_best_params.json"

    if not grid_file.exists() or not best_file.exists() or not params_file.exists():
        raise FileNotFoundError(
            "Generalized artifacts not found. Expected files under hoshea_zeng/outputs."
        )

    grid = pd.read_parquet(grid_file)
    best = pd.read_parquet(best_file)
    best["t0_utc"] = pd.to_datetime(best["t0_utc"], utc=True)
    best["year"] = best["t0_utc"].dt.year
    params = pd.read_json(params_file, typ="series").to_dict()
    return grid, best, params


def _build_total_games_by_year(
    event_types: list[str],
    path_freq_min: int,
    offsets_min: list[int],
) -> pd.DataFrame:
    price_file = REPO_ROOT / "data" / "processed" / "es_1min_bars_2010_2026.parquet"
    event_file = REPO_ROOT / "data" / "processed" / "macro_event_calendar_expanded_2010_2026.parquet"

    price = pd.read_parquet(price_file, columns=["close"]).sort_index()
    if price.index.tz is None:
        price.index = price.index.tz_localize("UTC")
    else:
        price.index = price.index.tz_convert("UTC")
    close = price["close"].astype(float)

    events = pd.read_parquet(event_file)
    events["timestamp_utc"] = pd.to_datetime(events["timestamp_utc"], utc=True)
    events = events[events["event_type"].isin(event_types)].sort_values("timestamp_utc")

    rows: list[pd.Timestamp] = []
    for row in events.itertuples(index=False):
        event_ts = row.timestamp_utc
        for offset in offsets_min:
            t0 = event_ts + pd.Timedelta(minutes=int(offset))
            path_idx = pd.date_range(
                t0,
                t0 + pd.Timedelta(minutes=60),
                freq=f"{int(path_freq_min)}min",
                tz="UTC",
            )
            if close.reindex(path_idx).isna().any():
                continue

            t_entry = t0 + pd.Timedelta(minutes=60)
            t_exit = t_entry + pd.Timedelta(minutes=30)
            if close.reindex([t_entry, t_exit]).isna().any():
                continue
            rows.append(t0)

    total = pd.DataFrame({"t0_utc": rows})
    total["year"] = pd.to_datetime(total["t0_utc"], utc=True).dt.year
    return total.groupby("year").size().rename("total_games").reset_index()


def plot_generalized_hyperparameter_diagnostics() -> None:
    grid, _, _ = _load_generalized_artifacts()

    group_summary = (
        grid.groupby("event_group")["pearson_ic"]
        .agg(mean_ic="mean", max_ic="max", min_ic="min")
        .reset_index()
    )
    freq_summary = (
        grid.groupby("path_freq_min")["pearson_ic"]
        .agg(mean_ic="mean", max_ic="max", min_ic="min")
        .reset_index()
    )
    lookback_summary = (
        grid.groupby("lookback_years")["pearson_ic"]
        .agg(mean_ic="mean", max_ic="max", min_ic="min")
        .reset_index()
    )

    group_summary.to_csv(OUTPUT_DIR / "dtw_generalized_group_summary.csv", index=False)
    freq_summary.to_csv(OUTPUT_DIR / "dtw_generalized_freq_summary.csv", index=False)
    lookback_summary.to_csv(OUTPUT_DIR / "dtw_generalized_lookback_summary.csv", index=False)

    # Figure 1: ranked IC across all grid combinations
    ranked = grid.sort_values("pearson_ic", ascending=False).reset_index(drop=True).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    colors = ranked["event_group"].map(
        {
            "core_major": "#4C78A8",
            "fomc_jolts_pce_claims": "#F58518",
        }
    ).fillna("#72B7B2")

    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    ax.bar(ranked["rank"], ranked["pearson_ic"], color=colors)
    ax.set_title("Generalized grid search: Pearson IC by hyperparameter rank")
    ax.set_xlabel("Rank (best to worst)")
    ax.set_ylabel("Pearson IC")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.0, color="black", linewidth=1)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "dtw_generalized_grid_ranked_ic.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: mean IC by frequency and lookback, split by event group
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, constrained_layout=True)
    for ax, group_name in zip(axes, sorted(grid["event_group"].unique())):
        pivot = (
            grid[grid["event_group"] == group_name]
            .pivot_table(
                index="lookback_years",
                columns="path_freq_min",
                values="pearson_ic",
                aggfunc="mean",
            )
            .sort_index()
        )
        im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")
        ax.set_title(f"{group_name}: mean Pearson IC")
        ax.set_xlabel("path_freq_min")
        ax.set_ylabel("lookback_years")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=axes, shrink=0.9)
    cbar.set_label("Pearson IC")
    fig.savefig(FIG_DIR / "dtw_generalized_grid_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_generalized_coverage_and_ic() -> None:
    _, best, params = _load_generalized_artifacts()

    event_types = [s.strip() for s in str(params["event_types"]).split(",") if s.strip()]
    path_freq_min = int(params["path_freq_min"])
    offsets = sorted(best["offset_min"].dropna().astype(int).unique().tolist())
    total_by_year = _build_total_games_by_year(
        event_types=event_types,
        path_freq_min=path_freq_min,
        offsets_min=offsets,
    )

    scored_by_year = best.groupby("year").size().rename("scored_games").reset_index()
    yearly = total_by_year.merge(scored_by_year, on="year", how="left")
    yearly["scored_games"] = yearly["scored_games"].fillna(0).astype(int)
    yearly["coverage_rate"] = yearly["scored_games"] / yearly["total_games"]

    yearly_ic = (
        best.groupby("year")
        .apply(
            lambda g: pd.Series(
                {
                    "n": int(len(g)),
                    "pearson_ic": g["dtw_factor"].corr(g["future_logret"], method="pearson"),
                    "spearman_ic": g["dtw_factor"].corr(g["future_logret"], method="spearman"),
                    "hit_rate": (
                        np.sign(g["dtw_factor"]).to_numpy()
                        == np.sign(g["future_logret"]).to_numpy()
                    ).mean(),
                }
            )
        )
        .reset_index()
    )

    yearly.to_csv(OUTPUT_DIR / "dtw_generalized_coverage_by_year.csv", index=False)
    yearly_ic.to_csv(OUTPUT_DIR / "dtw_generalized_ic_by_year.csv", index=False)

    # Coverage figure
    fig, ax1 = plt.subplots(figsize=(11.5, 4.8))
    ax1.bar(yearly["year"], yearly["total_games"], color="#D9D9D9", label="Total eligible games")
    ax1.bar(yearly["year"], yearly["scored_games"], color="#4C78A8", label="Scored games")
    ax1.set_ylabel("Game count")
    ax1.set_xlabel("Year")
    ax1.set_title("Best generalized factor: yearly coverage")
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(yearly["year"], yearly["coverage_rate"], color="#E45756", marker="o", label="Coverage rate")
    ax2.set_ylabel("Coverage rate")
    ax2.set_ylim(0, 1.02)

    lines, labels = [], []
    for ax in (ax1, ax2):
        l, lab = ax.get_legend_handles_labels()
        lines.extend(l)
        labels.extend(lab)
    ax1.legend(lines, labels, loc="lower right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "dtw_generalized_coverage_by_year.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Yearly IC figure
    fig, ax1 = plt.subplots(figsize=(11.5, 4.8))
    ax1.plot(yearly_ic["year"], yearly_ic["pearson_ic"], color="#4C78A8", marker="o", label="Pearson IC")
    ax1.plot(yearly_ic["year"], yearly_ic["spearman_ic"], color="#72B7B2", marker="o", label="Spearman IC")
    ax1.axhline(0.0, color="black", linewidth=1)
    ax1.set_ylabel("IC")
    ax1.set_xlabel("Year")
    ax1.set_title("Best generalized factor: yearly IC profile")
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(yearly_ic["year"], yearly_ic["n"], alpha=0.15, color="#F58518", label="Scored sample size")
    ax2.set_ylabel("Sample size")

    lines, labels = [], []
    for ax in (ax1, ax2):
        l, lab = ax.get_legend_handles_labels()
        lines.extend(l)
        labels.extend(lab)
    ax1.legend(lines, labels, loc="upper left")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "dtw_generalized_yearly_ic.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_overall_metrics()
    plot_event_level_ic()
    plot_t0_ic_decomposition()
    plot_generalized_hyperparameter_diagnostics()
    plot_generalized_coverage_and_ic()
    print(f"Wrote figures to {FIG_DIR}")
