#!/usr/bin/env python3
"""
Grid-search trend_down_stocks parameters and tabulate since-2010 stats.

Also: S&P 500 (point-in-time) rebound-by-horizon analysis (1–52 weeks).

Heavy step (load_daily_prices) runs once and is cached to disk; subsequent
runs reuse the cache.

Historical S&P 500 membership comes from data/sp500/ (fja05680/sp500),
so bargain filters use membership on the event date rather than a current
static list (avoids survivor / look-ahead bias from a present-day roster).

Usage:
  python experiment_trend_down_params.py
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from trend_down_stocks import (
    attach_future_prices,
    filter_short_term_rebounds,
    find_trending_down_stocks,
    load_daily_prices,
    suppress_similar_tuples,
)
from utilities.stock_stooq import STOOQ_SAVE_DIR

# Cached daily closes (built once; reused across experiments).
DAILY_PRICES_CACHE = STOOQ_SAVE_DIR / "daily_prices.pkl"
RESULTS_FILE = STOOQ_SAVE_DIR / "trend_down_param_sweep.csv"

# Historical S&P 500 membership (https://github.com/fja05680/sp500).
SP500_DIR = Path(__file__).resolve().parent / "data" / "sp500"
SP500_TICKER_START_END_FILE = SP500_DIR / "sp500_ticker_start_end.csv"
SP500_COMPARE_FILE = STOOQ_SAVE_DIR / "trend_down_sp500_compare.csv"
SP500_HORIZON_FILE = STOOQ_SAVE_DIR / "sp500_rebound_by_horizon.csv"
SP500_HORIZON_PLOT = STOOQ_SAVE_DIR / "sp500_rebound_by_horizon.png"

# Parameter grid (edit here to experiment).
TREND_DOWN_THRESH_VALUES = np.arange(0.01, 0.20, 0.01)
REBOUND_SHORT_TERM_PER_THRESH_VALUES = [0.03]
WAIT_FOR_SHORT_TERM_REBOUND_VALUES = [False]

STATS_SINCE = "2010-01-01"
HORIZON_WEEKS = list(range(1, 53))  # 1 .. 52 weeks
DAYS_PER_WEEK = 7

# Open-ended membership intervals use this as an exclusive upper bound.
_SP500_OPEN_END = pd.Timestamp("2262-01-01")


def load_or_build_daily_prices(cache_path: Path = DAILY_PRICES_CACHE) -> pd.DataFrame:
    """Load daily prices from cache, or build once via load_daily_prices()."""
    STOOQ_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f"Loading cached daily prices from {cache_path} ...")
        daily_df = pd.read_pickle(cache_path)
        # Older caches may lack open_price; rebuild so next-day open buys work.
        if "open_price" not in daily_df.columns:
            print("  Cache missing open_price; rebuilding ...")
            daily_df = load_daily_prices()
            print(f"Saving daily prices cache to {cache_path} ...")
            daily_df.to_pickle(cache_path)
        else:
            print(
                f"  {len(daily_df):,} rows, {daily_df['ticker'].nunique():,} tickers "
                f"({daily_df['date'].min().date()} → {daily_df['date'].max().date()})"
            )
        return daily_df

    daily_df = load_daily_prices()
    print(f"Saving daily prices cache to {cache_path} ...")
    daily_df.to_pickle(cache_path)
    return daily_df


def stats_since(
    result_df: pd.DataFrame,
    since: str = STATS_SINCE,
) -> dict:
    """
    Same summary as trend_down_stocks.main() print block:
    median / mean price_change and pct_rebound(>1), for events since `since`.
    """
    empty = {
        "n_events": 0,
        "n_with_future_since": 0,
        "median_price_change": float("nan"),
        "mean_price_change": float("nan"),
        "pct_rebound_gt_1": float("nan"),
    }
    if result_df.empty:
        return empty

    with_future = result_df.dropna(subset=["price_change"]).copy()
    with_future["date"] = pd.to_datetime(with_future["date"])
    with_future = with_future[with_future["date"] >= since]
    if with_future.empty:
        return {**empty, "n_events": len(result_df)}

    return {
        "n_events": len(result_df),
        "n_with_future_since": len(with_future),
        "median_price_change": float(with_future["price_change"].median()),
        "mean_price_change": float(with_future["price_change"].mean()),
        "pct_rebound_gt_1": float((with_future["price_change"] > 1).mean() * 100.0),
    }


def run_pipeline(
    daily_df: pd.DataFrame,
    trend_down_thresh: float,
    wait_for_short_term_rebound: bool,
    rebound_short_term_per_thresh: float,
) -> pd.DataFrame:
    """Run the full sieve; return the events DataFrame with future prices."""
    trending_down_stocks = find_trending_down_stocks(
        daily_df,
        trend_down_thresh=trend_down_thresh,
        trend_down_history=config.trend_down_history,
    )
    if wait_for_short_term_rebound:
        trending_down_stocks = filter_short_term_rebounds(
            trending_down_stocks,
            daily_df,
            rebound_short_term_days=config.rebound_short_term_days,
            rebound_short_term_per_thresh=rebound_short_term_per_thresh,
        )
    trending_down_stocks = suppress_similar_tuples(
        trending_down_stocks,
        trending_down_suppression=config.trending_down_suppression,
    )
    return attach_future_prices(
        trending_down_stocks,
        daily_df,
        future_change=config.future_change,
    )


def run_one(
    daily_df: pd.DataFrame,
    trend_down_thresh: float,
    wait_for_short_term_rebound: bool,
    rebound_short_term_per_thresh: float,
) -> dict:
    """Run the full sieve for one parameter combination and return stats."""
    result_df = run_pipeline(
        daily_df,
        trend_down_thresh=trend_down_thresh,
        wait_for_short_term_rebound=wait_for_short_term_rebound,
        rebound_short_term_per_thresh=rebound_short_term_per_thresh,
    )
    summary = stats_since(result_df)
    return {
        "trend_down_thresh": trend_down_thresh,
        "WAIT_FOR_SHORT_TERM_REBOUND": wait_for_short_term_rebound,
        "rebound_short_term_per_thresh": (
            rebound_short_term_per_thresh if wait_for_short_term_rebound else None
        ),
        **summary,
    }


def _normalize_ticker(ticker: str) -> str:
    return str(ticker).upper().replace(".US", "").strip()


def load_sp500_membership_intervals(
    path: Path = SP500_TICKER_START_END_FILE,
) -> pd.DataFrame:
    """
    Load point-in-time S&P 500 membership intervals.

    Source: data/sp500/sp500_ticker_start_end.csv from fja05680/sp500.
    Membership on date d is start_date <= d < end_date (end exclusive;
    missing end_date means still a member / open-ended).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"S&P 500 membership file not found: {path}. "
            "Clone https://github.com/fja05680/sp500 into data/sp500/."
        )
    df = pd.read_csv(path)
    df["ticker"] = df["ticker"].map(_normalize_ticker)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["end_exclusive"] = df["end_date"].fillna(_SP500_OPEN_END)

    # Stooq / SEC hyphen vs dot variants (e.g. BRK-B vs BRK.B).
    aliases: list[pd.DataFrame] = [df]
    for src, dst in (("-", "."), (".", "-")):
        alt = df.copy()
        alt["ticker"] = alt["ticker"].str.replace(src, dst, regex=False)
        aliases.append(alt)
    out = pd.concat(aliases, ignore_index=True)
    return out.drop_duplicates(subset=["ticker", "start_date", "end_exclusive"])


def ever_sp500_tickers(
    membership: pd.DataFrame | None = None,
    price_tickers: set[str] | None = None,
) -> set[str]:
    """Tickers that appear in the S&P 500 at any time in the history file."""
    if membership is None:
        membership = load_sp500_membership_intervals()
    tickers = set(membership["ticker"].unique())
    if price_tickers is not None:
        tickers = {t for t in tickers if t in price_tickers}
    return tickers


def filter_events_in_sp500(
    events: pd.DataFrame,
    membership: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Keep event rows whose ticker was in the S&P 500 on the event date.

    Requires columns: ticker, date. Preserves all other columns.
    """
    if events.empty:
        return events.copy()
    if membership is None:
        membership = load_sp500_membership_intervals()

    ev = events.copy()
    ev["_row_id"] = np.arange(len(ev))
    ev["ticker"] = ev["ticker"].map(_normalize_ticker)
    ev["date"] = pd.to_datetime(ev["date"])

    merged = ev.merge(membership, on="ticker", how="inner")
    in_index = (merged["date"] >= merged["start_date"]) & (
        merged["date"] < merged["end_exclusive"]
    )
    keep_ids = merged.loc[in_index, "_row_id"].unique()
    return (
        ev.loc[ev["_row_id"].isin(keep_ids)]
        .drop(columns=["_row_id"])
        .reset_index(drop=True)
    )


def filter_event_tuples_in_sp500(
    events: list[tuple],
    membership: pd.DataFrame | None = None,
) -> list[tuple]:
    """Point-in-time S&P 500 filter for (ticker, date, price, historical_average) tuples."""
    if not events:
        return []
    df = pd.DataFrame(
        events, columns=["ticker", "date", "price", "historical_average"]
    )
    kept = filter_events_in_sp500(df, membership=membership)
    return [
        (
            row.ticker,
            row.date,
            float(row.price),
            float(row.historical_average),
        )
        for row in kept.itertuples(index=False)
    ]


def sp500_membership_mask(
    events: pd.DataFrame,
    membership: pd.DataFrame | None = None,
) -> pd.Series:
    """Boolean mask (aligned to events.index): True if in S&P 500 on event date."""
    if events.empty:
        return pd.Series(dtype=bool, index=events.index)
    if membership is None:
        membership = load_sp500_membership_intervals()

    work = pd.DataFrame(
        {
            "ticker": events["ticker"].map(_normalize_ticker),
            "date": pd.to_datetime(events["date"]),
            "_row_id": np.arange(len(events)),
        }
    )
    merged = work.merge(membership, on="ticker", how="inner")
    in_index = (merged["date"] >= merged["start_date"]) & (
        merged["date"] < merged["end_exclusive"]
    )
    keep_ids = set(merged.loc[in_index, "_row_id"].tolist())
    return pd.Series(
        [i in keep_ids for i in range(len(events))],
        index=events.index,
        dtype=bool,
    )


def compare_sp500_vs_rest(
    trend_down_thresh: float = 0.10,
    wait_for_short_term_rebound: bool = False,
) -> pd.DataFrame:
    """
    Run one config and tabulate since-2010 stats for point-in-time S&P 500
    members vs other tickers.
    """
    print("=" * 72)
    print("S&P 500 (point-in-time) vs rest — trend-down stats")
    print("=" * 72)
    print(f"  trend_down_thresh           = {trend_down_thresh}")
    print(f"  WAIT_FOR_SHORT_TERM_REBOUND = {wait_for_short_term_rebound}")
    print(f"  stats since                 = {STATS_SINCE}")
    print(f"  membership                  = {SP500_TICKER_START_END_FILE.name}")
    print("=" * 72)

    daily_df = load_or_build_daily_prices()
    membership = load_sp500_membership_intervals()
    price_tickers = set(daily_df["ticker"].unique())
    ever = ever_sp500_tickers(membership, price_tickers=price_tickers)
    print(f"Ever-S&P 500 tickers with price data: {len(ever):,}")

    result_df = run_pipeline(
        daily_df,
        trend_down_thresh=trend_down_thresh,
        wait_for_short_term_rebound=wait_for_short_term_rebound,
        rebound_short_term_per_thresh=config.rebound_short_term_per_thresh,
    )
    result_df = result_df.copy()
    result_df["ticker"] = result_df["ticker"].map(_normalize_ticker)
    result_df["is_sp500"] = sp500_membership_mask(result_df, membership=membership)

    rows = []
    for label, mask in [
        ("all", slice(None)),
        ("sp500", result_df["is_sp500"]),
        ("non_sp500", ~result_df["is_sp500"]),
    ]:
        subset = result_df if label == "all" else result_df.loc[mask]
        summary = stats_since(subset)
        n_tickers = subset["ticker"].nunique() if not subset.empty else 0
        rows.append({"group": label, "n_tickers": n_tickers, **summary})
        print(
            f"\n{label}: tickers={n_tickers:,}, events={summary['n_events']:,}, "
            f"with_future_since_{STATS_SINCE}={summary['n_with_future_since']:,}"
        )
        if summary["n_with_future_since"]:
            print(
                f"  price-change stats since {STATS_SINCE}: "
                f"median={summary['median_price_change']:.3f}, "
                f"mean={summary['mean_price_change']:.3f}, "
                f"pct_rebound(>1)={summary['pct_rebound_gt_1']:.1f}%"
            )
        else:
            print(f"  No events with future prices since {STATS_SINCE}.")

    table = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print("Summary table")
    print("=" * 72)
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    table.to_csv(SP500_COMPARE_FILE, index=False)
    print(f"\nSaved table to {SP500_COMPARE_FILE}")
    return table


def collect_bargain_events(
    daily_df: pd.DataFrame,
    trend_down_thresh: float = config.trend_down_thresh,
    wait_for_short_term_rebound: bool = config.WAIT_FOR_SHORT_TERM_REBOUND,
    rebound_short_term_per_thresh: float = config.rebound_short_term_per_thresh,
) -> list[tuple]:
    """Run the trend-down sieve through suppress; return event tuples."""
    events = find_trending_down_stocks(
        daily_df,
        trend_down_thresh=trend_down_thresh,
        trend_down_history=config.trend_down_history,
    )
    if wait_for_short_term_rebound:
        events = filter_short_term_rebounds(
            events,
            daily_df,
            rebound_short_term_days=config.rebound_short_term_days,
            rebound_short_term_per_thresh=rebound_short_term_per_thresh,
        )
    events = suppress_similar_tuples(
        events,
        trending_down_suppression=config.trending_down_suppression,
    )
    return events


def collect_sp500_events(
    daily_df: pd.DataFrame,
    trend_down_thresh: float = config.trend_down_thresh,
    wait_for_short_term_rebound: bool = config.WAIT_FOR_SHORT_TERM_REBOUND,
    rebound_short_term_per_thresh: float = config.rebound_short_term_per_thresh,
) -> tuple[list[tuple], pd.DataFrame, set[str]]:
    """
    Run the trend-down sieve on ever-S&P 500 tickers, then keep only events
    where the ticker was in the index on that date.

    Returns (event tuples, ever-S&P daily prices, ever-S&P ticker set).
    """
    membership = load_sp500_membership_intervals()
    price_tickers = set(daily_df["ticker"].unique())
    ever = ever_sp500_tickers(membership, price_tickers=price_tickers)
    universe_df = daily_df[daily_df["ticker"].isin(ever)].copy()
    print(
        f"Ever-S&P 500 with price data: {len(ever):,} tickers, "
        f"{len(universe_df):,} daily rows"
    )
    events = collect_bargain_events(
        universe_df,
        trend_down_thresh=trend_down_thresh,
        wait_for_short_term_rebound=wait_for_short_term_rebound,
        rebound_short_term_per_thresh=rebound_short_term_per_thresh,
    )
    events = filter_event_tuples_in_sp500(events, membership=membership)
    print(f"S&P 500 (point-in-time) bargain events after suppress: {len(events):,}")
    return events, universe_df, ever


def sp500_rebound_by_horizon(
    trend_down_thresh: float = config.trend_down_thresh,
    wait_for_short_term_rebound: bool = config.WAIT_FOR_SHORT_TERM_REBOUND,
    weeks: list[int] | None = None,
    since: str = STATS_SINCE,
) -> pd.DataFrame:
    """
    Attach future prices at 1..52 weeks for point-in-time S&P 500 and
    all-ticker trend-down bargains; aggregate return stats at each horizon.

    Saves a CSV (group column) and a chart comparing both universes.
    """
    weeks = list(weeks) if weeks is not None else HORIZON_WEEKS

    print("=" * 72)
    print("Rebound by horizon: S&P 500 (PIT) vs all tickers (1–52 weeks)")
    print("=" * 72)
    print(f"  trend_down_thresh           = {trend_down_thresh}")
    print(f"  WAIT_FOR_SHORT_TERM_REBOUND = {wait_for_short_term_rebound}")
    print(f"  stats since                 = {since}")
    print(f"  horizons (weeks)            = {weeks[0]}..{weeks[-1]}")
    print(f"  membership                  = {SP500_TICKER_START_END_FILE.name}")
    print("=" * 72)

    daily_df = load_or_build_daily_prices()
    membership = load_sp500_membership_intervals()
    price_tickers = set(daily_df["ticker"].unique())
    ever = ever_sp500_tickers(membership, price_tickers=price_tickers)
    print(f"Ever-S&P 500 tickers with price data: {len(ever):,}")

    print("\nRunning trend-down sieve on all tickers ...")
    events_all = collect_bargain_events(
        daily_df,
        trend_down_thresh=trend_down_thresh,
        wait_for_short_term_rebound=wait_for_short_term_rebound,
        rebound_short_term_per_thresh=config.rebound_short_term_per_thresh,
    )
    events_sp500 = filter_event_tuples_in_sp500(events_all, membership=membership)
    print(f"All-ticker events after suppress: {len(events_all):,}")
    print(f"S&P 500 (point-in-time) events:   {len(events_sp500):,}")

    if not events_all and not events_sp500:
        print("No bargain events; nothing to plot.")
        return pd.DataFrame()

    groups = [
        ("sp500", events_sp500),
        ("all", events_all),
    ]

    rows: list[dict] = []
    for week in weeks:
        future_days = week * DAYS_PER_WEEK
        print(f"\n--- horizon: {week} week(s) = {future_days}d ---")
        for group_name, events in groups:
            if not events:
                continue
            result_df = attach_future_prices(
                events,
                daily_df,
                future_change=future_days,
            )
            summary = stats_since(result_df, since=since)
            rows.append(
                {
                    "group": group_name,
                    "weeks": week,
                    "future_days": future_days,
                    **summary,
                }
            )
            print(
                f"  {group_name:12s} since {since}: "
                f"n={summary['n_with_future_since']:,}, "
                f"median={summary['median_price_change']:.3f}, "
                f"mean={summary['mean_price_change']:.3f}, "
                f"pct_rebound(>1)={summary['pct_rebound_gt_1']:.1f}%"
            )

    table = pd.DataFrame(rows)
    STOOQ_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(SP500_HORIZON_FILE, index=False)
    print(f"\nSaved horizon table to {SP500_HORIZON_FILE}")

    _plot_rebound_horizon_comparison(table, since=since)
    return table


def _plot_rebound_horizon_comparison(
    table: pd.DataFrame,
    since: str = STATS_SINCE,
    plot_path: Path = SP500_HORIZON_PLOT,
) -> None:
    """Plot S&P 500 (PIT) vs all-ticker median/mean price_change and pct_rebound."""
    if table.empty:
        return

    styles = {
        "sp500": {
            "median": {"color": "#1f4e79", "marker": "o", "label": "S&P 500 median"},
            "mean": {"color": "#5dade2", "marker": "s", "label": "S&P 500 mean"},
            "pct": {
                "color": "#1f4e79",
                "marker": "o",
                "label": "S&P 500 pct rebound (>1)",
            },
        },
        "all": {
            "median": {"color": "#c0392b", "marker": "o", "label": "All median"},
            "mean": {"color": "#e67e22", "marker": "s", "label": "All mean"},
            "pct": {"color": "#c0392b", "marker": "o", "label": "All pct rebound (>1)"},
        },
    }

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax0, ax1 = axes

    for group_name, style in styles.items():
        g = table[table["group"] == group_name].sort_values("weeks")
        if g.empty:
            continue
        ax0.plot(
            g["weeks"],
            g["median_price_change"],
            markersize=3,
            linewidth=1.5,
            **style["median"],
        )
        ax0.plot(
            g["weeks"],
            g["mean_price_change"],
            markersize=3,
            linewidth=1.5,
            linestyle="--",
            **style["mean"],
        )
        ax1.plot(
            g["weeks"],
            g["pct_rebound_gt_1"],
            markersize=3,
            linewidth=1.5,
            **style["pct"],
        )

    ax0.axhline(1.0, color="#888888", linestyle=":", linewidth=1.0, label="breakeven (1.0)")
    ax0.set_ylabel("price_change (future / buy)")
    ax0.set_title(
        f"Trend-down bargains: S&P 500 (PIT) vs all — return vs holding horizon "
        f"(events since {since})"
    )
    ax0.legend(loc="best", fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax1.axhline(50.0, color="#888888", linestyle=":", linewidth=1.0, label="50%")
    ax1.set_xlabel("Holding horizon (weeks)")
    ax1.set_ylabel("% of events with price_change > 1")
    ax1.set_xlim(table["weeks"].min(), table["weeks"].max())
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    notes = []
    for group_name in ("sp500", "all"):
        g = table[table["group"] == group_name].sort_values("weeks")
        if g.empty:
            continue
        notes.append(
            f"{group_name}: n={int(g['n_with_future_since'].iloc[0]):,} (1w) → "
            f"{int(g['n_with_future_since'].iloc[-1]):,} ({int(g['weeks'].iloc[-1])}w)"
        )
    if notes:
        ax1.text(
            0.99,
            0.05,
            "\n".join(notes),
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
        )

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")
    plt.show()


def iter_param_grid():
    """Yield (thresh, wait, rebound_thresh) combos; skip redundant rebound when wait=False."""
    for thresh, wait in product(
        TREND_DOWN_THRESH_VALUES, WAIT_FOR_SHORT_TERM_REBOUND_VALUES
    ):
        if wait:
            for rebound in REBOUND_SHORT_TERM_PER_THRESH_VALUES:
                yield thresh, wait, rebound
        else:
            yield thresh, wait, None


def trend_down_param_sweep() -> None:
    print("=" * 72)
    print("Trend-down parameter sweep")
    print("=" * 72)
    print(f"  trend_down_thresh grid           = {TREND_DOWN_THRESH_VALUES}")
    print(f"  rebound_short_term_per_thresh    = {REBOUND_SHORT_TERM_PER_THRESH_VALUES}")
    print(f"  WAIT_FOR_SHORT_TERM_REBOUND      = {WAIT_FOR_SHORT_TERM_REBOUND_VALUES}")
    print(f"  daily cache                      = {DAILY_PRICES_CACHE}")
    print(f"  (fixed) trend_down_history       = {config.trend_down_history}d")
    print(f"  (fixed) trending_down_suppression = {config.trending_down_suppression}d")
    print(f"  (fixed) rebound_short_term_days  = {config.rebound_short_term_days}")
    print(f"  (fixed) future_change            = {config.future_change}d")
    print("=" * 72)

    daily_df = load_or_build_daily_prices()

    rows: list[dict] = []
    combos = list(iter_param_grid())
    for i, (thresh, wait, rebound) in enumerate(combos, start=1):
        print("\n" + "-" * 72)
        print(
            f"[{i}/{len(combos)}] thresh={thresh}, wait={wait}, "
            f"rebound_thresh={rebound}"
        )
        print("-" * 72)
        row = run_one(
            daily_df,
            trend_down_thresh=thresh,
            wait_for_short_term_rebound=wait,
            rebound_short_term_per_thresh=(
                rebound if rebound is not None else config.rebound_short_term_per_thresh
            ),
        )
        rows.append(row)
        print(
            f"  since {STATS_SINCE}: n={row['n_with_future_since']:,}, "
            f"median={row['median_price_change']:.3f}, "
            f"mean={row['mean_price_change']:.3f}, "
            f"pct_rebound(>1)={row['pct_rebound_gt_1']:.1f}%"
        )

    results = pd.DataFrame(rows)
    results = results.sort_values(
        ["WAIT_FOR_SHORT_TERM_REBOUND", "trend_down_thresh", "rebound_short_term_per_thresh"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    print("\n" + "=" * 72)
    print(f"Results (price-change stats since {STATS_SINCE})")
    print("=" * 72)
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    results.to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved table to {RESULTS_FILE}")


if __name__ == "__main__":
    sp500_rebound_by_horizon()

