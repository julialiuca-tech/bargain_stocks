#!/usr/bin/env python3
"""
Grid-search trend_down_stocks parameters and tabulate since-2010 stats.

Heavy step (load_daily_prices) runs once and is cached to disk; subsequent
runs reuse the cache.

Usage:
  python experiment_trend_down_params.py
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

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
FORTUNE500_TICKERS_FILE = (
    Path(__file__).resolve().parent / "data" / "fortune500" / "fortune500_tickers.csv"
)
FORTUNE500_COMPARE_FILE = STOOQ_SAVE_DIR / "trend_down_fortune500_compare.csv"

# Parameter grid (edit here to experiment).
TREND_DOWN_THRESH_VALUES = [0.05, 0.10, 0.15, 0.20]  # 5% … 20%
REBOUND_SHORT_TERM_PER_THRESH_VALUES = [0.03, 0.05]
WAIT_FOR_SHORT_TERM_REBOUND_VALUES = [True, False]

STATS_SINCE = "2010-01-01"


def load_or_build_daily_prices(cache_path: Path = DAILY_PRICES_CACHE) -> pd.DataFrame:
    """Load daily prices from cache, or build once via load_daily_prices()."""
    STOOQ_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f"Loading cached daily prices from {cache_path} ...")
        daily_df = pd.read_pickle(cache_path)
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


def load_fortune500_tickers(
    path: Path = FORTUNE500_TICKERS_FILE,
    price_tickers: set[str] | None = None,
) -> set[str]:
    """Load mapped Fortune 500 tickers; optionally intersect with available prices."""
    if not path.exists():
        raise FileNotFoundError(
            f"Fortune 500 ticker map not found: {path}. "
            "Expected data/fortune500/fortune500_tickers.csv"
        )
    df = pd.read_csv(path)
    tickers = set(df["ticker"].astype(str).str.upper())
    # Stooq uses e.g. BRK.B; SEC map may have BRK-B
    normalized = set()
    for t in tickers:
        normalized.add(t)
        normalized.add(t.replace("-", "."))
        normalized.add(t.replace(".", "-"))
    if price_tickers is not None:
        normalized = {t for t in normalized if t in price_tickers}
    return normalized


def compare_fortune500_vs_rest(
    trend_down_thresh: float = 0.15,
    wait_for_short_term_rebound: bool = False,
) -> pd.DataFrame:
    """
    Run one config and tabulate since-2010 stats for Fortune 500 vs other tickers.
    """
    print("=" * 72)
    print("Fortune 500 vs rest — trend-down stats")
    print("=" * 72)
    print(f"  trend_down_thresh           = {trend_down_thresh}")
    print(f"  WAIT_FOR_SHORT_TERM_REBOUND = {wait_for_short_term_rebound}")
    print(f"  stats since                 = {STATS_SINCE}")
    print("=" * 72)

    daily_df = load_or_build_daily_prices()
    price_tickers = set(daily_df["ticker"].unique())
    f500 = load_fortune500_tickers(price_tickers=price_tickers)
    print(
        f"Fortune 500 tickers with price data: {len(f500):,} "
        f"(from {FORTUNE500_TICKERS_FILE.name})"
    )

    result_df = run_pipeline(
        daily_df,
        trend_down_thresh=trend_down_thresh,
        wait_for_short_term_rebound=wait_for_short_term_rebound,
        rebound_short_term_per_thresh=config.rebound_short_term_per_thresh,
    )
    result_df = result_df.copy()
    result_df["ticker"] = result_df["ticker"].astype(str).str.upper()
    result_df["is_fortune500"] = result_df["ticker"].isin(f500)

    rows = []
    for label, mask in [
        ("all", slice(None)),
        ("fortune500", result_df["is_fortune500"]),
        ("non_fortune500", ~result_df["is_fortune500"]),
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

    table.to_csv(FORTUNE500_COMPARE_FILE, index=False)
    print(f"\nSaved table to {FORTUNE500_COMPARE_FILE}")
    return table


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


def main() -> None:
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
    main()
