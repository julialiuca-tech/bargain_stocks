#!/usr/bin/env python3
"""
Step-by-step debug / visualization for trend_down_stocks.py.

Step 1: load_daily_prices → find_trending_down_stocks → suppress_similar_tuples
Step 2: plot INTU with flagged trend-down points marked

Usage:
  python debug_trend_down_stocks.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

import config
from trend_down_stocks import (
    find_trending_down_stocks,
    load_daily_prices,
    suppress_similar_tuples,
)
from utilities.stock_stooq import STOOQ_SAVE_DIR

DEBUG_TICKER = "INTU"
PLOT_FILE = STOOQ_SAVE_DIR / f"debug_trend_down_{DEBUG_TICKER}.png"


def main() -> None:
    print("=" * 72)
    print("Debug trend_down_stocks — step 1+2 (INTU)")
    print("=" * 72)
    print(f"  trend_down_thresh           = {config.trend_down_thresh}")
    print(f"  trend_down_history          = {config.trend_down_history}d")
    print(f"  trending_down_suppression   = {config.trending_down_suppression}d")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # Step 1: load → find → suppress (scoped to INTU for a fast debug loop)
    # -------------------------------------------------------------------------
    daily_df = load_daily_prices()
    intu_df = daily_df[daily_df["ticker"] == DEBUG_TICKER].copy()
    if intu_df.empty:
        raise RuntimeError(
            f"No daily prices found for {DEBUG_TICKER}. "
            "Check data/stock_Stooq_daily_US/."
        )
    print(
        f"{DEBUG_TICKER}: {len(intu_df):,} daily rows "
        f"({intu_df['date'].min().date()} → {intu_df['date'].max().date()})"
    )

    trending_down_stocks = find_trending_down_stocks(intu_df)
    trending_down_stocks = suppress_similar_tuples(trending_down_stocks)

    events = pd.DataFrame(
        trending_down_stocks,
        columns=["ticker", "date", "price", "historical_average"],
    )
    print(f"\n{DEBUG_TICKER} trend-down events after suppression: {len(events)}")
    if not events.empty:
        print(events.to_string(index=False))

    # -------------------------------------------------------------------------
    # Step 2: plot INTU chart and mark trend-down points
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        intu_df["date"],
        intu_df["close_price"],
        color="#1f4e79",
        linewidth=1.0,
        label=f"{DEBUG_TICKER} close",
    )

    if not events.empty:
        ax.scatter(
            events["date"],
            events["price"],
            color="#c0392b",
            s=40,
            zorder=3,
            label="trend-down flag",
        )
        for row in events.itertuples(index=False):
            ax.annotate(
                row.date.strftime("%Y-%m-%d"),
                (row.date, row.price),
                textcoords="offset points",
                xytext=(5, 8),
                fontsize=7,
                color="#c0392b",
            )

    ax.set_title(
        f"{DEBUG_TICKER}: trend-down flags "
        f"(thresh={config.trend_down_thresh}, "
        f"history={config.trend_down_history}d, "
        f"suppress={config.trending_down_suppression}d)"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Close price")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    STOOQ_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_FILE, dpi=150)
    print(f"\nSaved plot to {PLOT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()
