#!/usr/bin/env python3
"""
Find stocks whose daily close has fallen below a historic average,
require a short-term rebound, suppress near-duplicate flags, and record
a longer-horizon future price for modeling.

Pipeline:
  1. Load daily closes from data/stock_Stooq_daily_US/.
  2. Flag each trading day where
     price <= historic_avg * (1 - trend_down_thresh),
     with historic_avg = mean close over the prior trend_down_history days.
  3. Keep only flags that, by the end of rebound_short_term_days, have
     rebounded by rebound_short_term_per_thresh; replace date/price with
     the window-end values.
  4. Suppress repeat flags for the same ticker within
     trending_down_suppression days (keeps the first detection date).
  5. Attach the price future_change days ahead (when available).

Usage:
  python trend_down_stocks.py              # full sieve
  python trend_down_stocks.py --today      # S&P 500 (PIT) bargains on latest day
  python trend_down_stocks.py --today 2026-07-22  # same, for a specific date
  python trend_down_stocks.py --debug      # plot INTU flags
  python trend_down_stocks.py --debug AAPL # plot another ticker
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import config
from utilities.stock_stooq import (
    STOCK_EXCHANGES,
    STOOQ_BASE_DIR,
    STOOQ_SAVE_DIR,
    process_stock_directory,
)

OUTPUT_FILE = STOOQ_SAVE_DIR / "trending_down_stocks.csv"
OUTPUT_FILE_DEBUG = STOOQ_SAVE_DIR / "debug_trending_down_stocks.csv"

_EVENT_COLUMNS = ["ticker", "date", "price", "historical_average"]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_daily_prices(base_dir: Path = STOOQ_BASE_DIR) -> pd.DataFrame:
    """Daily closes for all stocks under base_dir (ticker, date, close_price)."""
    print(f"Loading daily prices from {base_dir} ...")
    if not base_dir.is_dir():
        raise FileNotFoundError(
            f"Stooq data directory not found: {base_dir}. "
            "Run utilities/stock_stooq.py first."
        )

    frames: list[pd.DataFrame] = []
    for pattern in STOCK_EXCHANGES:
        df = process_stock_directory(pattern, base_dir=base_dir, verbose=False)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError(f"No stock price files found under {base_dir}")

    daily = pd.concat(frames, ignore_index=True)
    daily["ticker"] = daily["ticker"].str.upper().str.replace(".US", "", regex=False)
    daily = (
        daily[["ticker", "date", "close_price"]]
        .dropna(subset=["date", "close_price"])
        .sort_values(["ticker", "date"])
        .drop_duplicates(subset=["ticker", "date"], keep="last")
        .reset_index(drop=True)
    )
    print(
        f"Loaded {len(daily):,} daily rows for "
        f"{daily['ticker'].nunique():,} tickers "
        f"({daily['date'].min().date()} → {daily['date'].max().date()})"
    )
    return daily


# =============================================================================
# TREND-DOWN DETECTION
# =============================================================================

def find_trending_down_stocks(
    daily_df: pd.DataFrame,
    trend_down_thresh: float = config.trend_down_thresh,
    trend_down_history: int = config.trend_down_history,
) -> list[tuple]:
    """
    Flag every trading day where price is trend_down_thresh below the average
    of closes in the prior trend_down_history calendar days.

    Returns:
        trending_down_stocks: list of
        (ticker, date, price_on_date, historical_average)
    """
    if daily_df.empty:
        return []

    window = f"{trend_down_history}D"
    # Require roughly half the trading days expected in that calendar window.
    min_periods = max(trend_down_history * 5 // 14, 1)

    parts: list[pd.DataFrame] = []
    for ticker, group in daily_df.groupby("ticker", sort=False):
        g = group.sort_values("date").set_index("date")
        # Exclude today: average of prior window only.
        hist_avg = (
            g["close_price"]
            .shift(1)
            .rolling(window=window, min_periods=min_periods)
            .mean()
        )
        flagged = g.assign(historical_average=hist_avg)
        threshold = flagged["historical_average"] * (1.0 - trend_down_thresh)
        flagged = flagged[
            flagged["historical_average"].notna()
            & (flagged["close_price"] <= threshold)
        ]
        if not flagged.empty:
            parts.append(
                flagged.reset_index()[
                    ["ticker", "date", "close_price", "historical_average"]
                ]
            )

    if not parts:
        print("Flagged 0 trend-down events")
        return []

    flagged_df = pd.concat(parts, ignore_index=True)
    trending_down_stocks: list[tuple] = [
        (
            row.ticker,
            row.date,
            float(row.close_price),
            float(row.historical_average),
        )
        for row in flagged_df.itertuples(index=False)
    ]
    print(
        f"Flagged {len(trending_down_stocks):,} trend-down events "
        f"(thresh={trend_down_thresh}, history={trend_down_history}d)"
    )
    return trending_down_stocks


def _lookup_price_at_target(
    events: pd.DataFrame,
    daily_df: pd.DataFrame,
    target_col: str,
    direction: str = "forward",
    out_date_col: str = "lookup_date",
    out_price_col: str = "lookup_price",
) -> pd.DataFrame:
    """
    For each event row, look up the close on/after (forward) or on/before
    (backward) the date in target_col. Same merge_asof-per-ticker pattern
    used for future-price attachment.
    """
    prices = daily_df[["ticker", "date", "close_price"]].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.rename(columns={"date": out_date_col, "close_price": out_price_col})

    pieces: list[pd.DataFrame] = []
    price_groups = {t: g for t, g in prices.groupby("ticker", sort=False)}

    for ticker, ev in events.groupby("ticker", sort=False):
        ev = ev.sort_values(target_col).reset_index(drop=True)
        pr = price_groups.get(ticker)
        if pr is None or pr.empty:
            piece = ev.copy()
            piece[out_date_col] = pd.NaT
            piece[out_price_col] = pd.NA
            pieces.append(piece)
            continue

        pr = pr.sort_values(out_date_col).reset_index(drop=True)
        piece = pd.merge_asof(
            ev.drop(columns=["ticker"]),
            pr.drop(columns=["ticker"]),
            left_on=target_col,
            right_on=out_date_col,
            direction=direction,
        )
        piece.insert(0, "ticker", ticker)
        pieces.append(piece)

    return pd.concat(pieces, ignore_index=True) if pieces else events.copy()


def filter_short_term_rebounds(
    trending_down_stocks: list[tuple],
    daily_df: pd.DataFrame,
    rebound_short_term_days: int = config.rebound_short_term_days,
    rebound_short_term_per_thresh: float = config.rebound_short_term_per_thresh,
) -> list[tuple]:
    """
    Keep flags whose close at the end of the short-term window has risen by
    at least rebound_short_term_per_thresh.

    For kept events, date/price are replaced with the window-end lookup values.
    historical_average is unchanged (from the original flag day).

    Returns tuples: (ticker, date, price, historical_average)
    """
    if not trending_down_stocks:
        return []

    events = pd.DataFrame(
        trending_down_stocks,
        columns=["ticker", "date", "price", "historical_average"],
    )
    events["date"] = pd.to_datetime(events["date"])
    events["flag_price"] = events["price"]
    events["target_date"] = events["date"] + pd.Timedelta(days=rebound_short_term_days)
    events["rebound_floor"] = events["flag_price"] * (
        1.0 + rebound_short_term_per_thresh
    )

    looked_up = _lookup_price_at_target(
        events,
        daily_df,
        target_col="target_date",
        direction="forward",
        out_date_col="rebound_date",
        out_price_col="rebound_price",
    )

    kept_df = looked_up[
        looked_up["rebound_price"].notna()
        & (looked_up["rebound_price"] >= looked_up["rebound_floor"])
    ].copy()
    kept_df["date"] = kept_df["rebound_date"]
    kept_df["price"] = kept_df["rebound_price"]

    kept: list[tuple] = [
        (
            row.ticker,
            row.date,
            float(row.price),
            float(row.historical_average),
        )
        for row in kept_df.itertuples(index=False)
    ]
    print(
        f"Short-term rebound filter "
        f"({rebound_short_term_days}d, +{rebound_short_term_per_thresh:.1%}): "
        f"kept {len(kept):,} / {len(trending_down_stocks):,}"
    )
    return kept


def suppress_similar_tuples(
    trending_down_stocks: list[tuple],
    trending_down_suppression: int = config.trending_down_suppression,
) -> list[tuple]:
    """
    Keep the first flag per ticker, then ignore further flags for the same
    ticker until trending_down_suppression calendar days have elapsed.
    """
    if not trending_down_stocks:
        return []

    events = sorted(trending_down_stocks, key=lambda t: (t[0], pd.Timestamp(t[1])))
    kept: list[tuple] = []
    last_kept_date: dict[str, pd.Timestamp] = {}
    suppression = pd.Timedelta(days=trending_down_suppression)

    for event in events:
        ticker, event_date = event[0], event[1]
        event_ts = pd.Timestamp(event_date)
        prev = last_kept_date.get(ticker)
        if prev is not None and event_ts < prev + suppression:
            continue
        kept.append(event)
        last_kept_date[ticker] = event_ts

    print(
        f"Suppressed {len(trending_down_stocks) - len(kept):,} near-duplicate events "
        f"(window={trending_down_suppression}d); kept {len(kept):,}"
    )
    return kept


def attach_future_prices(
    trending_down_stocks: list[tuple],
    daily_df: pd.DataFrame,
    future_change: int = config.future_change,
) -> pd.DataFrame:
    """
    For each event, look up the close on/after date + future_change days.

    Returns columns: ticker, date, price, historical_average,
    future_date, future_price, price_change
    where price_change = future_price / price (NaN if future price missing).
    """
    empty = pd.DataFrame(
        columns=[*_EVENT_COLUMNS, "future_date", "future_price", "price_change"]
    )
    if not trending_down_stocks:
        return empty

    events = pd.DataFrame(trending_down_stocks, columns=_EVENT_COLUMNS)
    events["date"] = pd.to_datetime(events["date"])
    events["target_date"] = events["date"] + pd.Timedelta(days=future_change)

    result = _lookup_price_at_target(
        events,
        daily_df,
        target_col="target_date",
        direction="forward",
        out_date_col="future_date",
        out_price_col="future_price",
    )
    result["price_change"] = result["future_price"] / result["price"]
    result = result[[*_EVENT_COLUMNS, "future_date", "future_price", "price_change"]]

    n_with_future = result["future_price"].notna().sum()
    print(
        f"Attached future prices ({future_change}d ahead): "
        f"{n_with_future:,} / {len(result):,} events have a future price"
    )
    return result


# =============================================================================
# CURRENT-DAY BARGAINS
# =============================================================================

def current_day_bargains(
    daily_df: pd.DataFrame | None = None,
    as_of: str | pd.Timestamp | None = None,
    sp500_only: bool = True,
) -> pd.DataFrame:
    """
    Run the trend-down sieve and return bargains flagged on `as_of`.

    If `as_of` is None, uses the latest trading day present in `daily_df`.
    Applies the same optional short-term rebound filter and suppression as main().
    By default (`sp500_only=True`), keeps only tickers in the S&P 500 on `as_of`
    (point-in-time membership from data/sp500/).

    Columns: ticker, date, price, historical_average, discount
    (discount = 1 - price / historical_average).
    """
    # Lazy import avoids a circular dependency with experiment_trend_down_params.
    from experiment_trend_down_params import (
        ever_sp500_tickers,
        filter_events_in_sp500,
        load_sp500_membership_intervals,
    )

    if daily_df is None:
        daily_df = load_daily_prices()
    else:
        daily_df = daily_df.copy()

    daily_df["date"] = pd.to_datetime(daily_df["date"])
    data_end = daily_df["date"].max()
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else data_end
    as_of_ts = as_of_ts.normalize()

    if as_of_ts > data_end:
        raise ValueError(
            f"as_of {as_of_ts.date()} is after the latest price date "
            f"{data_end.date()}. Refresh Stooq data or pick an earlier date."
        )

    # Avoid look-ahead: only use prices on/before the evaluation day.
    daily_df = daily_df[daily_df["date"] <= as_of_ts].copy()
    empty = pd.DataFrame(columns=[*_EVENT_COLUMNS, "discount"])
    if daily_df.empty:
        return empty

    membership = None
    if sp500_only:
        membership = load_sp500_membership_intervals()
        ever = ever_sp500_tickers(
            membership, price_tickers=set(daily_df["ticker"].unique())
        )
        daily_df = daily_df[daily_df["ticker"].isin(ever)].copy()
        if daily_df.empty:
            return empty

    trending_down_stocks = find_trending_down_stocks(daily_df)
    if config.WAIT_FOR_SHORT_TERM_REBOUND:
        trending_down_stocks = filter_short_term_rebounds(
            trending_down_stocks, daily_df
        )
    trending_down_stocks = suppress_similar_tuples(trending_down_stocks)

    if not trending_down_stocks:
        return empty

    events = pd.DataFrame(trending_down_stocks, columns=_EVENT_COLUMNS)
    events["date"] = pd.to_datetime(events["date"])
    today = events[events["date"] == as_of_ts].copy()
    if today.empty:
        trading_days = daily_df["date"].drop_duplicates().sort_values()
        if as_of_ts not in set(trading_days):
            nearest = trading_days[trading_days <= as_of_ts]
            hint = (
                f" (nearest prior trading day in data: {nearest.iloc[-1].date()})"
                if len(nearest)
                else ""
            )
            print(f"Note: {as_of_ts.date()} is not a trading day in the price data{hint}.")
        return empty

    if sp500_only:
        n_before = len(today)
        today = filter_events_in_sp500(today, membership=membership)
        print(
            f"S&P 500 (point-in-time) filter on {as_of_ts.date()}: "
            f"kept {len(today):,} / {n_before:,}"
        )
        if today.empty:
            return empty

    today["discount"] = 1.0 - today["price"] / today["historical_average"]
    return today.sort_values(["discount", "ticker"], ascending=[False, True]).reset_index(
        drop=True
    )


def print_current_bargains(
    as_of: str | pd.Timestamp | None = None,
    sp500_only: bool = True,
) -> pd.DataFrame:
    """Load prices, find bargains on the current (or given) day, and print them."""
    universe = "S&P 500 (point-in-time)" if sp500_only else "all tickers"
    print("=" * 72)
    print(f"Current-day trend-down bargains — {universe}")
    print("=" * 72)
    print(f"  trend_down_thresh                = {config.trend_down_thresh}")
    print(f"  trend_down_history               = {config.trend_down_history}d")
    print(f"  WAIT_FOR_SHORT_TERM_REBOUND      = {config.WAIT_FOR_SHORT_TERM_REBOUND}")
    print(f"  rebound_short_term_days          = {config.rebound_short_term_days}")
    print(f"  rebound_short_term_per_thresh    = {config.rebound_short_term_per_thresh}")
    print(f"  trending_down_suppression        = {config.trending_down_suppression}d")
    print(f"  sp500_only                       = {sp500_only}")
    print("=" * 72)

    daily_df = load_daily_prices()
    data_end = pd.to_datetime(daily_df["date"]).max()
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else data_end
    print(f"Evaluating bargains for: {as_of_ts.date()}")
    print(f"Price data through:      {data_end.date()}")

    bargains = current_day_bargains(
        daily_df=daily_df, as_of=as_of_ts, sp500_only=sp500_only
    )
    print(f"\nBargains on {as_of_ts.date()}: {len(bargains):,}")
    if bargains.empty:
        print("None.")
    else:
        display = bargains.copy()
        display["date"] = display["date"].dt.strftime("%Y-%m-%d")
        display["price"] = display["price"].map(lambda x: f"{x:.4f}")
        display["historical_average"] = display["historical_average"].map(
            lambda x: f"{x:.4f}"
        )
        display["discount"] = display["discount"].map(lambda x: f"{x:.1%}")
        print(display.to_string(index=False))
    print("=" * 72)
    return bargains


# =============================================================================
# DEBUG / VISUALIZATION
# =============================================================================

DEFAULT_DEBUG_TICKER = "INTU"


def debug_trend_down(ticker: str = DEFAULT_DEBUG_TICKER) -> None:
    """
    Step-by-step debug for one ticker:
      1. load_daily_prices → find_trending_down_stocks → suppress_similar_tuples
      2. attach_future_prices → write CSV to OUTPUT_FILE_DEBUG
      3. plot close price with flagged trend-down points marked
    """
    ticker = ticker.upper()
    plot_file = STOOQ_SAVE_DIR / f"debug_trend_down_{ticker}.png"

    print("=" * 72)
    print(f"Debug trend_down_stocks — {ticker}")
    print("=" * 72)
    print(f"  trend_down_thresh           = {config.trend_down_thresh}")
    print(f"  trend_down_history          = {config.trend_down_history}d")
    print(f"  trending_down_suppression   = {config.trending_down_suppression}d")
    print(f"  future_change               = {config.future_change}d")
    print("=" * 72)

    daily_df = load_daily_prices()
    ticker_df = daily_df[daily_df["ticker"] == ticker].copy()
    if ticker_df.empty:
        raise RuntimeError(
            f"No daily prices found for {ticker}. "
            "Check data/stock_Stooq_daily_US/."
        )
    print(
        f"{ticker}: {len(ticker_df):,} daily rows "
        f"({ticker_df['date'].min().date()} → {ticker_df['date'].max().date()})"
    )

    trending_down_stocks = find_trending_down_stocks(ticker_df)
    trending_down_stocks = suppress_similar_tuples(trending_down_stocks)

    events = attach_future_prices(trending_down_stocks, ticker_df)
    print(f"\n{ticker} trend-down events after suppression: {len(events)}")
    if not events.empty:
        print(events.to_string(index=False))

    STOOQ_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUTPUT_FILE_DEBUG, index=False)
    print(f"Saved {len(events):,} rows to {OUTPUT_FILE_DEBUG}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        ticker_df["date"],
        ticker_df["close_price"],
        color="#1f4e79",
        linewidth=1.0,
        label=f"{ticker} close",
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
        f"{ticker}: trend-down flags "
        f"(thresh={config.trend_down_thresh}, "
        f"history={config.trend_down_history}d, "
        f"suppress={config.trending_down_suppression}d)"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Close price")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(plot_file, dpi=150)
    print(f"\nSaved plot to {plot_file}")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 72)
    print("Trend-down stock sieve (daily)")
    print("=" * 72)
    print(f"  trend_down_thresh                = {config.trend_down_thresh}")
    print(f"  trend_down_history               = {config.trend_down_history}d")
    print(f"  WAIT_FOR_SHORT_TERM_REBOUND      = {config.WAIT_FOR_SHORT_TERM_REBOUND}")
    print(f"  rebound_short_term_days          = {config.rebound_short_term_days}")
    print(f"  rebound_short_term_per_thresh    = {config.rebound_short_term_per_thresh}")
    print(f"  trending_down_suppression        = {config.trending_down_suppression}d")
    print(f"  future_change                    = {config.future_change}d")
    print("=" * 72)

    if OUTPUT_FILE.exists():
        print(f"Output file {OUTPUT_FILE} already exists. Skipping flagging stocks.")
        print(f"Loading existing data from {OUTPUT_FILE}...")
        result_df = pd.read_csv(OUTPUT_FILE)
    else:
        daily_df = load_daily_prices()

        trending_down_stocks = find_trending_down_stocks(daily_df)
        if config.WAIT_FOR_SHORT_TERM_REBOUND:
            trending_down_stocks = filter_short_term_rebounds(
                trending_down_stocks, daily_df
            )
        trending_down_stocks = suppress_similar_tuples(trending_down_stocks)
        result_df = attach_future_prices(trending_down_stocks, daily_df)

        STOOQ_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved {len(result_df):,} rows to {OUTPUT_FILE}")

    if not result_df.empty:
        with_future = result_df.dropna(subset=["price_change"]).copy()
        with_future["date"] = pd.to_datetime(with_future["date"])
        with_future = with_future[with_future["date"] >= "2010-01-01"]
        if not with_future.empty:
            print(
                f"Price-change stats since 2010 (future/current): "
                f"median={with_future['price_change'].median():.3f}, "
                f"mean={with_future['price_change'].mean():.3f}, "
                f"pct_rebound(>1)="
                f"{(with_future['price_change'] > 1).mean() * 100:.1f}%"
            )
        else:
            print("No events with future prices since 2010 to summarize.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trend-down stock sieve")
    parser.add_argument(
        "--debug",
        nargs="?",
        const=DEFAULT_DEBUG_TICKER,
        metavar="TICKER",
        help=f"Debug/plot one ticker (default: {DEFAULT_DEBUG_TICKER})",
    )
    parser.add_argument(
        "--today",
        nargs="?",
        const="LATEST",
        metavar="YYYY-MM-DD",
        help=(
            "Print S&P 500 (point-in-time) bargains for the latest trading day "
            "in the data (or for YYYY-MM-DD if given)"
        ),
    )
    parser.add_argument(
        "--all-tickers",
        action="store_true",
        help="With --today, include all tickers (skip S&P 500 filter)",
    )
    args = parser.parse_args()
    if args.debug is not None:
        debug_trend_down(args.debug)
    elif args.today is not None:
        print_current_bargains(
            None if args.today == "LATEST" else args.today,
            sp500_only=not args.all_tickers,
        )
    else:
        main()
