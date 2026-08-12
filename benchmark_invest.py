#!/usr/bin/env python3
"""
Simulate investment strategies since 2000 and compare end wealth.

Strategy 1 — S&P 500 (point-in-time) bargain stocks:
  Use trend_down_stocks events, limited to tickers that were in the S&P 500
  on the bargain date (historical membership from data/sp500/). Buy $1 of
  each bargain at the next trading day's open (signal is seen at the close);
  sell at the trading day closest to HOLD_MONTHS after the buy date.

Strategy 2 — index benchmark:
  On each Strategy-1 buy date, buy $1 of the Dow proxy (DIA ETF) at that
  day's open instead; sell at the trading day closest to HOLD_MONTHS later.
  Strategies 1 and 2 are paired 1:1 on the same signals (only signals where
  both the bargain ticker and DIA are tradable are kept).

Assumptions: fractional shares allowed, no taxes/fees, start cash $1,000,000.

Note: Stooq's US daily package does not include the ^DJI index series.
DJIA.US (ETF) only starts in 2022, so DIA.US is used as the Dow proxy
(available from 2005-02-25 in this dataset).

Usage:
  python benchmark_invest.py
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import config
from experiment_trend_down_params import (
    filter_events_in_sp500,
    load_or_build_daily_prices,
    load_sp500_membership_intervals,
    run_pipeline,
)
from trend_down_stocks import OUTPUT_FILE
from utilities.stock_stooq import STOOQ_BASE_DIR, STOOQ_SAVE_DIR, process_stock_directory

# =============================================================================
# CONSTANTS
# =============================================================================

START_CASH = 1_000_000.0
INVEST_AMOUNT = 1.0
HORIZON_START = pd.Timestamp("2000-01-01")
HOLD_MONTHS = 6

# Dow Jones Industrial Average proxy in the Stooq US ETF tree.
INDEX_TICKER = "DIA"
INDEX_ETF_PATTERN = "nyse_etf*"

TXN_DIR = STOOQ_SAVE_DIR / "benchmark_invest"
STRATEGY1_TXN_FILE = TXN_DIR / "strategy1_bargain_sp500_transactions.csv"
STRATEGY2_TXN_FILE = TXN_DIR / "strategy2_index_transactions.csv"
SUMMARY_FILE = TXN_DIR / "benchmark_summary.csv"
YEARLY_RETURNS_PLOT_FILE = TXN_DIR / "strategy1_vs_2_yearly_returns.png"


# =============================================================================
# DATA HELPERS
# =============================================================================

def _read_trend_down_events(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Load or build the raw trend-down event table (all tickers)."""
    if OUTPUT_FILE.exists():
        print(f"Loading bargain events from {OUTPUT_FILE} ...")
        events = pd.read_csv(OUTPUT_FILE)
    else:
        print("No cached trending_down_stocks.csv; running trend-down pipeline ...")
        events = run_pipeline(
            daily_df,
            trend_down_thresh=config.trend_down_thresh,
            wait_for_short_term_rebound=config.WAIT_FOR_SHORT_TERM_REBOUND,
            rebound_short_term_per_thresh=config.rebound_short_term_per_thresh,
        )
        STOOQ_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        events.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved {len(events):,} rows to {OUTPUT_FILE}")

    events = events.copy()
    events["ticker"] = (
        events["ticker"].astype(str).str.upper().str.replace(".US", "", regex=False)
    )
    events["date"] = pd.to_datetime(events["date"])
    return events[events["date"] >= HORIZON_START].copy()


def load_bargain_events(
    daily_df: pd.DataFrame,
    sp500_only: bool = True,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Trend-down events on/after HORIZON_START.
    Optionally restrict to tickers in the S&P 500 on the event date
    (point-in-time membership; avoids survivor bias from a current roster).
    Pass `events` to avoid reloading when computing multiple universes.
    """
    if events is None:
        events = _read_trend_down_events(daily_df)
    else:
        events = events.copy()

    n_since = len(events)
    if sp500_only:
        membership = load_sp500_membership_intervals()
        events = filter_events_in_sp500(events, membership=membership)
        print(
            f"S&P 500 (point-in-time) bargains since {HORIZON_START.date()}: "
            f"{len(events):,} / {n_since:,} events"
        )
    else:
        print(
            f"All-ticker bargains since {HORIZON_START.date()}: "
            f"{len(events):,} events"
        )

    return events.sort_values(["date", "ticker"]).reset_index(drop=True)


def load_index_prices(
    ticker: str = INDEX_TICKER,
    base_dir: Path = STOOQ_BASE_DIR,
) -> pd.DataFrame:
    """Load daily open/close for the Dow proxy ETF (DIA)."""
    etfs = process_stock_directory(INDEX_ETF_PATTERN, base_dir=base_dir, verbose=False)
    if etfs.empty:
        raise FileNotFoundError(f"No ETF data under {base_dir / INDEX_ETF_PATTERN}")

    etfs["ticker"] = etfs["ticker"].str.upper().str.replace(".US", "", regex=False)
    cols = ["ticker", "date", "close_price"]
    if "open_price" in etfs.columns:
        cols.append("open_price")
    idx = (
        etfs.loc[etfs["ticker"] == ticker, cols]
        .dropna(subset=["date", "close_price"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    if idx.empty:
        raise RuntimeError(f"No price history found for index proxy {ticker}")
    print(
        f"Loaded {ticker} proxy: {len(idx):,} days "
        f"({idx['date'].min().date()} → {idx['date'].max().date()})"
    )
    return idx


def _price_maps(
    daily_df: pd.DataFrame,
    price_col: str = "close_price",
) -> dict[str, tuple[pd.DatetimeIndex, pd.Series]]:
    """ticker -> (sorted dates, price_col aligned to those dates)."""
    if price_col not in daily_df.columns:
        raise KeyError(f"daily_df is missing required column {price_col!r}")
    maps: dict[str, tuple[pd.DatetimeIndex, pd.Series]] = {}
    for ticker, g in daily_df.groupby("ticker", sort=False):
        g = g.sort_values("date").dropna(subset=[price_col])
        dates = pd.DatetimeIndex(g["date"].to_numpy())
        prices = g[price_col].reset_index(drop=True)
        maps[ticker] = (dates, prices)
    return maps


def closest_trading_quote(
    dates: pd.DatetimeIndex,
    prices: pd.Series,
    target: pd.Timestamp,
) -> tuple[pd.Timestamp, float] | tuple[pd.NaT, float]:
    """Trading day (and price) closest in absolute time to target."""
    if len(dates) == 0:
        return pd.NaT, float("nan")

    target = pd.Timestamp(target)
    i = dates.searchsorted(target)
    candidates: list[int] = []
    if i < len(dates):
        candidates.append(int(i))
    if i > 0:
        candidates.append(int(i - 1))
    best = min(candidates, key=lambda j: abs(dates[j] - target))
    return pd.Timestamp(dates[best]), float(prices.iloc[best])


def next_trading_day_quote(
    dates: pd.DatetimeIndex,
    prices: pd.Series,
    after_date: pd.Timestamp,
) -> tuple[pd.Timestamp, float] | tuple[pd.NaT, float]:
    """First trading day strictly after after_date, with its price."""
    if len(dates) == 0:
        return pd.NaT, float("nan")
    after_date = pd.Timestamp(after_date)
    i = int(dates.searchsorted(after_date, side="right"))
    if i >= len(dates):
        return pd.NaT, float("nan")
    return pd.Timestamp(dates[i]), float(prices.iloc[i])


def same_day_quote(
    dates: pd.DatetimeIndex,
    prices: pd.Series,
    target: pd.Timestamp,
) -> tuple[pd.Timestamp, float] | tuple[pd.NaT, float]:
    """Exact trading-day quote on target, if that date exists."""
    if len(dates) == 0:
        return pd.NaT, float("nan")
    target = pd.Timestamp(target)
    i = dates.searchsorted(target)
    if i < len(dates) and dates[i] == target:
        return pd.Timestamp(dates[i]), float(prices.iloc[i])
    return pd.NaT, float("nan")


# =============================================================================
# SIMULATION
# =============================================================================

@dataclass
class PortfolioResult:
    name: str
    transactions: pd.DataFrame
    cash: float
    holdings_value: float
    skipped_buys: int
    # Equal-weighted closed-trade stats (comparable to experiment price_change).
    n_closed: int = 0
    mean_price_change: float = float("nan")
    median_price_change: float = float("nan")
    pct_rebound_gt_1: float = float("nan")
    capital_deployed: float = 0.0

    @property
    def total_equity(self) -> float:
        return self.cash + self.holdings_value

    @property
    def pnl(self) -> float:
        return self.total_equity - START_CASH


def simulate_strategy(
    name: str,
    trades: pd.DataFrame,
    start_cash: float = START_CASH,
    invest_amount: float = INVEST_AMOUNT,
    last_prices: dict[str, float] | None = None,
) -> PortfolioResult:
    """
    Execute buy/sell rows chronologically.

    `trades` columns:
      buy_date, buy_ticker, buy_price,
      sell_date, sell_price
    (sell_* may be NaN if no exit quote — position stays open).
    """
    cash = float(start_cash)
    # open lots: list of dicts with ticker, shares, buy_date, buy_price, sell_date, sell_price
    open_lots: list[dict] = []
    records: list[dict] = []
    skipped_buys = 0

    # Build a timeline of actions. Sells scheduled for a date must run before
    # buys on the same date so capital recycles within the day.
    actions: list[tuple[pd.Timestamp, int, int, dict]] = []
    # sort key: (date, kind_order, row_i, payload)  kind: 0=sell, 1=buy
    for i, row in trades.iterrows():
        buy_date = pd.Timestamp(row["buy_date"])
        actions.append((buy_date, 1, int(i), row))
        if pd.notna(row["sell_date"]) and pd.notna(row["sell_price"]):
            actions.append((pd.Timestamp(row["sell_date"]), 0, int(i), row))

    actions.sort(key=lambda a: (a[0], a[1], a[2]))

    # Track which lot index corresponds to each trade row for sells.
    lot_by_trade: dict[int, int] = {}

    for action_date, kind, trade_i, row in actions:
        if kind == 1:  # buy
            if cash < invest_amount:
                skipped_buys += 1
                records.append(
                    {
                        "action": "skip_buy",
                        "date": action_date,
                        "ticker": row["buy_ticker"],
                        "price": row["buy_price"],
                        "shares": 0.0,
                        "cash_delta": 0.0,
                        "cash_after": cash,
                        "reason": "insufficient_cash",
                    }
                )
                continue

            buy_price = float(row["buy_price"])
            shares = invest_amount / buy_price
            cash -= invest_amount
            lot_by_trade[trade_i] = len(open_lots)
            open_lots.append(
                {
                    "trade_i": trade_i,
                    "ticker": row["buy_ticker"],
                    "shares": shares,
                    "buy_date": action_date,
                    "buy_price": buy_price,
                    "sell_date": row["sell_date"],
                    "sell_price": row["sell_price"],
                    "open": True,
                }
            )
            records.append(
                {
                    "action": "buy",
                    "date": action_date,
                    "ticker": row["buy_ticker"],
                    "price": buy_price,
                    "shares": shares,
                    "cash_delta": -invest_amount,
                    "cash_after": cash,
                    "reason": "",
                }
            )
        else:  # sell
            lot_idx = lot_by_trade.get(trade_i)
            if lot_idx is None:
                continue  # buy was skipped
            lot = open_lots[lot_idx]
            if not lot["open"]:
                continue
            sell_price = float(row["sell_price"])
            proceeds = lot["shares"] * sell_price
            cash += proceeds
            lot["open"] = False
            records.append(
                {
                    "action": "sell",
                    "date": action_date,
                    "ticker": lot["ticker"],
                    "price": sell_price,
                    "shares": lot["shares"],
                    "cash_delta": proceeds,
                    "cash_after": cash,
                    "reason": "",
                }
            )

    # Mark-to-market any lots that never got a sell quote.
    holdings_value = 0.0
    last_prices = last_prices or {}
    for lot in open_lots:
        if not lot["open"]:
            continue
        mtm = last_prices.get(lot["ticker"])
        if mtm is None or pd.isna(mtm):
            records.append(
                {
                    "action": "mark_open",
                    "date": pd.NaT,
                    "ticker": lot["ticker"],
                    "price": float("nan"),
                    "shares": lot["shares"],
                    "cash_delta": 0.0,
                    "cash_after": cash,
                    "reason": "still_open_no_quote",
                }
            )
            continue
        holdings_value += lot["shares"] * float(mtm)
        records.append(
            {
                "action": "mark_open",
                "date": pd.NaT,
                "ticker": lot["ticker"],
                "price": float(mtm),
                "shares": lot["shares"],
                "cash_delta": 0.0,
                "cash_after": cash,
                "reason": "still_open_mtm",
            }
        )

    txns = pd.DataFrame.from_records(records)

    # Return stats: closed sells when available; otherwise MTM open lots at last_prices
    # (e.g. buys near the end of the sample with no exit quote yet).
    price_changes: list[float] = []
    executed_idx = set(lot_by_trade.keys())
    closed = trades.loc[
        trades.index.isin(executed_idx)
        & trades["sell_price"].notna()
        & trades["buy_price"].notna()
        & (trades["buy_price"] > 0)
    ]
    if not closed.empty:
        price_changes.extend((closed["sell_price"] / closed["buy_price"]).astype(float).tolist())

    for lot in open_lots:
        if not lot["open"] or lot["buy_price"] <= 0:
            continue
        mtm = last_prices.get(lot["ticker"])
        if mtm is None or pd.isna(mtm):
            continue
        price_changes.append(float(mtm) / float(lot["buy_price"]))

    if price_changes:
        pc = pd.Series(price_changes, dtype=float)
        n_closed = len(pc)
        mean_pc = float(pc.mean())
        median_pc = float(pc.median())
        pct_up = float((pc > 1).mean() * 100.0)
    else:
        n_closed, mean_pc, median_pc, pct_up = 0, float("nan"), float("nan"), float("nan")

    n_buys = int((txns["action"] == "buy").sum()) if not txns.empty else 0
    return PortfolioResult(
        name=name,
        transactions=txns,
        cash=cash,
        holdings_value=holdings_value,
        skipped_buys=skipped_buys,
        n_closed=n_closed,
        mean_price_change=mean_pc,
        median_price_change=median_pc,
        pct_rebound_gt_1=pct_up,
        capital_deployed=n_buys * invest_amount,
    )


def build_bargain_trades(
    bargains: pd.DataFrame,
    close_maps: dict[str, tuple[pd.DatetimeIndex, pd.Series]],
    open_maps: dict[str, tuple[pd.DatetimeIndex, pd.Series]],
) -> pd.DataFrame:
    """
    Signal at bargain close date; buy next trading day's open; sell closest
    trading day ~HOLD_MONTHS after the buy date (at close).
    """
    rows: list[dict] = []
    for row in bargains.itertuples(index=False):
        ticker = row.ticker
        signal_date = pd.Timestamp(row.date)

        open_dp = open_maps.get(ticker)
        close_dp = close_maps.get(ticker)
        if open_dp is None:
            buy_date, buy_price = pd.NaT, float("nan")
        else:
            buy_date, buy_price = next_trading_day_quote(
                open_dp[0], open_dp[1], signal_date
            )

        if pd.isna(buy_date) or pd.isna(buy_price) or close_dp is None:
            sell_date, sell_price = pd.NaT, float("nan")
            target = pd.NaT
        else:
            target = buy_date + pd.DateOffset(months=HOLD_MONTHS)
            sell_date, sell_price = closest_trading_quote(
                close_dp[0], close_dp[1], target
            )
            if pd.notna(sell_date) and sell_date < buy_date:
                sell_date, sell_price = pd.NaT, float("nan")

        rows.append(
            {
                "signal_ticker": ticker,
                "signal_date": signal_date,
                "buy_ticker": ticker,
                "buy_date": buy_date,
                "buy_price": buy_price,
                "target_sell_date": target,
                "sell_date": sell_date,
                "sell_price": sell_price,
            }
        )
    return pd.DataFrame(rows)


def build_strategy1_trades(
    bargains: pd.DataFrame,
    close_maps: dict[str, tuple[pd.DatetimeIndex, pd.Series]],
    open_maps: dict[str, tuple[pd.DatetimeIndex, pd.Series]],
) -> pd.DataFrame:
    """Alias kept for clarity at call sites."""
    return build_bargain_trades(bargains, close_maps, open_maps)


def build_strategy2_trades(
    s1_trades: pd.DataFrame,
    index_dates: pd.DatetimeIndex,
    index_opens: pd.Series,
    index_closes: pd.Series,
    index_ticker: str = INDEX_TICKER,
) -> pd.DataFrame:
    """
    One index trade per Strategy-1 row: buy DIA at the open on Strategy-1's
    buy_date; sell closest close ~HOLD_MONTHS later.
    """
    rows: list[dict] = []
    for row in s1_trades.itertuples(index=False):
        signal_ticker = row.signal_ticker
        signal_date = pd.Timestamp(row.signal_date) if pd.notna(row.signal_date) else pd.NaT
        s1_buy_date = pd.Timestamp(row.buy_date) if pd.notna(row.buy_date) else pd.NaT

        if pd.isna(s1_buy_date):
            buy_date, buy_price = pd.NaT, float("nan")
            sell_date, sell_price = pd.NaT, float("nan")
            target = pd.NaT
        else:
            buy_date, buy_price = same_day_quote(
                index_dates, index_opens, s1_buy_date
            )
            # If DIA has no session that day, take next DIA open.
            if pd.isna(buy_date):
                buy_date, buy_price = next_trading_day_quote(
                    index_dates, index_opens, s1_buy_date - pd.Timedelta(days=1)
                )
            # Reject if still missing or unreasonably far from Strategy-1 buy date.
            if (
                pd.isna(buy_date)
                or pd.isna(buy_price)
                or abs(buy_date - s1_buy_date) > pd.Timedelta(days=7)
            ):
                buy_date, buy_price = pd.NaT, float("nan")
                sell_date, sell_price = pd.NaT, float("nan")
                target = pd.NaT
            else:
                target = buy_date + pd.DateOffset(months=HOLD_MONTHS)
                sell_date, sell_price = closest_trading_quote(
                    index_dates, index_closes, target
                )
                if pd.notna(sell_date) and sell_date < buy_date:
                    sell_date, sell_price = pd.NaT, float("nan")

        rows.append(
            {
                "signal_ticker": signal_ticker,
                "signal_date": signal_date,
                "buy_ticker": index_ticker,
                "buy_date": buy_date,
                "buy_price": buy_price,
                "target_sell_date": target,
                "sell_date": sell_date,
                "sell_price": sell_price,
            }
        )
    return pd.DataFrame(rows)


def paired_tradable_mask(s1_trades: pd.DataFrame, s2_trades: pd.DataFrame) -> pd.Series:
    """True where both strategies can execute the buy for the same signal."""
    if len(s1_trades) != len(s2_trades):
        raise ValueError(
            f"Strategy trade tables must be 1:1 aligned "
            f"(got {len(s1_trades)} vs {len(s2_trades)})"
        )
    s1_ok = s1_trades["buy_price"].notna() & s1_trades["buy_date"].notna()
    s2_ok = s2_trades["buy_price"].notna() & s2_trades["buy_date"].notna()
    return s1_ok & s2_ok


def summary_row(result: PortfolioResult) -> dict:
    return {
        "strategy": result.name,
        "n_buys": int((result.transactions["action"] == "buy").sum()),
        "n_sells": int((result.transactions["action"] == "sell").sum()),
        "n_closed": result.n_closed,
        "skipped_buys": result.skipped_buys,
        "median_price_change": result.median_price_change,
        "mean_price_change": result.mean_price_change,
        "pct_rebound_gt_1": result.pct_rebound_gt_1,
        "capital_deployed": result.capital_deployed,
        "cash": result.cash,
        "holdings_value": result.holdings_value,
        "total_equity": result.total_equity,
        "pnl_vs_start": result.pnl,
        "pnl_per_dollar_deployed": (
            result.pnl / result.capital_deployed
            if result.capital_deployed > 0
            else float("nan")
        ),
    }


# =============================================================================
# YEARLY ANALYSIS
# =============================================================================

def _closed_returns_by_buy_year(txns: pd.DataFrame) -> pd.DataFrame:
    """FIFO-match buys to sells; return median/mean price_change by buy year."""
    pending: dict[str, deque] = defaultdict(deque)
    closed_rows: list[dict] = []
    for row in txns.itertuples(index=False):
        if row.action == "buy":
            pending[row.ticker].append(row)
        elif row.action == "sell":
            if not pending[row.ticker]:
                continue
            buy = pending[row.ticker].popleft()
            if buy.price and buy.price > 0:
                closed_rows.append(
                    {
                        "year": int(buy.date.year),
                        "price_change": float(row.price) / float(buy.price),
                    }
                )

    closed = pd.DataFrame(closed_rows)
    if closed.empty:
        out = pd.DataFrame(columns=["median_return", "mean_return"])
        out.index.name = "year"
        return out
    return (
        closed.groupby("year")["price_change"]
        .agg(median_return="median", mean_return="mean")
    )


def _plot_yearly_returns(
    stats: pd.DataFrame,
    plot_file: Path = YEARLY_RETURNS_PLOT_FILE,
) -> Path:
    """Plot Strategy 1 vs 2 median/mean returns by buy year."""
    years = stats["year"].astype(int)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        years,
        stats["s1_median_return"],
        color="#1f4e79",
        marker="o",
        markersize=4,
        linewidth=1.5,
        label="S1 median (S&P 500 PIT)",
    )
    ax.plot(
        years,
        stats["s1_mean_return"],
        color="#5dade2",
        marker="s",
        markersize=4,
        linewidth=1.5,
        linestyle="--",
        label="S1 mean (S&P 500 PIT)",
    )
    ax.plot(
        years,
        stats["s2_median_return"],
        color="#c0392b",
        marker="o",
        markersize=4,
        linewidth=1.5,
        label=f"S2 median ({INDEX_TICKER})",
    )
    ax.plot(
        years,
        stats["s2_mean_return"],
        color="#e67e22",
        marker="s",
        markersize=4,
        linewidth=1.5,
        linestyle="--",
        label=f"S2 mean ({INDEX_TICKER})",
    )
    ax.axhline(1.0, color="#888888", linestyle=":", linewidth=1.0, label="breakeven (1.0)")

    ax.set_title(
        f"Strategy 1 vs 2: per-trade return by buy year "
        f"(hold ≈ {HOLD_MONTHS} months)"
    )
    ax.set_xlabel("Buy year")
    ax.set_ylabel("price_change (sell / buy)")
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_file, dpi=150)
    print(f"Saved yearly returns plot to {plot_file}")
    plt.show()
    return plot_file


def analyze_year_stats(
    s1_txn_file: Path = STRATEGY1_TXN_FILE,
    s2_txn_file: Path = STRATEGY2_TXN_FILE,
) -> pd.DataFrame:
    """
    Group Strategy-1 transactions by buy year, with Strategy-2 returns alongside.

    For each year report: Strategy-1 buy count, distinct companies, and
    median/mean return (sell_price / buy_price) for both Strategy 1 (S&P 500
    point-in-time bargains) and Strategy 2 (DIA on the same signals). Buy year
    attributes entries to market regimes (e.g. 2008–2010, 2020–2022, 2024–2026).
    Also plots the four return curves over years.
    """
    for path in (s1_txn_file, s2_txn_file):
        if not path.exists():
            raise FileNotFoundError(
                f"Transaction file not found: {path}. Run main() first."
            )

    s1 = pd.read_csv(s1_txn_file)
    s1["date"] = pd.to_datetime(s1["date"])
    s2 = pd.read_csv(s2_txn_file)
    s2["date"] = pd.to_datetime(s2["date"])

    buys = s1.loc[s1["action"] == "buy"].copy()
    buys["year"] = buys["date"].dt.year.astype(int)

    s1_ret = _closed_returns_by_buy_year(s1).rename(
        columns={
            "median_return": "s1_median_return",
            "mean_return": "s1_mean_return",
        }
    )
    s2_ret = _closed_returns_by_buy_year(s2).rename(
        columns={
            "median_return": "s2_median_return",
            "mean_return": "s2_mean_return",
        }
    )

    stats = (
        buys.groupby("year")
        .agg(n_transactions=("ticker", "size"), n_companies=("ticker", "nunique"))
        .join(s1_ret, how="left")
        .join(s2_ret, how="left")
        .sort_index()
        .reset_index()
    )

    print("\n" + "=" * 88)
    print("Strategy 1 vs 2 yearly stats (grouped by buy year)")
    print(f"  S1: {s1_txn_file.name}")
    print(f"  S2: {s2_txn_file.name}")
    print("  return = sell_price / buy_price")
    print("=" * 88)
    print(
        f"{'year':>6}  {'n_tx':>6}  {'n_cos':>6}  "
        f"{'s1_med':>8}  {'s1_mean':>8}  "
        f"{'s2_med':>8}  {'s2_mean':>8}"
    )

    def _fmt(val: float) -> str:
        return f"{val:8.3f}" if pd.notna(val) else f"{'n/a':>8}"

    for row in stats.itertuples(index=False):
        print(
            f"{int(row.year):6d}  {int(row.n_transactions):6d}  "
            f"{int(row.n_companies):6d}  "
            f"{_fmt(row.s1_median_return)}  {_fmt(row.s1_mean_return)}  "
            f"{_fmt(row.s2_median_return)}  {_fmt(row.s2_mean_return)}"
        )
    print("=" * 88)

    if not stats.empty:
        _plot_yearly_returns(stats)

    return stats


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 72)
    print("Benchmark invest: S&P 500 (PIT) bargains vs DIA (signal-paired)")
    print("=" * 72)
    print(f"  start_cash      = ${START_CASH:,.0f}")
    print(f"  invest_amount   = ${INVEST_AMOUNT:,.0f} per signal")
    print(f"  horizon_start   = {HORIZON_START.date()}")
    print(f"  hold            = {HOLD_MONTHS} months (closest trading day)")
    print(f"  index_proxy     = {INDEX_TICKER} (Stooq has no ^DJI in US daily zip)")
    print("=" * 72)

    daily_df = load_or_build_daily_prices()
    events_since = _read_trend_down_events(daily_df)
    bargains_sp500 = load_bargain_events(
        daily_df, sp500_only=True, events=events_since
    )
    if bargains_sp500.empty:
        print("No S&P 500 (point-in-time) bargain events to trade.")
        return

    if "open_price" not in daily_df.columns:
        raise RuntimeError(
            "daily_df is missing open_price. Rebuild the daily_prices cache "
            "(delete derived_data/daily_prices.pkl and re-run)."
        )

    close_maps = _price_maps(daily_df, price_col="close_price")
    open_maps = _price_maps(daily_df, price_col="open_price")
    index_df = load_index_prices()
    if "open_price" not in index_df.columns:
        raise RuntimeError(f"{INDEX_TICKER} data is missing open_price.")
    index_dates = pd.DatetimeIndex(index_df["date"].to_numpy())
    index_opens = index_df["open_price"].reset_index(drop=True)
    index_closes = index_df["close_price"].reset_index(drop=True)
    index_start = index_dates.min()

    # Strategies 1 & 2: 1:1 paired S&P 500 signals where both ticker and DIA trade.
    # S1 buys next-day open after the signal close; S2 buys DIA open on that same buy date.
    s1_all = build_bargain_trades(bargains_sp500, close_maps, open_maps)
    s2_all = build_strategy2_trades(
        s1_all, index_dates, index_opens, index_closes
    )
    pair_ok = paired_tradable_mask(s1_all, s2_all)
    n_dropped = int((~pair_ok).sum())
    s1_trades = s1_all.loc[pair_ok].reset_index(drop=True)
    s2_trades = s2_all.loc[pair_ok].reset_index(drop=True)

    if len(s1_trades) != len(s2_trades):
        raise RuntimeError("Paired strategies diverged after filtering.")

    print(
        f"\nPaired S&P 500 signals (1 bargain buy ↔ 1 {INDEX_TICKER} buy): "
        f"{len(s1_trades):,}"
    )
    if n_dropped:
        print(
            f"  Dropped {n_dropped:,} S&P 500 signals lacking a next-day open "
            f"in either the bargain ticker or {INDEX_TICKER} "
            f"(DIA history starts {index_start.date()})"
        )

    last_px = {
        t: float(prices.iloc[-1])
        for t, (_dates, prices) in close_maps.items()
        if len(prices)
    }
    if len(index_closes):
        last_px[INDEX_TICKER] = float(index_closes.iloc[-1])

    s1 = simulate_strategy("strategy1_bargain_sp500", s1_trades, last_prices=last_px)
    s2 = simulate_strategy("strategy2_index", s2_trades, last_prices=last_px)

    TXN_DIR.mkdir(parents=True, exist_ok=True)
    s1.transactions.to_csv(STRATEGY1_TXN_FILE, index=False)
    s2.transactions.to_csv(STRATEGY2_TXN_FILE, index=False)

    summary = pd.DataFrame([summary_row(s1), summary_row(s2)])
    summary.to_csv(SUMMARY_FILE, index=False)

    print("\n" + "=" * 72)
    print("Results")
    print("=" * 72)
    print(
        "Note: absolute P&L scales with # of bets; per-trade price_change is "
        "the fair quality comparison (same idea as experiment_trend_down_params)."
    )
    for _, row in summary.iterrows():
        print(f"\n{row['strategy']}:")
        print(f"  buys / sells / closed = {row['n_buys']:,} / {row['n_sells']:,} / {row['n_closed']:,}")
        print(f"  skipped buys          = {row['skipped_buys']:,}")
        print(
            f"  per-trade price_change: "
            f"median={row['median_price_change']:.3f}, "
            f"mean={row['mean_price_change']:.3f}, "
            f"pct_rebound(>1)={row['pct_rebound_gt_1']:.1f}%"
        )
        print(f"  capital deployed      = ${row['capital_deployed']:,.2f}")
        print(f"  cash at hand          = ${row['cash']:,.2f}")
        print(f"  in stock market       = ${row['holdings_value']:,.2f}")
        print(f"  total equity          = ${row['total_equity']:,.2f}")
        print(f"  P&L vs start          = ${row['pnl_vs_start']:,.2f}")
        print(f"  P&L / $ deployed      = {row['pnl_per_dollar_deployed']:.4f}")

    print(f"\nSaved transactions:")
    print(f"  {STRATEGY1_TXN_FILE}")
    print(f"  {STRATEGY2_TXN_FILE}")
    print(f"  {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
    analyze_year_stats()
