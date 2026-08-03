#!/usr/bin/env python3
"""
Simulate investment strategies since 2000 and compare end wealth.

Strategy 1 — Fortune 500 bargain stocks:
  Use trend_down_stocks events, limited to Fortune 500. Buy $1 of each
  bargain at the recorded date/price; sell at the trading day closest to
  6 months later.

Strategy 2 — index benchmark:
  On each Strategy-1 bargain date, buy $1 of the Dow proxy (DIA ETF)
  instead; sell at the trading day closest to 6 months later.
  Strategies 1 and 2 are paired 1:1 on the same signals (only signals where
  both the bargain ticker and DIA are tradable are kept).

Strategy 3 — all bargain stocks (no Fortune 500 filter):
  Same rules as Strategy 1, but over every trend-down signal (still limited
  to the DIA-available date window so the horizon matches Strategies 1–2).

Assumptions: fractional shares allowed, no taxes/fees, start cash $1,000,000.

Note: Stooq's US daily package does not include the ^DJI index series.
DJIA.US (ETF) only starts in 2022, so DIA.US is used as the Dow proxy
(available from 2005-02-25 in this dataset).

Usage:
  python benchmark_invest.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import config
from experiment_trend_down_params import (
    load_fortune500_tickers,
    load_or_build_daily_prices,
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
STRATEGY1_TXN_FILE = TXN_DIR / "strategy1_bargain_f500_transactions.csv"
STRATEGY2_TXN_FILE = TXN_DIR / "strategy2_index_transactions.csv"
STRATEGY3_TXN_FILE = TXN_DIR / "strategy3_bargain_all_transactions.csv"
SUMMARY_FILE = TXN_DIR / "benchmark_summary.csv"


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
    fortune500_only: bool = True,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Trend-down events on/after HORIZON_START.
    Optionally restrict to Fortune 500 tickers.
    Pass `events` to avoid reloading when computing multiple universes.
    """
    if events is None:
        events = _read_trend_down_events(daily_df)
    else:
        events = events.copy()

    n_since = len(events)
    if fortune500_only:
        price_tickers = set(daily_df["ticker"].unique())
        f500 = load_fortune500_tickers(price_tickers=price_tickers)
        events = events[events["ticker"].isin(f500)].copy()
        print(
            f"Fortune 500 bargains since {HORIZON_START.date()}: "
            f"{len(events):,} / {n_since:,} events ({len(f500):,} F500 tickers)"
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
    """Load daily closes for the Dow proxy ETF (DIA)."""
    etfs = process_stock_directory(INDEX_ETF_PATTERN, base_dir=base_dir, verbose=False)
    if etfs.empty:
        raise FileNotFoundError(f"No ETF data under {base_dir / INDEX_ETF_PATTERN}")

    etfs["ticker"] = etfs["ticker"].str.upper().str.replace(".US", "", regex=False)
    idx = (
        etfs.loc[etfs["ticker"] == ticker, ["ticker", "date", "close_price"]]
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


def _price_maps(daily_df: pd.DataFrame) -> dict[str, tuple[pd.DatetimeIndex, pd.Series]]:
    """ticker -> (sorted dates, close_price aligned to those dates)."""
    maps: dict[str, tuple[pd.DatetimeIndex, pd.Series]] = {}
    for ticker, g in daily_df.groupby("ticker", sort=False):
        g = g.sort_values("date")
        dates = pd.DatetimeIndex(g["date"].to_numpy())
        prices = g["close_price"].reset_index(drop=True)
        maps[ticker] = (dates, prices)
    return maps


def closest_trading_quote(
    dates: pd.DatetimeIndex,
    prices: pd.Series,
    target: pd.Timestamp,
) -> tuple[pd.Timestamp, float] | tuple[pd.NaT, float]:
    """Trading day (and close) closest in absolute time to target."""
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

    # Closed-trade quality stats from the input trade table (sell/buy),
    # restricted to buys that actually executed.
    executed_idx = set(lot_by_trade.keys())
    closed = trades.loc[
        trades.index.isin(executed_idx)
        & trades["sell_price"].notna()
        & trades["buy_price"].notna()
        & (trades["buy_price"] > 0)
    ].copy()
    if not closed.empty:
        closed["price_change"] = closed["sell_price"] / closed["buy_price"]
        n_closed = len(closed)
        mean_pc = float(closed["price_change"].mean())
        median_pc = float(closed["price_change"].median())
        pct_up = float((closed["price_change"] > 1).mean() * 100.0)
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
    price_maps: dict[str, tuple[pd.DatetimeIndex, pd.Series]],
) -> pd.DataFrame:
    """Buy at recorded bargain date/price; sell nearest trading day ~HOLD_MONTHS out."""
    rows: list[dict] = []
    for row in bargains.itertuples(index=False):
        ticker = row.ticker
        buy_date = pd.Timestamp(row.date)
        buy_price = float(row.price)
        target = buy_date + pd.DateOffset(months=HOLD_MONTHS)

        dates_prices = price_maps.get(ticker)
        if dates_prices is None:
            sell_date, sell_price = pd.NaT, float("nan")
        else:
            sell_date, sell_price = closest_trading_quote(
                dates_prices[0], dates_prices[1], target
            )
            if pd.notna(sell_date) and sell_date < buy_date:
                sell_date, sell_price = pd.NaT, float("nan")

        rows.append(
            {
                "signal_ticker": ticker,
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
    price_maps: dict[str, tuple[pd.DatetimeIndex, pd.Series]],
) -> pd.DataFrame:
    """Alias kept for clarity at call sites."""
    return build_bargain_trades(bargains, price_maps)


def build_strategy2_trades(
    bargains: pd.DataFrame,
    index_dates: pd.DatetimeIndex,
    index_prices: pd.Series,
    index_ticker: str = INDEX_TICKER,
) -> pd.DataFrame:
    """One index trade per bargain row (same order/length as Strategy 1 inputs)."""
    rows: list[dict] = []
    for row in bargains.itertuples(index=False):
        signal_ticker = row.ticker
        signal_date = pd.Timestamp(row.date)
        target = signal_date + pd.DateOffset(months=HOLD_MONTHS)

        buy_date, buy_price = closest_trading_quote(
            index_dates, index_prices, signal_date
        )
        sell_date, sell_price = closest_trading_quote(
            index_dates, index_prices, target
        )

        # Reject if the closest buy quote is unreasonably far (e.g. pre-DIA history).
        if pd.isna(buy_date) or abs(buy_date - signal_date) > pd.Timedelta(days=7):
            buy_date, buy_price = pd.NaT, float("nan")
            sell_date, sell_price = pd.NaT, float("nan")
        elif pd.notna(sell_date) and pd.notna(buy_date) and sell_date < buy_date:
            sell_date, sell_price = pd.NaT, float("nan")

        rows.append(
            {
                "signal_ticker": signal_ticker,
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


def filter_tradable(trades: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with a valid buy quote (sell may still be missing)."""
    return trades[trades["buy_price"].notna() & trades["buy_date"].notna()].copy()


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
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 72)
    print("Benchmark invest: F500 bargains vs all bargains vs Dow proxy (DIA)")
    print("=" * 72)
    print(f"  start_cash      = ${START_CASH:,.0f}")
    print(f"  invest_amount   = ${INVEST_AMOUNT:,.0f} per signal")
    print(f"  horizon_start   = {HORIZON_START.date()}")
    print(f"  hold            = {HOLD_MONTHS} months (closest trading day)")
    print(f"  index_proxy     = {INDEX_TICKER} (Stooq has no ^DJI in US daily zip)")
    print("=" * 72)

    daily_df = load_or_build_daily_prices()
    events_since = _read_trend_down_events(daily_df)
    bargains_f500 = load_bargain_events(
        daily_df, fortune500_only=True, events=events_since
    )
    bargains_all = load_bargain_events(
        daily_df, fortune500_only=False, events=events_since
    )
    if bargains_f500.empty:
        print("No Fortune 500 bargain events to trade.")
        return

    price_maps = _price_maps(daily_df)
    index_df = load_index_prices()
    index_dates = pd.DatetimeIndex(index_df["date"].to_numpy())
    index_prices = index_df["close_price"].reset_index(drop=True)
    index_start = index_dates.min()

    # Strategies 1 & 2: 1:1 paired F500 signals where both ticker and DIA trade.
    s1_all = build_bargain_trades(bargains_f500, price_maps)
    s2_all = build_strategy2_trades(bargains_f500, index_dates, index_prices)
    pair_ok = paired_tradable_mask(s1_all, s2_all)
    n_dropped = int((~pair_ok).sum())
    s1_trades = s1_all.loc[pair_ok].reset_index(drop=True)
    s2_trades = s2_all.loc[pair_ok].reset_index(drop=True)

    if len(s1_trades) != len(s2_trades):
        raise RuntimeError("Paired strategies diverged after filtering.")

    print(
        f"\nPaired F500 signals (1 bargain buy ↔ 1 {INDEX_TICKER} buy): "
        f"{len(s1_trades):,}"
    )
    if n_dropped:
        print(
            f"  Dropped {n_dropped:,} F500 signals lacking a same-day-ish quote in "
            f"either the bargain ticker or {INDEX_TICKER} "
            f"(DIA history starts {index_start.date()})"
        )

    # Strategy 3: all tickers, same DIA-available date window as Strategies 1–2.
    bargains_all_window = bargains_all[bargains_all["date"] >= index_start].copy()
    s3_trades = filter_tradable(build_bargain_trades(bargains_all_window, price_maps))
    print(
        f"Strategy 3 all-ticker bargains in DIA window: {len(s3_trades):,} "
        f"(from {len(bargains_all_window):,} signals on/after {index_start.date()})"
    )

    last_px = {
        t: float(prices.iloc[-1])
        for t, (_dates, prices) in price_maps.items()
        if len(prices)
    }
    if len(index_prices):
        last_px[INDEX_TICKER] = float(index_prices.iloc[-1])

    s1 = simulate_strategy("strategy1_bargain_f500", s1_trades, last_prices=last_px)
    s2 = simulate_strategy("strategy2_index", s2_trades, last_prices=last_px)
    s3 = simulate_strategy("strategy3_bargain_all", s3_trades, last_prices=last_px)

    TXN_DIR.mkdir(parents=True, exist_ok=True)
    s1.transactions.to_csv(STRATEGY1_TXN_FILE, index=False)
    s2.transactions.to_csv(STRATEGY2_TXN_FILE, index=False)
    s3.transactions.to_csv(STRATEGY3_TXN_FILE, index=False)

    summary = pd.DataFrame([summary_row(s1), summary_row(s2), summary_row(s3)])
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
    print(f"  {STRATEGY3_TXN_FILE}")
    print(f"  {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
