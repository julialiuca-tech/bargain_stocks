#!/usr/bin/env python3
"""
Step-by-step sanity checks for bargain event definitions.
"""

from __future__ import annotations

import sys
from collections import defaultdict, deque
from types import ModuleType

# benchmark_invest / experiment_trend_down_params import matplotlib at module
# level; stub it so this debug script stays light and interactive.
if "matplotlib" not in sys.modules:
    _mpl = ModuleType("matplotlib")
    _mpl.use = lambda *a, **k: None
    sys.modules["matplotlib"] = _mpl
    sys.modules["matplotlib.pyplot"] = ModuleType("matplotlib.pyplot")

import pandas as pd

import config
from benchmark_invest import (
    HOLD_MONTHS,
    STRATEGY3_TXN_FILE,
    _load_prepared_sec_features,
    _read_trend_down_events,
    load_bargain_events,
)
from experiment_trend_down_params import load_or_build_daily_prices


def _normalize_ticker(ticker: str) -> str:
    return str(ticker).upper().replace(".US", "")


def verify_bargain_event_single(
    ticker: str,
    date,
    daily_df: pd.DataFrame,
    *,
    thresh: float = config.trend_down_thresh,
    history_days: int = config.trend_down_history,
    prices_by_ticker: dict[str, pd.DataFrame] | None = None,
) -> bool:
    """
    Return True iff (ticker, date) is a bargain moment under the config rule:

      current_close <= historic_avg * (1 - thresh)

    historic_avg = mean of daily closes in the prior ``history_days`` calendar
    days, excluding the evaluation day itself. Operationally: for each trading
    session d in (event_date - history_days, event_date], take the previous
    session's close, then average those values.

    Uses only config.py parameters plus raw daily closes — no detection helpers.
    Pass ``prices_by_ticker`` to avoid rescanning the full daily frame.
    """
    event_date = pd.Timestamp(date).normalize()
    ticker = _normalize_ticker(ticker)

    if prices_by_ticker is not None:
        prices = prices_by_ticker.get(ticker)
        if prices is None or prices.empty:
            return False
    else:
        prices = daily_df.loc[
            daily_df["ticker"].map(_normalize_ticker) == ticker,
            ["date", "close_price"],
        ].copy()
        if prices.empty:
            return False
        prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
        prices = prices.sort_values("date")

    closes = prices.drop_duplicates(subset=["date"], keep="last").set_index("date")[
        "close_price"
    ]
    closes = closes.sort_index()
    if event_date not in closes.index:
        return False

    current = float(closes.loc[event_date])
    if not pd.notna(current) or current <= 0:
        return False

    # Previous-session close labeled on each trading day, then calendar window.
    prior_close_on_day = closes.shift(1)
    window_start = event_date - pd.Timedelta(days=int(history_days))
    hist = prior_close_on_day.loc[
        (prior_close_on_day.index > window_start)
        & (prior_close_on_day.index <= event_date)
    ].dropna()
    if hist.empty:
        return False

    historic_avg = float(hist.mean())
    if not pd.notna(historic_avg) or historic_avg <= 0:
        return False

    return current <= historic_avg * (1.0 - float(thresh))


def _prices_by_ticker(
    daily_df: pd.DataFrame, tickers: set[str]
) -> dict[str, pd.DataFrame]:
    """Subset/normalize daily closes for the tickers we will verify."""
    want = {_normalize_ticker(t) for t in tickers}
    sub = daily_df.loc[
        daily_df["ticker"].map(_normalize_ticker).isin(want),
        ["ticker", "date", "close_price"],
    ].copy()
    sub["ticker"] = sub["ticker"].map(_normalize_ticker)
    sub["date"] = pd.to_datetime(sub["date"]).dt.normalize()
    sub = sub.sort_values(["ticker", "date"])
    return {
        t: g[["date", "close_price"]].reset_index(drop=True)
        for t, g in sub.groupby("ticker", sort=False)
    }


def verify_bargain_events_in_batch() -> None:
    # Same load path as benchmark_invest.main() (bargain event universe).
    daily_df = load_or_build_daily_prices()
    events_since = _read_trend_down_events(daily_df)
    bargains_sp500 = load_bargain_events(
        daily_df, sp500_only=True, events=events_since
    )
    if bargains_sp500.empty:
        raise SystemExit("No S&P 500 bargain events to verify.")

    n_sample = min(100, len(bargains_sp500))
    sample = bargains_sp500.sample(n=n_sample).reset_index(drop=True)
    prices = _prices_by_ticker(daily_df, set(sample["ticker"]))

    results = [
        verify_bargain_event_single(
            row.ticker, row.date, daily_df, prices_by_ticker=prices
        )
        for row in sample.itertuples(index=False)
    ]
    n_ok = sum(results)
    pct = 100.0 * n_ok / n_sample

    print(
        f"verify_bargain_event: {n_ok}/{n_sample} sampled S&P 500 bargains "
        f"satisfy definition "
        f"(thresh={config.trend_down_thresh}, "
        f"history={config.trend_down_history}d) → {pct:.1f}%"
    )
    if n_ok != n_sample:
        bad = sample.loc[[i for i, ok in enumerate(results) if not ok]]
        print("Failures:")
        print(
            bad[["ticker", "date", "price", "historical_average"]].to_string(
                index=False
            )
        )
        raise SystemExit(1)


def _closes_by_ticker(
    daily_df: pd.DataFrame, tickers: set[str]
) -> dict[str, pd.Series]:
    """ticker -> close_price Series indexed by normalized trading date."""
    frames = _prices_by_ticker(daily_df, tickers)
    out: dict[str, pd.Series] = {}
    for t, g in frames.items():
        s = (
            g.drop_duplicates(subset=["date"], keep="last")
            .set_index("date")["close_price"]
            .sort_index()
        )
        out[t] = s
    return out


def _closest_trading_day(
    dates: pd.DatetimeIndex, target: pd.Timestamp
) -> pd.Timestamp:
    """Trading date closest in absolute time to target (evaluation-day logic)."""
    if len(dates) == 0:
        return pd.NaT
    target = pd.Timestamp(target).normalize()
    i = int(dates.searchsorted(target))
    candidates: list[int] = []
    if i < len(dates):
        candidates.append(i)
    if i > 0:
        candidates.append(i - 1)
    if not candidates:
        return pd.NaT
    best = min(candidates, key=lambda j: abs(dates[j] - target))
    return pd.Timestamp(dates[best]).normalize()


def _previous_trading_day(
    dates: pd.DatetimeIndex, buy_date: pd.Timestamp
) -> pd.Timestamp:
    """Last session strictly before buy_date (the bargain signal day)."""
    buy_date = pd.Timestamp(buy_date).normalize()
    if len(dates) == 0:
        return pd.NaT
    i = int(dates.searchsorted(buy_date, side="left"))
    if i <= 0:
        return pd.NaT
    prev = pd.Timestamp(dates[i - 1]).normalize()
    return prev if prev < buy_date else pd.NaT


def _pair_sells_fifo(txns: pd.DataFrame) -> dict[int, tuple[pd.Timestamp, float]]:
    """Map buy row index → (sell_date, sell_price) via FIFO per ticker + shares."""
    buys = txns.loc[txns["action"] == "buy"]
    sells = txns.loc[txns["action"] == "sell"].sort_values("date")
    queues: dict[str, deque] = defaultdict(deque)
    for row in sells.itertuples():
        queues[_normalize_ticker(row.ticker)].append(row)

    paired: dict[int, tuple[pd.Timestamp, float]] = {}
    for idx, buy in buys.iterrows():
        ticker = _normalize_ticker(buy["ticker"])
        buy_date = pd.Timestamp(buy["date"]).normalize()
        buy_shares = float(buy["shares"])
        q = queues[ticker]
        held: deque = deque()
        matched = None
        while q:
            sell = q.popleft()
            sell_date = pd.Timestamp(sell.date).normalize()
            if sell_date < buy_date:
                continue
            if abs(float(sell.shares) - buy_shares) <= 1e-9:
                matched = sell
                break
            held.append(sell)
        while held:
            q.appendleft(held.pop())
        if matched is not None:
            paired[idx] = (pd.Timestamp(matched.date).normalize(), float(matched.price))
    return paired


def verify_strategy3_buys(
    n_sample: int = 100,
    hold_months: int = HOLD_MONTHS,
    proba_thresh: float = config.GOOD_BUY_PROBA_THRESH,
) -> None:
    """
    Sample Strategy-3 buys and check:
      1. Paired sell (if any) is the closest trading day to buy + hold_months,
         and the recorded sell price equals that day's close.
      2. The bargain signal (prior session) scores y_pred_proba >= thresh.
    """
    from baseline_model import score_bargain_events

    if not STRATEGY3_TXN_FILE.exists():
        raise SystemExit(
            f"Missing {STRATEGY3_TXN_FILE}. Run python benchmark_invest.py first."
        )

    txns = pd.read_csv(STRATEGY3_TXN_FILE)
    txns["date"] = pd.to_datetime(txns["date"])
    txns["ticker"] = txns["ticker"].map(_normalize_ticker)
    buys = txns.loc[txns["action"] == "buy"].copy()
    if buys.empty:
        raise SystemExit("No Strategy 3 buy events in the transaction file.")

    n_sample = min(n_sample, len(buys))
    sample = buys.sample(n=n_sample).copy()
    sell_map = _pair_sells_fifo(txns)

    daily_df = load_or_build_daily_prices()
    closes = _closes_by_ticker(daily_df, set(sample["ticker"]))

    # Reconstruct bargain signal dates (buy is next session after the flag).
    signal_rows: list[dict] = []
    for idx, buy in sample.iterrows():
        ticker = buy["ticker"]
        buy_date = pd.Timestamp(buy["date"]).normalize()
        series = closes.get(ticker)
        dates = pd.DatetimeIndex(series.index) if series is not None else pd.DatetimeIndex([])
        signal_date = _previous_trading_day(dates, buy_date)
        signal_rows.append(
            {
                "buy_idx": idx,
                "ticker": ticker,
                "date": signal_date,
            }
        )
    signals = pd.DataFrame(signal_rows)

    print(
        f"\nStrategy 3: scoring {len(signals):,} sampled signal dates "
        f"({config.GOOD_BUY_MODEL_PATH.name}, "
        f"keep y_pred_proba >= {proba_thresh:.2f}) ..."
    )
    sec, feats = _load_prepared_sec_features()
    to_score = signals.dropna(subset=["date"]).copy()
    scored = score_bargain_events(
        to_score[["ticker", "date"]],
        config.GOOD_BUY_MODEL_PATH,
        sec=sec,
        feats=feats,
    )
    if not scored.empty:
        scored["ticker"] = scored["ticker"].map(_normalize_ticker)
        scored["date"] = pd.to_datetime(scored["date"]).dt.normalize()
        score_lookup = scored.set_index(["ticker", "date"])["y_pred_proba"]
    else:
        score_lookup = pd.Series(dtype=float)

    failures: list[dict] = []
    scores: list[dict] = []
    n_no_sell = 0
    for rec in signal_rows:
        idx = rec["buy_idx"]
        buy = sample.loc[idx]
        ticker = rec["ticker"]
        buy_date = pd.Timestamp(buy["date"]).normalize()
        signal_date = rec["date"]
        reasons: list[str] = []

        series = closes.get(ticker)
        expected_sell_date = pd.NaT
        expected_sell_price = float("nan")
        if series is not None and len(series):
            dates = pd.DatetimeIndex(series.index)
            target = buy_date + pd.DateOffset(months=int(hold_months))
            cand = _closest_trading_day(dates, target)
            if pd.notna(cand) and cand >= buy_date:
                expected_sell_date = cand
                expected_sell_price = float(series.loc[expected_sell_date])

        actual = sell_map.get(idx)
        if actual is None:
            n_no_sell += 1
            if pd.notna(expected_sell_date):
                reasons.append(
                    f"missing sell; expected {expected_sell_date.date()} "
                    f"@ {expected_sell_price:.6g}"
                )
        else:
            actual_date, actual_price = actual
            if pd.isna(expected_sell_date):
                reasons.append(
                    f"unexpected sell on {actual_date.date()} @ {actual_price:.6g}"
                )
            else:
                if actual_date != expected_sell_date:
                    reasons.append(
                        f"sell date {actual_date.date()} != expected "
                        f"{expected_sell_date.date()} "
                        f"(target { (buy_date + pd.DateOffset(months=int(hold_months))).date() })"
                    )
                if not pd.isna(expected_sell_price) and abs(
                    actual_price - expected_sell_price
                ) > max(1e-6, 1e-6 * abs(expected_sell_price)):
                    reasons.append(
                        f"sell price {actual_price:.6g} != close "
                        f"{expected_sell_price:.6g} on {expected_sell_date.date()}"
                    )

        if pd.isna(signal_date):
            proba = float("nan")
            reasons.append("could not reconstruct bargain signal date")
        else:
            try:
                proba = float(score_lookup.loc[(ticker, signal_date)])
            except KeyError:
                proba = float("nan")
            if pd.isna(proba):
                reasons.append(
                    f"no model score for signal {ticker} {signal_date.date()}"
                )
            elif float(proba) < float(proba_thresh):
                reasons.append(
                    f"y_pred_proba={float(proba):.4f} < thresh {proba_thresh:.2f}"
                )

        scores.append(
            {
                "ticker": ticker,
                "signal_date": (
                    signal_date.date() if pd.notna(signal_date) else None
                ),
                "buy_date": buy_date.date(),
                "y_pred_proba": proba,
            }
        )

        if reasons:
            failures.append(
                {
                    "ticker": ticker,
                    "buy_date": buy_date.date(),
                    "signal_date": (
                        signal_date.date() if pd.notna(signal_date) else None
                    ),
                    "y_pred_proba": proba,
                    "actual_sell_date": (
                        actual[0].date() if actual is not None else None
                    ),
                    "expected_sell_date": (
                        expected_sell_date.date()
                        if pd.notna(expected_sell_date)
                        else None
                    ),
                    "actual_sell_price": actual[1] if actual is not None else None,
                    "expected_sell_price": (
                        expected_sell_price
                        if pd.notna(expected_sell_price)
                        else None
                    ),
                    "reasons": "; ".join(reasons),
                }
            )

    n_ok = n_sample - len(failures)
    score_df = pd.DataFrame(scores).sort_values(
        ["y_pred_proba", "ticker", "buy_date"],
        ascending=[False, True, True],
        na_position="last",
    )
    print("\nModel scores for sampled Strategy 3 buys:")
    print(score_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if score_df["y_pred_proba"].notna().any():
        p = score_df["y_pred_proba"]
        print(
            f"  y_pred_proba: min={p.min():.4f}, median={p.median():.4f}, "
            f"max={p.max():.4f}  (thresh={proba_thresh:.2f})"
        )
    print(
        f"verify_strategy3: {n_ok}/{n_sample} sampled buys pass "
        f"(hold={hold_months} months, "
        f"y_pred_proba >= {proba_thresh:.2f})"
    )
    if n_no_sell:
        print(f"  Buys with no paired sell in the txn file: {n_no_sell}")
    if failures:
        fail_df = pd.DataFrame(failures)
        print("Failures:")
        print(fail_df.to_string(index=False))
        raise SystemExit(1)


def verify_sec_feature_date(
    n_sample: int = 100,
    max_lag_months: int = 3,
) -> None:
    """
    Sample bargain moments, join each to its latest SEC filing
    (same asof logic as baseline_model.join_events_to_features), and check
    that file_date is within ``max_lag_months`` of the bargain date.
    """
    from baseline_model import join_events_to_features, prepare_bargain_events

    daily_df = load_or_build_daily_prices()
    events_since = _read_trend_down_events(daily_df)
    bargains = load_bargain_events(
        daily_df, sp500_only=True, events=events_since
    )
    if bargains.empty:
        raise SystemExit("No bargain events to verify.")

    n_sample = min(n_sample, len(bargains))
    sample = bargains.sample(n=n_sample).reset_index(drop=True)
    prepared = prepare_bargain_events(sample)
    if prepared.empty:
        raise SystemExit("Sampled bargains had no CIK mapping.")

    sec, feats = _load_prepared_sec_features()
    joined = join_events_to_features(prepared, sec, feats)
    if joined.empty:
        raise SystemExit("No sampled bargains joined to SEC features.")

    out = joined[["ticker", "date", "file_date", "period", "cik"]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["file_date"] = pd.to_datetime(out["file_date"]).dt.normalize()
    out["lag_days"] = (out["date"] - out["file_date"]).dt.days
    # Calendar months between file_date and bargain date (DateOffset-style).
    max_lag = pd.DateOffset(months=int(max_lag_months))
    out["ok"] = out["file_date"] + max_lag >= out["date"]

    print("\nSEC file_date for sampled bargain moments:")
    print(
        out[["ticker", "date", "file_date", "lag_days", "period"]]
        .sort_values(["lag_days", "ticker", "date"], ascending=[False, True, True])
        .to_string(index=False)
    )

    lag = out["lag_days"]
    print(
        f"\nBargain date − SEC file_date (days): "
        f"n={len(out)}, min={lag.min()}, median={lag.median():.1f}, "
        f"mean={lag.mean():.1f}, max={lag.max()}  "
        f"(fail if lag > {max_lag_months} months)"
    )
    n_ok = int(out["ok"].sum())
    print(
        f"verify_sec_feature_date: {n_ok}/{len(out)} within "
        f"{max_lag_months} months of bargain date"
    )
    if n_ok < len(sample):
        print(
            f"  (joined {len(out)}/{n_sample} sampled bargains to SEC features)"
        )

    failures = out.loc[~out["ok"]].copy()
    if not failures.empty:
        print("Failures (file_date more than "
              f"{max_lag_months} months before bargain date):")
        print(
            failures[
                ["ticker", "date", "file_date", "lag_days", "period"]
            ].to_string(index=False)
        )
        raise SystemExit(1)


if __name__ == "__main__":
    # verify_bargain_events_in_batch()
    # verify_strategy3_buys()
    verify_sec_feature_date()
