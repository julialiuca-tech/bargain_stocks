#!/usr/bin/env python3
"""
Central configuration for bargain_stocks.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
STOOQ_BASE_DIR = DATA_DIR / "stock_Stooq_daily_US"
STOOQ_SAVE_DIR = STOOQ_BASE_DIR / "derived_data"

# =============================================================================
# TREND-DOWN DETECTION
# =============================================================================

# Flag when current price is at least this fraction below the historic average.
# 0.1 means current <= historic_avg * 0.9 (10% below).
trend_down_thresh = 0.10

# Prior calendar days of daily closes used for the historic average
# (excludes the evaluation day).
trend_down_history = 15

# After a ticker is flagged, suppress further flags for this many calendar days
# from the flag date (so the first detection date is kept).
trending_down_suppression = 30

# Look ahead this many calendar days from the flag date to record a future price
# (nearest trading day on/after that target; supervision signal).
future_change = 93

# If True, only keep flags that rebound by rebound_short_term_per_thresh
# by the end of rebound_short_term_days (and rewrite date/price to that window).
WAIT_FOR_SHORT_TERM_REBOUND = False

# Short-term rebound filter: keep a flag only if the close at the end of
# this many calendar days after the flag date is up by at least
# rebound_short_term_per_thresh.
rebound_short_term_days = 10

# Minimum fractional rebound by window end (0.03 = +3%).
rebound_short_term_per_thresh = 0.03

# =============================================================================
# MODEL-FILTERED BARGAIN INVESTING (Strategy 3)
# =============================================================================

# Only invest when baseline good_buy model y_pred_proba exceeds this threshold.
GOOD_BUY_PROBA_THRESH = 0.75

# Saved scoring bundle from baseline_model.py (RF top-40 by default).
GOOD_BUY_MODEL_PATH = (
    PROJECT_ROOT / "derived_data" / "models" / "good_buy_model_rf_top40.pkl"
)
