#!/usr/bin/env python3
"""
Baseline classifier: is a trend-down "bargain" a good buy?

Label:
  good_buy = 1 if price_change > 1 else 0
  (price_change = future_price / price from trending_down_stocks.csv)

Features:
  SEC featurized quarterly tags from sec_report_predict, augmented with
  ratio features (_augment) and quarter-over-quarter gradients (_change_qN),
  matching prep_data_feature_label() in sec_report_predict/baseline_model.py.

Join:
  1. Map bargain tickers → CIK.
  2. merge_asof each bargain date to the latest SEC filing with
     file_date <= bargain date (same CIK).
  3. Attach (augmented) featurized row on (cik, period).

Usage:
  python baseline_model.py

Train/val split (edit SPLIT_STRATEGY near top of file):
  None                — random row split (current default)
  {"date": "bottom"}  — time-based, ~TRAIN_FRAC of rows in train
  {"cik": "random"}   — hold out random companies

Scoring (after training):
  from baseline_model import score_bargain_events
  scored = score_bargain_events(
      events_df,  # needs date + ticker (or cik)
      "derived_data/models/good_buy_model_rf_top40.pkl",
  )  # adds y_pred, y_pred_proba after SEC feature lookup
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from utilities.stock_stooq import get_cik_ticker_mapping

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = config.PROJECT_ROOT
SEC_PROJECT_ROOT = PROJECT_ROOT.parent / "sec_report_predict"

TRENDING_DOWN_FILE = config.STOOQ_SAVE_DIR / "trending_down_stocks.csv"
SEC_TABLE_FILE = SEC_PROJECT_ROOT / "data" / "sec_table_w_file_date.csv"
FEATURIZED_FILE = (
    SEC_PROJECT_ROOT / "data" / "featurized_since_2011" / "featurized_all_quarters.csv"
)
# Offline fallback (same file used by sec_report_predict).
COMPANY_TICKERS_EXCHANGE_FILE = (
    SEC_PROJECT_ROOT / "data" / "company_tickers_exchange.json"
)

MODEL_DIR = PROJECT_ROOT / "derived_data" / "models"
Y_LABEL = "good_buy"
# Match sec_report_predict/config.py ML feature settings.
USE_RATIO_FEATURES = True
FILTER_OUTLIERS_FROM_RATIOS = True
SUFFIXES_TO_ENHANCE_W_GRADIENT = ("_current", "_augment")
FEATURE_SUFFIXES = ("_current", "_augment") if USE_RATIO_FEATURES else ("_current",)
QUARTER_GRADIENTS = [1, 2, 4]
# QUARTER_GRADIENTS = []
COMPLETENESS_THRESHOLD = 0.20
TOP_K_FEATURES = 50
TRAIN_FRAC = 0.70
RANDOM_SEED = 42

# Train/val split experiments (same idea as sec_report_predict):
#   {"date": "bottom"}  — earliest dates until ~TRAIN_FRAC of rows (time-based)
#   {"date": "top"}     — latest dates in train
#   {"cik": "random"}   — hold out random companies
#   {"cik": "bottom"}   — split companies by sorted CIK
#   None                — random row split
SPLIT_STRATEGY: dict[str, str] | None = {"cik": "random"} 
DEBUG_PRINT = False
# Edges for y_pred_proba calibration-style reporting on val.
PROBA_BIN_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Meta columns that are not model features.
META_COLS = {
    "ticker",
    "date",
    "price",
    "historical_average",
    "future_date",
    "future_price",
    "price_change",
    "cik",
    "period",
    "form_type",
    "file_date",
    "form",
    "data_qtr",
    Y_LABEL,
}


def _normalize_cik(series: pd.Series) -> pd.Series:
    """Zero-pad CIKs to 10 digits (string)."""
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.lstrip("0")
        .replace({"": "0"})
        .str.zfill(10)
    )


def _import_sec_feature_tools():
    """
    Import compute_ratio_features / enhance_tags_w_gradient / outlier flagging
    from sec_report_predict without clobbering bargain_stocks' config module.
    """
    sec_root = str(SEC_PROJECT_ROOT)
    if not Path(sec_root).is_dir():
        raise FileNotFoundError(f"Missing sec_report_predict at {sec_root}")

    saved_config = sys.modules.get("config")
    # Drop bargain config so sec modules bind to sec_report_predict/config.py.
    sys.modules.pop("config", None)
    if sec_root not in sys.path:
        sys.path.insert(0, sec_root)
    try:
        from feature_augment import (  # type: ignore
            compute_ratio_features,
            flag_outliers_by_hard_limits,
        )
        from featurize import enhance_tags_w_gradient  # type: ignore
        return (
            compute_ratio_features,
            flag_outliers_by_hard_limits,
            enhance_tags_w_gradient,
        )
    finally:
        if saved_config is not None:
            sys.modules["config"] = saved_config


def prep_featurized_features(
    df_featurized_data: pd.DataFrame,
    quarters_for_gradient_comp: list[int] | None = None,
) -> pd.DataFrame:
    """
    Mirror the feature side of sec_report_predict.baseline_model.prep_data_feature_label:
    dedupe → ratio features → gradient features → outlier rejection.

    Does *not* join labels; bargain labels are attached later via merge_asof.
    """
    if quarters_for_gradient_comp is None:
        quarters_for_gradient_comp = list(QUARTER_GRADIENTS)

    (
        compute_ratio_features,
        flag_outliers_by_hard_limits,
        enhance_tags_w_gradient,
    ) = _import_sec_feature_tools()

    df_features = df_featurized_data.copy()
    print(f"Features loaded: {df_features.shape}")

    before = len(df_features)
    df_features = df_features.drop_duplicates(subset=["cik", "period"], keep="last")
    if len(df_features) != before:
        print(f"Removed {before - len(df_features):,} duplicate (cik, period) rows")
        print(f"Features after deduplication: {df_features.shape}")

    if USE_RATIO_FEATURES:
        df_features = compute_ratio_features(df_features)
        print(f"Ratio features computed: {df_features.shape}")

    if quarters_for_gradient_comp is not None:
        df_features["period"] = (
            df_features["period"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
        )
        df_features = enhance_tags_w_gradient(
            df_features,
            df_extra_history_for_gradient=None,
            quarters_for_gradient_comp=quarters_for_gradient_comp,
            suffixes_to_enhance=list(SUFFIXES_TO_ENHANCE_W_GRADIENT),
        )
        print(f"Gradient features loaded: {df_features.shape}")

    if FILTER_OUTLIERS_FROM_RATIOS and USE_RATIO_FEATURES:
        df_features = flag_outliers_by_hard_limits(df_features)
        df_features = df_features[df_features["flag_outlier"] == False].copy()
        df_features.drop(columns=["flag_outlier"], inplace=True)
        print(
            f"Outliers filtered from ratio features: "
            f"{df_features.shape} records remaining"
        )

    df_features["cik"] = _normalize_cik(df_features["cik"])
    df_features["period"] = (
        df_features["period"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )
    return df_features


# =============================================================================
# DATA LOADING / JOIN
# =============================================================================

def _load_ticker_to_cik() -> dict[str, str]:
    """Prefer local company_tickers_exchange.json; fall back to SEC API."""
    path = COMPANY_TICKERS_EXCHANGE_FILE
    if path.exists():
        payload = json.loads(path.read_text())
        rows = payload.get("data", [])
        # fields: cik, name, ticker, exchange
        ticker_to_cik = {
            str(row[2]).upper(): str(row[0]).zfill(10)
            for row in rows
            if len(row) >= 3 and row[2]
        }
        print(f"Loaded {len(ticker_to_cik):,} ticker→CIK from {path.name}")
        return ticker_to_cik

    _cik_to_ticker, ticker_to_cik = get_cik_ticker_mapping()
    if not ticker_to_cik:
        raise RuntimeError(
            "Could not load ticker→CIK mapping "
            f"(missing {path} and SEC API unavailable)."
        )
    return {str(t).upper(): str(c).zfill(10) for t, c in ticker_to_cik.items()}


def load_bargain_events(path: Path = TRENDING_DOWN_FILE) -> pd.DataFrame:
    """Load trend-down events and attach CIK + binary label."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python trend_down_stocks.py first."
        )

    events = pd.read_csv(path)
    events["date"] = pd.to_datetime(events["date"])
    events["ticker"] = events["ticker"].astype(str).str.upper()
    events = events.dropna(subset=["price_change"]).copy()
    events[Y_LABEL] = (events["price_change"] > 1.0).astype(int)

    ticker_to_cik = _load_ticker_to_cik()
    events["cik"] = events["ticker"].map(ticker_to_cik)
    before = len(events)
    events = events.dropna(subset=["cik"]).copy()
    events["cik"] = _normalize_cik(events["cik"])
    print(
        f"Bargain events: {len(events):,} with CIK "
        f"(dropped {before - len(events):,} without mapping); "
        f"good_buy rate={(events[Y_LABEL].mean() * 100):.1f}%"
    )
    return events.reset_index(drop=True)


def load_sec_table(path: Path = SEC_TABLE_FILE) -> pd.DataFrame:
    """Load (cik, period, form_type, file_date) auxiliary table."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Build it via "
            "sec_report_predict/utility_data.py:build_sec_table_w_file_date()."
        )
    sec = pd.read_csv(path)
    sec["cik"] = _normalize_cik(sec["cik"])
    sec["period"] = sec["period"].astype(str)
    sec["file_date"] = pd.to_datetime(sec["file_date"])
    sec = sec.dropna(subset=["cik", "file_date", "period"]).copy()
    # Prefer one filing per (cik, file_date); keep last period if ties.
    sec = sec.drop_duplicates(subset=["cik", "file_date"], keep="last")
    return sec.reset_index(drop=True)


def load_featurized_sec(path: Path = FEATURIZED_FILE) -> pd.DataFrame:
    """Load raw SEC featurized quarterly rows (augmentation happens in prep)."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}.")
    print(f"Loading featurized data from {path} ...")
    feats = pd.read_csv(path)
    feats["cik"] = _normalize_cik(feats["cik"])
    feats["period"] = (
        feats["period"].astype(str).str.replace(r"\.0$", "", regex=True)
    )
    print(f"  Raw featurized shape: {feats.shape}")
    return feats


def join_events_to_features(
    events: pd.DataFrame,
    sec: pd.DataFrame,
    feats: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each bargain event, take the latest SEC filing with file_date <= date,
    then attach featurized columns on (cik, period).
    """
    print("Joining events → latest SEC filing (merge_asof backward) ...")
    # pandas >=2.2 requires the asof keys to be *globally* sorted (not by cik first).
    joined = pd.merge_asof(
        events.sort_values("date"),
        sec.sort_values("file_date"),
        by="cik",
        left_on="date",
        right_on="file_date",
        direction="backward",
    )
    n_with_filing = joined["period"].notna().sum()
    print(
        f"  Events with a prior filing: {n_with_filing:,} / {len(joined):,} "
        f"({n_with_filing / max(len(joined), 1) * 100:.1f}%)"
    )

    joined = joined.dropna(subset=["period"]).copy()
    joined["period"] = joined["period"].astype(str)

    print("Joining filings → featurized SEC tags on (cik, period) ...")
    # Avoid colliding non-feature cols already present on events/sec.
    feat_meta = {"cik", "period", "form", "data_qtr"}
    feat_cols = [c for c in feats.columns if c not in feat_meta or c in ("cik", "period")]
    df = joined.merge(feats[feat_cols], on=["cik", "period"], how="inner")
    msg = f"  Rows with features: {len(df):,}"
    if Y_LABEL in df.columns and len(df):
        msg += f" (good_buy rate={(df[Y_LABEL].mean() * 100):.1f}%)"
    print(msg)
    return df


def prepare_bargain_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize bargain event rows for SEC feature lookup.

    Requires columns: date, and either cik or ticker.
    price_change / good_buy are optional (needed only for labeled eval).
    """
    if events is None or len(events) == 0:
        return pd.DataFrame()
    out = events.copy()
    if "date" not in out.columns:
        raise KeyError("Bargain events need a 'date' column")
    out["date"] = pd.to_datetime(out["date"])

    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].astype(str).str.upper()

    if "cik" not in out.columns or out["cik"].isna().all():
        if "ticker" not in out.columns:
            raise KeyError("Bargain events need 'cik' or 'ticker'")
        ticker_to_cik = _load_ticker_to_cik()
        out["cik"] = out["ticker"].map(ticker_to_cik)

    before = len(out)
    out = out.dropna(subset=["cik", "date"]).copy()
    out["cik"] = _normalize_cik(out["cik"])
    if before != len(out):
        print(f"Dropped {before - len(out):,} events without CIK/date")

    if "price_change" in out.columns and Y_LABEL not in out.columns:
        valid = out["price_change"].notna()
        out.loc[valid, Y_LABEL] = (out.loc[valid, "price_change"] > 1.0).astype(int)

    return out.reset_index(drop=True)


# =============================================================================
# MODELING
# =============================================================================

def select_feature_columns(
    df: pd.DataFrame,
    strategy: str = "completeness",
) -> list[str]:
    """Pick numeric feature columns (by suffix / completeness)."""
    suffix_cols = [
        c for c in df.columns
        if any(c.endswith(suf) for suf in FEATURE_SUFFIXES)
    ]
    change_cols = [
        c for c in df.columns
        if "_change" in c and c not in suffix_cols and c not in META_COLS
    ]
    feature_cols = suffix_cols + change_cols
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    if strategy == "all":
        return feature_cols
    if strategy == "current":
        return [c for c in feature_cols if c.endswith("_current")]
    if strategy == "completeness":
        completeness = df[feature_cols].notna().mean()
        return completeness[completeness >= COMPLETENESS_THRESHOLD].index.tolist()
    raise ValueError(f"Unknown feature strategy: {strategy}")


def _random_split(
    df: pd.DataFrame,
    train_prop: float,
    random_seed: int | None = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random row-level train/val split."""
    total = len(df)
    train_size = int(total * train_prop)
    rng = np.random.default_rng(random_seed)
    train_idx = rng.choice(total, size=train_size, replace=False)
    train_mask = np.zeros(total, dtype=bool)
    train_mask[train_idx] = True
    df_train = df.iloc[train_mask].copy()
    df_val = df.iloc[~train_mask].copy()
    print(
        f"Random split: train={len(df_train):,} ({len(df_train) / max(total, 1):.1%}), "
        f"val={len(df_val):,} ({len(df_val) / max(total, 1):.1%})"
    )
    return df_train, df_val


def split_train_val_by_column(
    df: pd.DataFrame,
    train_prop: float,
    by_column: str | None,
    split_for_training: str = "random",
    random_seed: int | None = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by unique values of `by_column` until ~train_prop of rows are in train.

    split_for_training:
      - 'random': shuffle unique values, then take a prefix for train
      - 'bottom': sorted ascending (earliest dates / smallest cik first → train)
      - 'top': sorted descending (latest / largest first → train)
    If by_column is None, falls back to a random row split.
    """
    if split_for_training not in ("random", "top", "bottom"):
        raise ValueError(
            f"Invalid split_for_training={split_for_training!r}; "
            "use 'random', 'top', or 'bottom'."
        )
    if not 0.0 <= train_prop <= 1.0:
        raise ValueError(f"train_prop must be in [0, 1], got {train_prop}")

    if by_column is None:
        return _random_split(df, train_prop, random_seed=random_seed)
    if by_column not in df.columns:
        raise KeyError(f"Column {by_column!r} not in dataframe")

    value_counts = df[by_column].value_counts()
    if split_for_training == "random":
        values = value_counts.index.to_numpy().copy()
        rng = np.random.default_rng(random_seed)
        rng.shuffle(values)
    elif split_for_training == "top":
        values = value_counts.sort_index(ascending=False).index.to_numpy()
    else:  # bottom
        values = value_counts.sort_index(ascending=True).index.to_numpy()

    target_cutoff = int(len(df) * train_prop)
    shuffled_counts = value_counts.loc[values]
    cumulative = shuffled_counts.cumsum()
    train_values = shuffled_counts[cumulative <= target_cutoff].index.tolist()
    # Ensure non-empty train when possible.
    if not train_values and len(values):
        train_values = [values[0]]

    train_mask = df[by_column].isin(train_values)
    df_train = df.loc[train_mask].copy()
    df_val = df.loc[~train_mask].copy()

    n_train_keys = len(train_values)
    n_val_keys = value_counts.shape[0] - n_train_keys
    print(
        f"Split by '{by_column}' ({split_for_training}): "
        f"train={len(df_train):,} rows / {n_train_keys:,} {by_column}s "
        f"({len(df_train) / max(len(df), 1):.1%}), "
        f"val={len(df_val):,} rows / {n_val_keys:,} {by_column}s "
        f"({len(df_val) / max(len(df), 1):.1%})"
    )
    if len(df_train) and len(df_val):
        print(
            f"  train {by_column} range: {df_train[by_column].min()} → {df_train[by_column].max()}"
        )
        print(
            f"  val   {by_column} range: {df_val[by_column].min()} → {df_val[by_column].max()}"
        )
    return df_train, df_val


def split_data_for_train_val(
    df_work: pd.DataFrame,
    train_val_split_prop: float = TRAIN_FRAC,
    train_val_split_strategy: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Experiment-friendly train/val split (mirrors sec_report_predict).

    train_val_split_strategy examples:
      {"date": "bottom"}, {"cik": "random"}, None (random rows)
    """
    if train_val_split_strategy is None:
        train_val_split_strategy = SPLIT_STRATEGY

    try:
        if not train_val_split_strategy:
            print("Splitting data randomly (row-level)")
            return split_train_val_by_column(
                df_work, train_val_split_prop, None, "random"
            )
        by_column = list(train_val_split_strategy.keys())[0]
        split_for_training = train_val_split_strategy[by_column]
        print(f"Splitting data by {by_column} using {split_for_training} strategy")
        return split_train_val_by_column(
            df_work, train_val_split_prop, by_column, split_for_training
        )
    except Exception as exc:
        print(f"Error with split_strategy ({exc}); falling back to random split")
        return split_train_val_by_column(
            df_work, train_val_split_prop, None, "random"
        )


def split_train_val_by_date(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backward-compatible wrapper: earliest dates until ~train_frac of rows.

    Prefer split_data_for_train_val(..., {"date": "bottom"}) for experiments.
    """
    return split_data_for_train_val(
        df,
        train_val_split_prop=train_frac,
        train_val_split_strategy={"date": "bottom"},
    )


def train_and_eval(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: list[str],
    model_name: str = "rf",
    debug_print: bool | None = None,
) -> dict:
    """Train RF or XGB baseline and optionally print validation metrics."""
    if debug_print is None:
        debug_print = DEBUG_PRINT
    if not feature_cols:
        raise RuntimeError("No feature columns selected.")
    if model_name not in ("rf", "xgb"):
        raise ValueError(f"Unknown model_name={model_name!r}; use 'rf' or 'xgb'.")

    X_train = df_train[feature_cols].copy()
    X_val = df_val[feature_cols].copy()
    y_train = df_train[Y_LABEL]
    y_val = df_val[Y_LABEL]

    # Median impute from training fold only.
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)

    if model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            class_weight="balanced_subsample",
        )
        display_name = "RandomForest"
    else:
        # Match sec_report_predict/utility_binary_classifier.py defaults.
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            eval_metric="logloss",
        )
        display_name = "XGBoost"

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "model": model_name,
        "n_features": len(feature_cols),
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, y_proba)),
        "corr_proba_price_change": float(
            np.corrcoef(y_proba, df_val["price_change"].to_numpy())[0, 1]
        ),
    }

    importance = (
        pd.DataFrame(
            {"feature": feature_cols, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    if debug_print:
        print(f"\nValidation metrics ({display_name}):")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:28s} = {v:.4f}")
            else:
                print(f"  {k:28s} = {v}")

        print(f"\nTop 15 features ({display_name}):")
        print(importance.head(15).to_string(index=False))

    return {
        "model": model,
        "model_name": model_name,
        "metrics": metrics,
        "feature_importance": importance,
        "feature_cols": feature_cols,
        "impute_medians": medians,
    }


def save_model_bundle(result: dict, path: Path) -> Path:
    """
    Persist a trained model + scoring metadata for later inference.

    model_bundle keys:
      model, model_name, feature_cols, impute_medians, metrics, y_label
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "model": result["model"],
        "model_name": result["model_name"],
        "feature_cols": list(result["feature_cols"]),
        "impute_medians": result["impute_medians"],
        "metrics": result["metrics"],
        "y_label": Y_LABEL,
    }
    with path.open("wb") as f:
        pickle.dump(model_bundle, f)
    print(
        f"Saved model_bundle ({model_bundle['model_name']}, "
        f"{len(model_bundle['feature_cols'])} features) → {path}"
    )
    return path


def load_model_bundle(path: Path) -> dict:
    """Load a model_bundle written by save_model_bundle()."""
    with Path(path).open("rb") as f:
        return pickle.load(f)


def score_feature_frame(df: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    """
    Score rows that already have (most of) the needed feature columns.

    Returns a copy of df with y_pred and y_pred_proba added.
    Missing feature columns / values are filled with training medians.
    """
    feature_cols = list(model_bundle["feature_cols"])
    # Build in one shot to avoid fragmented frame.insert PerformanceWarnings.
    X = df.reindex(columns=feature_cols)

    medians = model_bundle["impute_medians"]
    if isinstance(medians, pd.Series):
        X = X.fillna(medians.reindex(feature_cols))
    else:
        X = X.fillna(medians)

    model = model_bundle["model"]
    out = df.copy()
    out["y_pred"] = model.predict(X)
    out["y_pred_proba"] = model.predict_proba(X)[:, 1]
    return out


def score_bargain_events(
    events: pd.DataFrame,
    model_bundle: dict | Path | str,
    sec: pd.DataFrame | None = None,
    feats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Score bargain events: look up latest SEC features and predict good_buy proba.

    Args:
        events: Bargain rows with at least (date, ticker) or (date, cik).
        model_bundle: Model bundle dict, or path to a good_buy_model_*.pkl.
        sec: Optional preloaded sec_table_w_file_date (loaded if None).
        feats: Optional preloaded *prepared* featurized table (ratios + gradients).
               If None, loads and runs prep_featurized_features() (slow).

    Returns:
        Events that successfully joined to SEC features, with y_pred and
        y_pred_proba added. Events without a prior filing / features are dropped.
    """
    if isinstance(model_bundle, (str, Path)):
        model_bundle = load_model_bundle(model_bundle)

    events_prep = prepare_bargain_events(events)
    if events_prep.empty:
        print("No bargain events to score")
        return events_prep.copy()

    if sec is None:
        sec = load_sec_table()
    if feats is None:
        feats = prep_featurized_features(
            load_featurized_sec(),
            quarters_for_gradient_comp=list(QUARTER_GRADIENTS),
        )

    joined = join_events_to_features(events_prep, sec, feats)
    if joined.empty:
        print("No events could be joined to SEC features")
        return joined

    scored = score_feature_frame(joined, model_bundle)
    print(
        f"Scored {len(scored):,} / {len(events_prep):,} events "
        f"with model={model_bundle.get('model_name', '?')} "
        f"({len(model_bundle['feature_cols'])} features)"
    )
    return scored


def report_val_proba_bins(
    df_val: pd.DataFrame,
    result: dict,
    bin_edges: list[float] | None = None,
    model_label: str | None = None,
) -> pd.DataFrame:
    """
    Bin validation rows by y_pred_proba; report volume and accuracy per bin.

    Accuracy in a bin = fraction of rows where the model's hard prediction
    (threshold 0.5) matches the true label.
    """
    if bin_edges is None:
        bin_edges = list(PROBA_BIN_EDGES)
    if Y_LABEL not in df_val.columns:
        raise KeyError(f"df_val needs label column {Y_LABEL!r}")

    model_bundle = {
        "model": result["model"],
        "feature_cols": result["feature_cols"],
        "impute_medians": result["impute_medians"],
    }
    scored = score_feature_frame(df_val, model_bundle)
    y_true = scored[Y_LABEL].astype(int)
    y_proba = scored["y_pred_proba"].astype(float)
    y_pred = scored["y_pred"].astype(int)

    # Closed on the right so y_pred_proba == 1.0 falls in the last bin.
    labels = [
        f"[{bin_edges[i]:.1f}, {bin_edges[i + 1]:.1f}]"
        if i == 0
        else f"({bin_edges[i]:.1f}, {bin_edges[i + 1]:.1f}]"
        for i in range(len(bin_edges) - 1)
    ]
    bins = pd.cut(
        y_proba,
        bins=bin_edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    total = len(scored)
    rows = []
    for label in labels:
        mask = bins == label
        n = int(mask.sum())
        if n == 0:
            acc = float("nan")
            mean_proba = float("nan")
            pos_rate = float("nan")
        else:
            acc = float((y_pred[mask] == y_true[mask]).mean())
            mean_proba = float(y_proba[mask].mean())
            pos_rate = float(y_true[mask].mean())
        rows.append(
            {
                "proba_bin": label,
                "n": n,
                "pct": n / max(total, 1),
                "accuracy": acc,
                "mean_proba": mean_proba,
                "good_buy_rate": pos_rate,
            }
        )

    report = pd.DataFrame(rows)
    title = model_label or result.get("model_name", "model")
    print("\n" + "=" * 72)
    print(f"Val y_pred_proba bins ({title}, n={total:,})")
    print("=" * 72)
    display = report.copy()
    display["pct"] = display["pct"].map(lambda x: f"{100 * x:.1f}%")
    for col in ("accuracy", "mean_proba", "good_buy_rate"):
        display[col] = display[col].map(
            lambda x: f"{x:.4f}" if pd.notna(x) else "n/a"
        )
    print(display.to_string(index=False))
    return report


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 72)
    print("Baseline model: bargain good_buy from SEC features")
    print("=" * 72)

    events = load_bargain_events()
    sec = load_sec_table()
    feats = load_featurized_sec()
    feats = prep_featurized_features(
        feats,
        quarters_for_gradient_comp=list(QUARTER_GRADIENTS),
    )
    df = join_events_to_features(events, sec, feats)
    if df.empty:
        raise RuntimeError("Joined dataset is empty; check CIK/period coverage.")

    feature_cols = select_feature_columns(df, strategy="completeness")
    print(
        f"Selected {len(feature_cols):,} features "
        f"(completeness >= {COMPLETENESS_THRESHOLD:.0%})"
    )

    df_train, df_val = split_data_for_train_val(
        df,
        train_val_split_prop=TRAIN_FRAC,
        train_val_split_strategy=SPLIT_STRATEGY,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
 
    results = {}
    for model_name in ("rf", "xgb"):
        full = train_and_eval(
            df_train, df_val, feature_cols, model_name=model_name
        )
        results[model_name] = full
        importance_path = MODEL_DIR / f"feature_importance_ranking_{model_name}.csv"
        full["feature_importance"].to_csv(importance_path, index=False)
        print(f"\nSaved feature importance to {importance_path}")

        top_k_cols = (
            full["feature_importance"]
            .head(TOP_K_FEATURES)["feature"]
            .tolist()
        )
        print(
            f"\nRetraining {model_name} with top {len(top_k_cols)} features "
            f"by importance..."
        )
        topk = train_and_eval(
            df_train, df_val, top_k_cols, model_name=model_name
        )
        results[f"{model_name}_top{TOP_K_FEATURES}"] = topk
        topk_path = (
            MODEL_DIR / f"feature_importance_ranking_{model_name}_top{TOP_K_FEATURES}.csv"
        )
        topk["feature_importance"].to_csv(topk_path, index=False)
        print(f"Saved top-{TOP_K_FEATURES} feature importance to {topk_path}")

        # Final scoring models: top-K feature set (enough for bargain-time inference).
        save_model_bundle(
            topk,
            MODEL_DIR / f"good_buy_model_{model_name}_top{TOP_K_FEATURES}.pkl",
        )

    print("\n" + "=" * 72)
    print("Model comparison (validation):")
    print("=" * 72)
    rows = []
    for name, result in results.items():
        m = result["metrics"]
        rows.append(
            {
                "model": name,
                "n_features": m["n_features"],
                "accuracy": f"{m['accuracy']:.4f}",
                "precision": f"{m['precision']:.4f}",
                "recall": f"{m['recall']:.4f}",
                "roc_auc": f"{m['roc_auc']:.4f}",
                "corr_proba_price_change": f"{m['corr_proba_price_change']:.4f}",
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    best = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    print(
        f"\nBest by ROC-AUC: {best} "
        f"({results[best]['metrics']['roc_auc']:.4f})"
    )
    report_val_proba_bins(
        df_val,
        results[best],
        bin_edges=PROBA_BIN_EDGES,
        model_label=best,
    )


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # Happens when stdout is piped to a consumer that exits early (e.g. head).
        # Exit quietly so this is not mistaken for a training failure.
        try:
            sys.stdout.close()
        except Exception:
            pass
        sys.exit(0)
