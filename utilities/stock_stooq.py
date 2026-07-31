#!/usr/bin/env python3
"""
Stooq US daily stock-price utilities.

1. Download d_us_txt.zip manually from https://stooq.com/db/h/ (captcha required)
   and save it under data/stooq/.
2. prepare_stooq_data() unzips and reorganizes into data/stock_Stooq_daily_US/.
3. load_stooq_stock_data() / month_end_price_stooq() / price_trend() turn that
   tree into derived CSVs under data/stock_Stooq_daily_US/derived_data/.

Usage:
  python utilities/stock_stooq.py
"""

from __future__ import annotations

import glob
import json
import re
import shutil
import urllib.request
import zipfile
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

STOOQ_DOWNLOAD_DIR = DATA_DIR / "stooq"
ZIP_FILE_PATH = STOOQ_DOWNLOAD_DIR / "d_us_txt.zip"
EXTRACT_DIR = STOOQ_DOWNLOAD_DIR / "extracted"
STOOQ_DATA_DIR = EXTRACT_DIR / "data" / "daily" / "us"  # layout inside the zip

STOOQ_BASE_DIR = DATA_DIR / "stock_Stooq_daily_US"
STOOQ_SAVE_DIR = STOOQ_BASE_DIR / "derived_data"

STOOQ_HISTORY_PAGE = "https://stooq.com/db/h/"
EXCHANGES = ("nasdaq", "nyse", "nysemkt")

STOCK_EXCHANGES = {
    "nasdaq_stock*": "df_nasdaq",
    "nyse_stock*": "df_nyse",
    "nysemkt_stock*": "df_nysemkt",
}

# Filters applied when building month-end prices
DATE_RANGE_START = "2000-01-01"
DATE_RANGE_END = "2030-01-01"
MIN_PRICE = 1.0
MAX_PRICE = 1000.0
TREND_HORIZONS_MONTHS = (12, 6, 3, 1)


# =============================================================================
# MANUAL DOWNLOAD INSTRUCTIONS
# =============================================================================

def _print_manual_instructions(dest: Path = ZIP_FILE_PATH) -> None:
    print()
    print("=" * 72)
    print("Manual Stooq US daily download")
    print("=" * 72)
    print(f"1. Open:  {STOOQ_HISTORY_PAGE}")
    print("2. In the United States row, click the Daily / ASCII link (d_us_txt).")
    print("3. Solve the 4-character image captcha, then download the zip.")
    print(f"4. Save / move the file to:\n      {dest}")
    print("   (File must be named d_us_txt.zip)")
    print("=" * 72)
    print()


# =============================================================================
# PREPARE (unzip + reorganize)
# =============================================================================

def normalize_directory_name(name: str) -> str:
    """e.g. 'nyse stocks' → 'nyse_stocks'."""
    return name.replace(" ", "_").lower()


def flatten_stocks_directory(stocks_dir_path: Path, exchange_name: str) -> list[Path]:
    """
    Flatten Stooq's numbered stock subdirs (1/, 2/, ...) into
    {exchange}_stocks_{n}/ at STOOQ_BASE_DIR.
    """
    created: list[Path] = []
    if not stocks_dir_path.is_dir():
        print(f"  Warning: missing {stocks_dir_path}")
        return created

    numbered = sorted(
        (p for p in stocks_dir_path.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )

    if numbered:
        print(f"  Found {len(numbered)} numbered subdirectories")
        for subdir in numbered:
            new_dir = STOOQ_BASE_DIR / f"{exchange_name}_stocks_{subdir.name}"
            new_dir.mkdir(parents=True, exist_ok=True)
            files_copied = 0
            for file_path in subdir.rglob("*"):
                if file_path.is_file():
                    dest = new_dir / file_path.name
                    if not dest.exists():
                        shutil.copy2(file_path, dest)
                        files_copied += 1
            print(f"    Created {new_dir.name} ({files_copied:,} files)")
            created.append(new_dir)
        return created

    new_dir = STOOQ_BASE_DIR / normalize_directory_name(f"{exchange_name} stocks")
    new_dir.mkdir(parents=True, exist_ok=True)
    files_copied = 0
    for file_path in stocks_dir_path.rglob("*"):
        if file_path.is_file():
            dest = new_dir / file_path.name
            if not dest.exists():
                shutil.copy2(file_path, dest)
                files_copied += 1
    print(f"    Created {new_dir.name} ({files_copied:,} files)")
    created.append(new_dir)
    return created


def process_etfs_directory(etfs_dir_path: Path, exchange_name: str) -> Path | None:
    """Copy ETF files into {exchange}_etfs/."""
    if not etfs_dir_path.is_dir():
        return None
    new_dir = STOOQ_BASE_DIR / normalize_directory_name(f"{exchange_name} etfs")
    new_dir.mkdir(parents=True, exist_ok=True)
    files_copied = 0
    for file_path in etfs_dir_path.rglob("*"):
        if file_path.is_file():
            dest = new_dir / file_path.name
            if not dest.exists():
                shutil.copy2(file_path, dest)
                files_copied += 1
    print(f"    Created {new_dir.name} ({files_copied:,} files)")
    return new_dir


def unzip_stooq_data(zip_path: Path = ZIP_FILE_PATH) -> bool:
    if not zip_path.is_file():
        print(f"Error: zip not found at {zip_path}")
        _print_manual_instructions(zip_path)
        return False
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} → {EXTRACT_DIR}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(EXTRACT_DIR)
    except zipfile.BadZipFile as exc:
        print(f"Error: bad zip file: {exc}")
        return False
    print("Unzip complete")
    return True


def reorganize_directories() -> bool:
    if not STOOQ_DATA_DIR.is_dir():
        print(f"Error: expected extracted tree at {STOOQ_DATA_DIR}")
        return False

    if STOOQ_BASE_DIR.exists():
        print(f"Removing previous prepared data at {STOOQ_BASE_DIR}")
        shutil.rmtree(STOOQ_BASE_DIR)
    STOOQ_BASE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reorganizing from {STOOQ_DATA_DIR}")
    for item in sorted(STOOQ_DATA_DIR.iterdir()):
        if not item.is_dir():
            continue
        parts = item.name.lower().split()
        if len(parts) < 2:
            print(f"  Skipping unexpected directory: {item.name}")
            continue
        exchange_name, dir_type = parts[0], parts[1]
        print(f"\n  Processing: {item.name}")
        if dir_type == "stocks":
            flatten_stocks_directory(item, exchange_name)
        elif dir_type == "etfs":
            process_etfs_directory(item, exchange_name)
        else:
            print(f"  Skipping unknown type: {dir_type}")
    print("\nReorganization complete")
    return True


def verify_directory_structure() -> bool:
    if not STOOQ_BASE_DIR.is_dir():
        print(f"Prepared directory missing: {STOOQ_BASE_DIR}")
        return False

    actual = sorted(p.name for p in STOOQ_BASE_DIR.iterdir() if p.is_dir())
    etf_re = re.compile(rf"^({'|'.join(EXCHANGES)})_etfs$")
    stocks_num_re = re.compile(rf"^({'|'.join(EXCHANGES)})_stocks_(\d+)$")
    stocks_re = re.compile(rf"^({'|'.join(EXCHANGES)})_stocks$")

    print(f"\nPrepared directories ({len(actual)}):")
    for name in actual:
        path = STOOQ_BASE_DIR / name
        n_files = sum(1 for _ in path.glob("*.txt")) + sum(1 for _ in path.glob("*.csv"))
        kind = (
            "etfs"
            if etf_re.match(name)
            else "stocks"
            if stocks_num_re.match(name) or stocks_re.match(name)
            else "other"
        )
        print(f"  {name:25s}  {n_files:>7,} files  [{kind}]")
    return True


def cleanup_extract_dir() -> None:
    if EXTRACT_DIR.exists():
        print(f"Removing temporary extract dir: {EXTRACT_DIR}")
        shutil.rmtree(EXTRACT_DIR)


def prepare_stooq_data(
    zip_path: Path = ZIP_FILE_PATH,
    keep_extract: bool = False,
) -> bool:
    """Unzip d_us_txt.zip and produce data/stock_Stooq_daily_US/."""
    print("=" * 72)
    print("Stooq data preparation")
    print(f"  Zip:    {zip_path}")
    print(f"  Output: {STOOQ_BASE_DIR}")
    print("=" * 72)

    if not unzip_stooq_data(zip_path):
        return False
    if not reorganize_directories():
        return False
    verify_directory_structure()
    if not keep_extract:
        cleanup_extract_dir()

    print("=" * 72)
    print(f"Prepared data ready at: {STOOQ_BASE_DIR}")
    print("=" * 72)
    return True


# =============================================================================
# HELPERS (ported from sec_report_predict/utility_data.py)
# =============================================================================

def get_cik_ticker_mapping() -> tuple[dict[str, str], dict[str, str]]:
    """
    Fetch CIK ↔ ticker maps from SEC company_tickers.json.

    Returns:
        (cik_to_ticker, ticker_to_cik)
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {
        "User-Agent": "bargain_stocks research (local-dev@example.com)",
        "Host": "www.sec.gov",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        cik_to_ticker: dict[str, str] = {}
        ticker_to_cik: dict[str, str] = {}
        for entry in data.values():
            cik = str(entry["cik_str"]).zfill(10)
            ticker = entry["ticker"]
            cik_to_ticker[cik] = ticker
            ticker_to_cik[ticker] = cik
        return cik_to_ticker, ticker_to_cik
    except Exception as exc:
        print(f"Error loading CIK↔ticker mapping: {exc}")
        return {}, {}


def filter_by_date_range(
    df: pd.DataFrame,
    date_col: str,
    start_date: str = DATE_RANGE_START,
    end_date: str = DATE_RANGE_END,
) -> pd.DataFrame:
    return df[
        (df[date_col] >= pd.to_datetime(start_date))
        & (df[date_col] <= pd.to_datetime(end_date))
    ]


def filter_by_price_range(
    df: pd.DataFrame,
    price_col: str,
    min_price: float,
    max_price: float,
) -> pd.DataFrame:
    return df[(df[price_col] >= min_price) & (df[price_col] <= max_price)]


def remove_cik_w_missing_month(month_end_df: pd.DataFrame) -> pd.DataFrame:
    """Drop (cik, ticker) series that have gaps in consecutive year_month values."""
    if month_end_df.empty:
        print("No data provided for missing-month filter")
        return month_end_df

    required_cols = ["cik", "ticker", "year_month"]
    if not all(col in month_end_df.columns for col in required_cols):
        print(f"Missing required columns. Need: {required_cols}")
        return month_end_df

    violations: list[tuple[str, str, list[str]]] = []

    for (cik, ticker), group in month_end_df.groupby(["cik", "ticker"]):
        group = group.sort_values("year_month")
        months = group["year_month"].tolist()
        is_consecutive = True
        missing_months: list[str] = []

        if len(months) > 1:
            month_ints = [int(str(month).replace("-", "")) for month in months]
            for i in range(len(month_ints) - 1):
                current_month = month_ints[i]
                next_month = month_ints[i + 1]
                if current_month % 100 == 12:
                    expected_next = (current_month // 100 + 1) * 100 + 1
                else:
                    expected_next = current_month + 1

                if next_month != expected_next:
                    is_consecutive = False
                    temp_month = current_month
                    while temp_month < next_month:
                        if temp_month % 100 == 12:
                            temp_month = (temp_month // 100 + 1) * 100 + 1
                        else:
                            temp_month += 1
                        if temp_month < next_month:
                            missing_months.append(
                                f"{temp_month // 100:04d}-{temp_month % 100:02d}"
                            )

        if not is_consecutive:
            violations.append((cik, ticker, missing_months))

    if not violations:
        print("No (cik, ticker) pairs with missing months found")
        return month_end_df

    print(f"Removing {len(violations)} (cik, ticker) pairs with missing months")
    violation_pairs = {(cik, ticker) for cik, ticker, _ in violations}
    mask = ~month_end_df.apply(
        lambda row: (row["cik"], row["ticker"]) in violation_pairs, axis=1
    )
    filtered_df = month_end_df[mask].copy()
    print(f"Removed {len(month_end_df) - len(filtered_df):,} records")
    return filtered_df


def price_trend(month_end_df: pd.DataFrame, trend_horizon_in_months: int) -> pd.DataFrame:
    """
    Look-ahead price trends from month-end closes.

    Returns columns: cik, ticker, month_end_date, trend_up_or_down, trend_5per_up,
    price_return, close_price, future_close_price.
    """
    print(f"Computing price trends ({trend_horizon_in_months}-month horizon)...")
    if month_end_df.empty:
        return pd.DataFrame()

    month_end_df = month_end_df.copy()
    month_end_df["year_month_horizon"] = month_end_df["year_month"] + trend_horizon_in_months

    future_df = month_end_df[["cik", "ticker", "year_month", "close_price"]].rename(
        columns={"year_month": "year_month_horizon", "close_price": "future_close_price"}
    )

    trend_df = (
        month_end_df.merge(future_df, on=["cik", "ticker", "year_month_horizon"], how="left")
        .dropna(subset=["future_close_price"])
        .assign(
            trend_up_or_down=lambda x: (x["future_close_price"] > x["close_price"]).astype(int),
            trend_5per_up=lambda x: (x["future_close_price"] > x["close_price"] * 1.05).astype(int),
            price_return=lambda x: x["future_close_price"] / x["close_price"],
        )[
            [
                "cik",
                "ticker",
                "month_end_date",
                "trend_up_or_down",
                "trend_5per_up",
                "price_return",
                "close_price",
                "future_close_price",
            ]
        ]
    )
    print(f"  Trend records: {len(trend_df):,}")
    return trend_df


# =============================================================================
# LOAD / DERIVE
# =============================================================================

def process_stock_directory(
    directory_pattern: str,
    base_dir: Path | str = STOOQ_BASE_DIR,
    verbose: bool = True,
) -> pd.DataFrame:
    """Read closing prices from directories matching pattern (e.g. 'nasdaq_stock*')."""
    if verbose:
        print(f"Processing {directory_pattern}...")
    base_dir = Path(base_dir)
    directories = sorted(glob.glob(str(base_dir / directory_pattern)))

    if not directories:
        if verbose:
            print(f"  No directories matching {base_dir / directory_pattern}")
        return pd.DataFrame()

    if verbose:
        print(f"  Found {len(directories)} directories: {[Path(d).name for d in directories]}")
    all_stock_data: list[pd.DataFrame] = []

    for directory in directories:
        dir_path = Path(directory)
        if verbose:
            print(f"  Processing directory: {dir_path.name}")
        data_files = list(dir_path.glob("*.csv")) + list(dir_path.glob("*.txt"))
        if not data_files:
            if verbose:
                print("    No data files found")
            continue

        preferred = [f for f in data_files if "_" in f.name or "-" in f.name]
        regular = [f for f in data_files if "_" not in f.name and "-" not in f.name]
        if verbose:
            print(
                f"    {len(data_files)} files; "
                f"skipping {len(preferred)} preferred; "
                f"processing {len(regular)} regular"
            )

        for data_file in data_files:
            try:
                if "_" in data_file.name or "-" in data_file.name:
                    continue
                if data_file.stat().st_size == 0:
                    continue

                df = pd.read_csv(data_file)
                if df.empty:
                    continue
                if "<CLOSE>" not in df.columns and "Close" not in df.columns:
                    continue

                ticker = data_file.stem
                if ticker.endswith(".US"):
                    ticker = ticker[:-3]

                if "<CLOSE>" in df.columns:
                    close_col, date_col = "<CLOSE>", "<DATE>"
                else:
                    close_col, date_col = "Close", "Date"

                stock_df = df[[date_col, close_col]].copy()
                stock_df["ticker"] = ticker
                stock_df["exchange"] = dir_path.name
                stock_df = stock_df.rename(columns={date_col: "date", close_col: "close_price"})
                stock_df = stock_df[["ticker", "exchange", "date", "close_price"]]
                all_stock_data.append(stock_df)
            except Exception as exc:
                if verbose:
                    print(f"    Error processing {data_file.name}: {exc}")
                continue

    if not all_stock_data:
        if verbose:
            print(f"  No valid stock data for {directory_pattern}")
        return pd.DataFrame()

    combined_df = pd.concat(all_stock_data, ignore_index=True)
    combined_df["date"] = pd.to_datetime(combined_df["date"], format="%Y%m%d", errors="coerce")
    combined_df = combined_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    if verbose:
        print(
            f"  Processed {len(combined_df):,} records for "
            f"{combined_df['ticker'].nunique():,} tickers"
        )
    return combined_df


def load_stooq_stock_data(base_dir: Path | str = STOOQ_BASE_DIR) -> pd.DataFrame:
    """
    Load closing prices from all prepared Stooq stock directories.

    Returns columns: ticker, cik, exchange, date, close_price.
    Rows without a CIK mapping are dropped.
    """
    base_dir = Path(base_dir)
    print("Loading Stooq stock data...")
    print(f"  Base directory: {base_dir}")

    if not base_dir.is_dir():
        print(f"  Base directory does not exist: {base_dir}")
        print("  Tip: run prepare_stooq_data() first")
        return pd.DataFrame()

    all_dataframes: list[pd.DataFrame] = []
    for directory_pattern, output_name in STOCK_EXCHANGES.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {directory_pattern} -> {output_name}")
        print("=" * 60)
        df = process_stock_directory(directory_pattern, base_dir=base_dir)
        if not df.empty:
            all_dataframes.append(df)

    if not all_dataframes:
        print("No data found in any exchange")
        return pd.DataFrame()

    df_combined = pd.concat(all_dataframes, ignore_index=True)
    df_combined["ticker"] = df_combined["ticker"].str.upper().str.replace(".US", "", regex=False)

    print("\nAdding CIK mappings...")
    _, ticker_to_cik = get_cik_ticker_mapping()
    df_combined["cik"] = df_combined["ticker"].map(ticker_to_cik)

    initial_count = len(df_combined)
    df_combined = df_combined.dropna(subset=["cik"])
    print(f"Dropped {initial_count - len(df_combined):,} records with null CIK")
    print(
        f"Mapped {len(df_combined):,} / {initial_count:,} records "
        f"({(len(df_combined) / initial_count * 100) if initial_count else 0:.1f}%)"
    )

    df_combined = df_combined[["ticker", "cik", "exchange", "date", "close_price"]]
    df_combined = df_combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    print(f"\n{'=' * 60}")
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(df_combined):,}")
    print(f"Unique tickers: {df_combined['ticker'].nunique():,}")
    print(f"Unique exchanges: {df_combined['exchange'].nunique():,}")
    if not df_combined.empty:
        print(f"Date range: {df_combined['date'].min()} to {df_combined['date'].max()}")
    return df_combined


def month_end_price_stooq(df_combined: pd.DataFrame) -> pd.DataFrame:
    """One row per (cik, ticker, year_month) using the last trading day's close."""
    df = df_combined.copy()
    df["year_month"] = df["date"].dt.to_period("M")
    month_end_df = df.loc[df.groupby(["cik", "ticker", "year_month"])["date"].idxmax()].copy()
    month_end_df = month_end_df.rename(columns={"date": "month_end_date"})
    return month_end_df[
        ["cik", "ticker", "month_end_date", "close_price", "year_month"]
    ].reset_index(drop=True)


def build_derived_stooq_data(
    force_rebuild_month_end: bool = False,
    date_start: str = DATE_RANGE_START,
    date_end: str = DATE_RANGE_END,
) -> bool:
    """Load prepared Stooq files and write month-end + trend CSVs to derived_data/."""
    STOOQ_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    month_end_price_file = STOOQ_SAVE_DIR / "month_end_price_stooq.csv"

    print("Loading stock data and generating month-end prices...")
    df_combined = load_stooq_stock_data()
    if df_combined.empty:
        print("No stock data loaded.")
        return False

    if force_rebuild_month_end or not month_end_price_file.exists():
        df_combined = filter_by_date_range(
            df_combined, "date", start_date=date_start, end_date=date_end
        )
        month_end_df = month_end_price_stooq(df_combined)
        month_end_df = remove_cik_w_missing_month(month_end_df)
        month_end_df = filter_by_price_range(
            month_end_df, "close_price", min_price=MIN_PRICE, max_price=MAX_PRICE
        )
        month_end_df.to_csv(month_end_price_file, index=False)
        print(f"Month-end prices saved to: {month_end_price_file}")
    else:
        month_end_df = pd.read_csv(month_end_price_file)
        print(f"Using existing month-end prices: {month_end_price_file}")

    month_end_df["year_month"] = pd.to_datetime(month_end_df["year_month"]).dt.to_period("M")

    for horizon in TREND_HORIZONS_MONTHS:
        trend_df = price_trend(month_end_df, trend_horizon_in_months=horizon)
        if len(trend_df) > 0:
            output_file = STOOQ_SAVE_DIR / f"price_trends_{horizon}month.csv"
            trend_df.to_csv(output_file, index=False)
            print(f"{horizon}-month trends saved to: {output_file}")

    print("Derived Stooq data complete.")
    return True


# =============================================================================
# MAIN
# =============================================================================

def _path_mtime_date(path: Path) -> date | None:
    """Return the local calendar date of path's mtime, or None if missing."""
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).date()


def main() -> None:
    today = date.today()

    prepared_date = _path_mtime_date(STOOQ_BASE_DIR)
    if prepared_date != today:
        zip_date = _path_mtime_date(ZIP_FILE_PATH)
        if zip_date == today:
            print(
                f"Raw Stooq zip at {ZIP_FILE_PATH} is up-to-date "
                f"(last modified {zip_date})."
            )
            print("Using it to bring stock_Stooq_daily_US/ up to date...")
            if not prepare_stooq_data():
                return
        else:
            _print_manual_instructions()
            return
    else:
        print(
            f"Stock data at {STOOQ_BASE_DIR} seems up-to-date "
            f"(last modified {prepared_date})."
        )

    build_derived_stooq_data()


if __name__ == "__main__":
    main()
