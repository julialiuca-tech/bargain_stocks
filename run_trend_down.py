#!/usr/bin/env python3
"""
Automate the "latest trend-down bargains" refresh.

Only step 1 needs a human (Stooq captcha). Everything after that is automatic:

  1. Open https://stooq.com/db/h/ and wait for you to download d_us_txt.zip
  2. Move the zip into data/stooq/ (replacing any older copy)
  3. Unzip + reorganize via utilities.stock_stooq.prepare_stooq_data()
  4. Run trend_down_stocks --today --days=21

Usage:
  python run_trend_down.py
  python run_trend_down.py --days 21
  python run_trend_down.py --zip ~/Downloads/d_us_txt.zip   # skip wait; use this file
  python run_trend_down.py --skip-download                  # reuse zip already in data/stooq/
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path

from utilities.stock_stooq import (
    STOOQ_HISTORY_PAGE,
    STOOQ_SAVE_DIR,
    ZIP_FILE_PATH,
    prepare_stooq_data,
)
from trend_down_stocks import print_current_bargains

DOWNLOADS_DIR = Path.home() / "Downloads"
DAILY_PRICES_CACHE = STOOQ_SAVE_DIR / "daily_prices.pkl"
DEFAULT_DAYS = 21
POLL_SECONDS = 2.0
STABLE_SECONDS = 4.0  # zip size must stop changing before we treat it as done


def _zip_candidates(directory: Path) -> list[Path]:
    """Possible Stooq US daily zip names under directory (Chrome may rename copies)."""
    if not directory.is_dir():
        return []
    patterns = ("d_us_txt.zip", "d_us_txt*.zip", "d_us_txt (*).zip")
    found: set[Path] = set()
    for pat in patterns:
        found.update(directory.glob(pat))
    # Ignore incomplete browser downloads.
    return sorted(
        (
            p
            for p in found
            if p.is_file()
            and not p.name.endswith((".crdownload", ".partial", ".download", ".tmp"))
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _is_download_in_progress(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    for p in directory.iterdir():
        name = p.name.lower()
        if "d_us_txt" in name and name.endswith(
            (".crdownload", ".partial", ".download", ".tmp")
        ):
            return True
    return False


def _file_stable(path: Path, wait: float = STABLE_SECONDS) -> bool:
    """True if path exists and size is unchanged for ``wait`` seconds."""
    if not path.is_file():
        return False
    size1 = path.stat().st_size
    time.sleep(wait)
    if not path.is_file():
        return False
    size2 = path.stat().st_size
    return size1 == size2 and size1 > 0


def _prompt_manual_download() -> None:
    print()
    print("=" * 72)
    print("Step 1 (manual): download Stooq US daily ASCII zip")
    print("=" * 72)
    print(f"1. Open:  {STOOQ_HISTORY_PAGE}")
    print("2. United States row → Daily / ASCII (d_us_txt).")
    print("3. Solve the captcha and download the zip.")
    print(f"4. Leave the file in {DOWNLOADS_DIR} (default) — this script will move it.")
    print(f"   Target install path: {ZIP_FILE_PATH}")
    print("=" * 72)
    print()
    try:
        webbrowser.open(STOOQ_HISTORY_PAGE)
        print(f"Opened browser to {STOOQ_HISTORY_PAGE}")
    except Exception as exc:
        print(f"Could not open browser automatically ({exc}); open the URL manually.")


def wait_for_downloaded_zip(
    downloads_dir: Path = DOWNLOADS_DIR,
    already_present: Path | None = None,
    timeout_sec: float | None = None,
) -> Path:
    """
    Wait until a finished d_us_txt*.zip appears under Downloads (or return
    ``already_present`` if provided and valid).
    """
    if already_present is not None:
        path = Path(already_present).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--zip not found: {path}")
        if not _file_stable(path, wait=1.0):
            raise RuntimeError(f"Zip still changing size: {path}")
        return path

    print(f"Watching {downloads_dir} for d_us_txt*.zip ...")
    print("(Ctrl-C to abort)")
    start = time.time()
    last_msg = 0.0
    while True:
        if timeout_sec is not None and (time.time() - start) > timeout_sec:
            raise TimeoutError(
                f"Timed out after {timeout_sec:.0f}s waiting for Stooq zip "
                f"in {downloads_dir}"
            )

        if _is_download_in_progress(downloads_dir):
            now = time.time()
            if now - last_msg > 10:
                print("  Download in progress...")
                last_msg = now
            time.sleep(POLL_SECONDS)
            continue

        cands = _zip_candidates(downloads_dir)
        # Prefer a zip modified after this wait started (fresh download).
        fresh = [
            p
            for p in cands
            if p.stat().st_mtime >= start - 5  # small slack for clock skew
        ]
        pick = fresh[0] if fresh else None
        if pick is not None and _file_stable(pick):
            mtime = datetime.fromtimestamp(pick.stat().st_mtime)
            print(
                f"Found download: {pick.name} "
                f"({pick.stat().st_size / 1e6:.1f} MB, mtime={mtime})"
            )
            return pick

        now = time.time()
        if now - last_msg > 15:
            print("  Still waiting for captcha download to finish...")
            last_msg = now
        time.sleep(POLL_SECONDS)


def install_zip(src: Path, dest: Path = ZIP_FILE_PATH) -> Path:
    """Replace dest with src (move when possible; copy+remove otherwise)."""
    src = Path(src).resolve()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if dest.resolve() == src.resolve():
            print(f"Zip already at install path: {dest}")
            return dest
        print(f"Removing old zip: {dest}")
        dest.unlink()

    print(f"Installing zip:\n  from {src}\n  to   {dest}")
    try:
        shutil.move(str(src), str(dest))
    except OSError:
        shutil.copy2(src, dest)
        src.unlink()
    return dest


def clear_daily_price_cache() -> None:
    """Drop cached daily_prices.pkl so later tools don't reuse stale bars."""
    if DAILY_PRICES_CACHE.exists():
        print(f"Removing stale price cache: {DAILY_PRICES_CACHE}")
        DAILY_PRICES_CACHE.unlink()


def run(
    days: int = DEFAULT_DAYS,
    zip_path: Path | None = None,
    skip_download: bool = False,
    sp500_only: bool = True,
) -> None:
    print("=" * 72)
    print("run_trend_down: refresh Stooq data → current bargains")
    print("=" * 72)

    if skip_download:
        if not ZIP_FILE_PATH.is_file():
            raise SystemExit(
                f"--skip-download set but missing {ZIP_FILE_PATH}. "
                "Download d_us_txt.zip first."
            )
        print(f"Reusing existing zip: {ZIP_FILE_PATH}")
    else:
        _prompt_manual_download()
        downloaded = wait_for_downloaded_zip(already_present=zip_path)
        install_zip(downloaded, ZIP_FILE_PATH)

    print("\nStep 2–3: prepare Stooq directories (unzip + reorganize) ...")
    if not prepare_stooq_data(ZIP_FILE_PATH):
        raise SystemExit("prepare_stooq_data() failed.")

    clear_daily_price_cache()

    print(f"\nStep 4: trend-down bargains (--today --days={days}) ...")
    print_current_bargains(days_from_today=days, sp500_only=sp500_only)
    print("\nDone.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download (manual) + prepare Stooq data + list recent bargains"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        metavar="N",
        help=f"Lookback window for --today bargains (default: {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="Path to an already-downloaded d_us_txt.zip (skip Downloads watch)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help=f"Reuse existing {ZIP_FILE_PATH} without waiting for a new download",
    )
    parser.add_argument(
        "--all-tickers",
        action="store_true",
        help="Include all tickers (skip S&P 500 point-in-time filter)",
    )
    args = parser.parse_args(argv)
    try:
        run(
            days=args.days,
            zip_path=args.zip,
            skip_download=args.skip_download,
            sp500_only=not args.all_tickers,
        )
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
