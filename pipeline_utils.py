"""Shared helper functions for the Bundesliga data pipeline.

Input: values from the ESPN and WhoScored source loaders.
Output: cleaned values, common constants, and small reusable helpers.
"""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
from datetime import date, datetime
from typing import Any

import pandas as pd


TARGET_SEASON = "2425"
DEFAULT_LEAGUE = "GER-Bundesliga"
RQ4_MIN_MATCHES_FOR_LEADERBOARD = 5
RQ4_MIN_MATCHES_PER_SIDE_FOR_DELTA = 5
ESPN_ATHLETE_URL = (
    "https://site.web.api.espn.com/apis/common/v3/sports/"
    "soccer/athletes/{athlete_id}"
)


def ensure_soccerdata() -> Any:
    """Import `soccerdata` and install it once if it is missing.

    Input: no direct input.
    Output: imported `soccerdata` module.
    """
    os.environ.setdefault("SOCCERDATA_LOGLEVEL", "WARNING")

    try:
        module = importlib.import_module("soccerdata")
    except ImportError:
        if sys.version_info >= (3, 14):
            pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
            raise RuntimeError(
                "soccerdata is currently incompatible with Python "
                f"{pyver}. Use Python 3.12 or 3.13."
            ) from None

        print(
            "[setup] Package 'soccerdata' is missing. "
            "Starting installation ..."
        )
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "soccerdata"]
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Automatic installation of 'soccerdata' failed."
            ) from exc

        module = importlib.import_module("soccerdata")

    for logger_name in ("TLSRequests", "TLSLibrary"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False

    return module


def normalize_season_label(season_code: str) -> str:
    """Convert a short season code into a readable season label.

    Input: string season code such as `"2425"`.
    Output: string label such as `"2024/2025"`.
    """
    code = str(season_code).strip()
    if len(code) == 4 and code.isdigit():
        return f"20{code[:2]}/20{code[2:]}"
    return code


def season_reference_date(season_code: str) -> date:
    """Return the reference date used for age calculations.

    Input: string season code such as `"2425"`.
    Output: date object, usually June 30 of the season end year.
    """
    code = str(season_code).strip()
    if len(code) != 4 or not code.isdigit():
        return date.today()
    return date(int(f"20{code[2:]}"), 6, 30)


def parse_espn_display_dob(value: object) -> date | None:
    """Parse an ESPN date-of-birth value into a Python date.

    Input: raw ESPN date value.
    Output: parsed date or `None`.
    """
    if not value:
        return None

    text = str(value).strip()
    if not text:
        return None

    for fmt in ("%Y-%m-%dT%H:%MZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    return None


def age_years_at_reference_date(
    birth_date: date,
    reference_date: date,
) -> float:
    """Return the age in years at a chosen reference date.

    Input: birth date and reference date.
    Output: float age in years.
    """
    return (reference_date - birth_date).days / 365.25


def should_log_progress(current: int, total: int, interval: int) -> bool:
    """Decide whether a progress update should be printed.

    Input: current step, total steps, and logging interval.
    Output: `True` when a progress message should be shown.
    """
    if total <= 0:
        return True
    return current == 1 or current == total or current % max(1, interval) == 0


def format_progress(label: str, current: int, total: int) -> str:
    """Build a simple text progress bar.

    Input: progress label, current count, and total count.
    Output: formatted progress string.
    """
    if total <= 0:
        return f"[progress] {label}: [{'#' * 24}] 0/0 (100.0%, remaining=0)"

    safe_current = max(0, min(int(current), int(total)))
    percent = (safe_current / total) * 100
    remaining = max(0, int(total) - safe_current)
    bar_width = 24
    filled = int(round((safe_current / total) * bar_width))
    bar = "#" * filled + "-" * (bar_width - filled)
    return (
        f"[progress] {label}: [{bar}] {safe_current}/{total} "
        f"({percent:.1f}%, remaining={remaining})"
    )


def parse_int(value: object) -> int | None:
    """Convert a loose source value into an integer when possible.

    Input: raw value from JSON, CSV, or pandas objects.
    Output: integer value or `None`.
    """
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float(value: object) -> float | None:
    """Convert a loose source value into a float when possible.

    Input: raw value from JSON, CSV, or pandas objects.
    Output: float value or `None`.
    """
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def to_bool(value: object) -> bool:
    """Convert loose truthy values into a Python boolean.

    Input: raw boolean-like value.
    Output: `True` or `False`.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def fix_mojibake(text: str) -> str:
    """Repair common UTF-8 mojibake in player and team names.

    Input: source text value.
    Output: cleaned text value.
    """
    if not text:
        return text

    if not any(ch in text for ch in ("\u00c3", "\u00e2", "\u20ac", "\u2122")):
        return text

    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def round_numeric_columns(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    digits: int,
) -> pd.DataFrame:
    """Round selected numeric columns for display output.

    Input: DataFrame, column names, and number of digits.
    Output: copied DataFrame with rounded columns.
    """
    out = df.copy()
    for column in columns:
        if column in out.columns:
            numeric = pd.to_numeric(out[column], errors="coerce")
            out[column] = numeric.round(digits)
    return out


def fit_correlation(x: pd.Series, y: pd.Series) -> float:
    """Return the Pearson correlation for two numeric Series.

    Input: two pandas Series.
    Output: float correlation or `nan`.
    """
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 2:
        return float("nan")
    return float(valid["x"].corr(valid["y"]))
