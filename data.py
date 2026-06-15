"""Data layer: yfinance fetch with CSV cache + Robinhood-snapshot import.

The backtest is faithful to what's in the Robinhood account because Robinhood's
daily series were verified to match yfinance split-adjusted OHLCV. yfinance is
used as the primary source so the engine is re-runnable offline by the user and
can pull the underlying indices needed for the decay decomposition.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

_COLS = ["open", "high", "low", "close", "volume"]


def _cache_path(ticker: str) -> Path:
    return DATA_DIR / f"{ticker.upper()}.csv"


def _fetch_yf(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        ticker, start=start, end=end, auto_adjust=True,
        progress=False, threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):     # single ticker still nests sometimes
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df.dropna()


def load_one(ticker: str, start: str, end: str | None,
             refresh: bool = False) -> pd.DataFrame:
    """Load a single symbol's daily OHLCV, using the CSV cache when possible."""
    path = _cache_path(ticker)
    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col="date", parse_dates=["date"])
        df = df[_COLS]
        # extend the cache forward if the requested window runs past it
        want_start = pd.Timestamp(start)
        if df.index.min() <= want_start:
            return df.loc[start:end]
    df = _fetch_yf(ticker, start, end)
    df.to_csv(path)
    return df.loc[start:end]


def load_prices(tickers: list[str], start: str, end: str | None,
                refresh: bool = False) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for t in dict.fromkeys(tickers):           # dedupe, preserve order
        try:
            out[t] = load_one(t, start, end, refresh)
        except Exception as exc:               # noqa: BLE001 - surface, don't swallow
            errors[t] = str(exc)
    if errors:
        msg = "; ".join(f"{k}: {v}" for k, v in errors.items())
        print(f"[data] WARNING failed to load: {msg}")
    if not out:
        raise RuntimeError("No price data loaded for any ticker.")
    return out


def risk_free_series(start: str, end: str | None, fallback: float = 0.045) -> pd.Series:
    """Daily-aligned annualized risk-free rate from ^IRX (13-wk T-bill).

    Returns an annualized rate (e.g. 0.045) indexed by date; falls back to a
    flat constant if the series cannot be fetched.
    """
    try:
        irx = load_one("^IRX", start, end, refresh=False)
        rf = (irx["close"] / 100.0).clip(lower=0.0)
        rf.name = "rf"
        return rf
    except Exception as exc:                   # noqa: BLE001
        print(f"[data] WARNING ^IRX unavailable ({exc}); using flat rf={fallback}")
        idx = pd.bdate_range(start=start, end=end or pd.Timestamp.today())
        return pd.Series(fallback, index=idx, name="rf")


def import_robinhood_csv(path: str | os.PathLike) -> pd.DataFrame:
    """Load a Robinhood-exported OHLCV CSV (date,open,high,low,close,volume)."""
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    missing = [c for c in _COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Robinhood CSV missing columns: {missing}")
    return df[_COLS].sort_index()


def daily_returns(df: pd.DataFrame) -> pd.Series:
    """Simple daily close-to-close returns."""
    return df["close"].pct_change().rename("ret")


def align(*series: pd.Series) -> tuple[pd.Series, ...]:
    """Inner-join several series on common dates (name-collision safe)."""
    named = [s.rename(i) for i, s in enumerate(series)]   # avoid duplicate labels
    frame = pd.concat(named, axis=1).dropna()
    return tuple(frame[i] for i in range(len(series)))
