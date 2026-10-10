"""Market data sources for paper trading and backtests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..brokers.base import MarketData
from ..market_hours import IST

OHLCV = ["open", "high", "low", "close", "volume"]


def resample(df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    """Resample to N-minute candles anchored at the 09:15 NSE open."""
    if df.empty:
        return df
    out = df.resample(f"{interval_minutes}min", origin="start_day", offset="15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["open"])


def load_csv(path: str | Path) -> pd.DataFrame:
    """CSV with columns datetime/date, open, high, low, close[, volume]. Naive times are treated as IST."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    ts_col = next((c for c in ("datetime", "date", "timestamp", "time") if c in df.columns), None)
    if ts_col is None:
        raise ValueError("CSV needs a datetime/date/timestamp column")
    idx = pd.to_datetime(df[ts_col])
    idx = idx.dt.tz_localize(IST) if idx.dt.tz is None else idx.dt.tz_convert(IST)
    df = df.set_index(idx).drop(columns=[ts_col])
    if "volume" not in df.columns:
        df["volume"] = 0
    return df[OHLCV].astype(float).sort_index()


class YFinanceFeed(MarketData):
    """Free Yahoo Finance data (NSE symbols get a .NS suffix). Can lag the exchange by a few minutes."""

    def __init__(self, suffix: str = ".NS"):
        try:
            import yfinance  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise SystemExit("yfinance is not installed: pip install yfinance") from e
        self.suffix = suffix

    def _fetch(self, symbol: str, period: str, interval: str = "1m") -> pd.DataFrame:
        import yfinance as yf

        df = yf.Ticker(symbol + self.suffix).history(period=period, interval=interval, auto_adjust=False)
        if df.empty:
            return df
        df.columns = [c.lower() for c in df.columns]
        df.index = df.index.tz_convert(IST)
        return df[OHLCV]

    def ltp(self, symbols: list[str]) -> dict[str, float]:
        out = {}
        for s in symbols:
            df = self._fetch(s, "1d")
            if not df.empty:
                out[s] = float(df["close"].iloc[-1])
        return out

    def candles(self, symbol: str, interval_minutes: int, count: int) -> pd.DataFrame:
        return resample(self._fetch(symbol, "5d"), interval_minutes).tail(count)

    def history(self, symbol: str, interval_minutes: int) -> pd.DataFrame:
        """As much intraday history as Yahoo allows: ~60 days of 5m bars, or 7 days of 1m bars."""
        if interval_minutes % 5 == 0:
            return resample(self._fetch(symbol, "59d", "5m"), interval_minutes)
        return resample(self._fetch(symbol, "7d"), interval_minutes)


class KiteFeed(MarketData):
    """Use Zerodha market data with the paper broker (real-time quotes, simulated orders)."""

    def __init__(self, broker):
        self.broker = broker

    def ltp(self, symbols):
        return self.broker.ltp(symbols)

    def candles(self, symbol, interval_minutes, count):
        return self.broker.candles(symbol, interval_minutes, count)
