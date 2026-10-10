"""Technical indicators on pandas Series/DataFrames (OHLCV with lowercase columns)."""

from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    # No losses at all -> RSI 100; flat -> 50.
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return out


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP that resets every trading day (typical price weighted by volume)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.date
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vol = df["volume"].groupby(day).cumsum()
    return (pv / vol.where(vol > 0)).astype(float)


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.Series:
    """Supertrend direction: +1 (up) / -1 (down)."""
    a = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper, lower = (hl2 + mult * a).to_numpy(), (hl2 - mult * a).to_numpy()
    close = df["close"].to_numpy()
    direction = [1] * len(df)
    fu, fl = upper.copy(), lower.copy()
    for i in range(1, len(df)):
        if pd.isna(a.iloc[i]):
            continue
        if pd.isna(fu[i - 1]):  # first bar after the ATR warm-up
            continue
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        if close[i] > fu[i - 1]:
            direction[i] = 1
        elif close[i] < fl[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    return pd.Series(direction, index=df.index)
