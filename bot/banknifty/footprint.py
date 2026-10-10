"""Footprint candles built from the futures tick stream.

Each candle stores, per price row, how much volume traded at the bid (sell aggressor) and at
the ask (buy aggressor). From that we derive delta, CVD, point of control, value area,
diagonal imbalances, stacked imbalances, absorption, book pressure and OI change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..brokers.base import MarketData
from ..config import BankNiftyConfig, SessionConfig
from ..market_hours import IST
from .ticks import Tick, TradeClassifier

FRAME_COLUMNS = [
    "open", "high", "low", "close", "volume", "buy_vol", "sell_vol", "delta", "delta_pct",
    "poc", "vah", "val", "stacked_buy", "stacked_sell", "absorb_bull", "absorb_bear",
    "book_imb", "oi", "oi_chg", "has_flow",
]


@dataclass
class FootprintBar:
    start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    buy_vol: float = 0.0
    sell_vol: float = 0.0
    levels: dict[float, list[float]] = field(default_factory=dict)  # row price -> [sell_vol, buy_vol]
    book_imb_sum: float = 0.0
    book_n: int = 0
    oi_open: int = 0
    oi_close: int = 0
    has_flow: bool = False

    def add_trade(self, price: float, qty: int, side: int, row: float) -> None:
        self.volume += qty
        if side == 0:
            return
        self.has_flow = True
        lvl = self.levels.setdefault(round(math.floor(price / row) * row, 6), [0.0, 0.0])
        if side > 0:
            self.buy_vol += qty
            lvl[1] += qty
        else:
            self.sell_vol += qty
            lvl[0] += qty


def value_area(levels: dict[float, list[float]], share: float = 0.7) -> tuple[float, float, float]:
    """(poc, vah, val) of a footprint. Expands from the POC toward the busier neighbour."""
    if not levels:
        return (math.nan,) * 3
    prices = sorted(levels)
    vols = [sum(levels[p]) for p in prices]
    i = int(np.argmax(vols))
    lo = hi = i
    total, acc = sum(vols), vols[i]
    while acc < share * total and (lo > 0 or hi < len(prices) - 1):
        up = vols[hi + 1] if hi < len(prices) - 1 else -1
        down = vols[lo - 1] if lo > 0 else -1
        if up >= down:
            hi += 1
            acc += up
        else:
            lo -= 1
            acc += down
    return prices[i], prices[hi], prices[lo]


def stacked_imbalances(levels: dict[float, list[float]], row: float, ratio: float, min_qty: float) -> tuple[int, int]:
    """Longest run of consecutive buy and sell diagonal imbalances.

    Buy imbalance at P: ask volume at P >= ratio x bid volume at P - row.
    Sell imbalance at P: bid volume at P >= ratio x ask volume at P + row.
    """
    if not levels:
        return 0, 0
    lo, hi = min(levels), max(levels)
    n = int(round((hi - lo) / row)) + 1
    prices = [round(lo + k * row, 6) for k in range(n)]
    get = lambda p: levels.get(p) or levels.get(round(p, 6)) or [0.0, 0.0]  # noqa: E731

    def longest(flags):
        best = run = 0
        for f in flags:
            run = run + 1 if f else 0
            best = max(best, run)
        return best

    buy = [get(p)[1] >= min_qty and get(p)[1] >= ratio * get(p - row)[0] for p in prices]
    sell = [get(p)[0] >= min_qty and get(p)[0] >= ratio * get(p + row)[1] for p in prices]
    return longest(buy), longest(sell)


def absorption(bar: FootprintBar, share: float) -> tuple[bool, bool]:
    """Bullish: heavy selling in the bottom quarter of the bar, yet it closed in the upper half.
    Bearish: heavy buying in the top quarter, yet it closed in the lower half."""
    rng = bar.high - bar.low
    flow = bar.buy_vol + bar.sell_vol
    if rng <= 0 or flow <= 0:
        return False, False
    bottom = sum(v[0] for p, v in bar.levels.items() if p <= bar.low + 0.25 * rng)
    top = sum(v[1] for p, v in bar.levels.items() if p >= bar.high - 0.25 * rng)
    mid = bar.low + rng / 2
    return bottom / flow >= share and bar.close >= mid, top / flow >= share and bar.close <= mid


class FootprintAggregator(MarketData):
    """Consumes ticks for all subscribed instruments; builds footprint candles for the signal symbol."""

    def __init__(self, signal_symbol: str, interval_minutes: int, session: SessionConfig, bn: BankNiftyConfig):
        self.signal_symbol = signal_symbol
        self.interval = timedelta(minutes=interval_minutes)
        self.session, self.bn = session, bn
        self.classifier = TradeClassifier()
        self.bars: list[FootprintBar] = []
        self.history = pd.DataFrame(columns=FRAME_COLUMNS)  # seeded OHLCV without order flow
        self.last_prices: dict[str, float] = {}
        self.last_tick_at: datetime | None = None
        self._frame_cache: tuple[int, int, pd.DataFrame] | None = None
        self._ticks = 0

    # ------------------------------------------------------------------ ingest
    def bar_start(self, ts: datetime) -> datetime:
        ts = ts.astimezone(IST)
        open_dt = ts.replace(hour=self.session.market_open.hour, minute=self.session.market_open.minute, second=0, microsecond=0)
        n = max(int((ts - open_dt) // self.interval), 0)
        return open_dt + n * self.interval

    def on_tick(self, t: Tick) -> None:
        self.last_prices[t.symbol] = t.ltp
        if t.symbol != self.signal_symbol:
            return
        self._ticks += 1
        self.last_tick_at = t.ts
        qty, side = self.classifier.classify(t)
        start = self.bar_start(t.ts)
        bar = self.bars[-1] if self.bars and self.bars[-1].start == start else None
        if bar is None:
            if t.ts.astimezone(IST).time() < self.session.market_open:
                return
            bar = FootprintBar(start, t.ltp, t.ltp, t.ltp, t.ltp, oi_open=t.oi, oi_close=t.oi)
            self.bars.append(bar)
            if len(self.bars) > 2000:
                del self.bars[:500]
        bar.high, bar.low, bar.close = max(bar.high, t.ltp), min(bar.low, t.ltp), t.ltp
        if t.oi:
            bar.oi_open = bar.oi_open or t.oi
            bar.oi_close = t.oi
        if qty:
            bar.add_trade(t.ltp, qty, side, self.bn.footprint_row_size)
        depth = t.bid_qty + t.ask_qty
        if depth > 0:
            bar.book_imb_sum += (t.bid_qty - t.ask_qty) / depth
            bar.book_n += 1

    def seed(self, df: pd.DataFrame) -> None:
        """Pre-load plain OHLCV history (e.g. from Dhan's chart API) so indicators are warm at startup."""
        if df is None or df.empty:
            return
        h = df[["open", "high", "low", "close", "volume"]].copy()
        if self.bars:
            h = h[h.index < self.bars[0].start]
        for c in FRAME_COLUMNS:
            if c not in h.columns:
                h[c] = False if c == "has_flow" else math.nan
        self.history = h[FRAME_COLUMNS]
        self._frame_cache = None

    # ------------------------------------------------------------------ MarketData
    def ltp(self, symbols: list[str]) -> dict[str, float]:
        return {s: self.last_prices[s] for s in symbols if s in self.last_prices}

    def candles(self, symbol: str, interval_minutes: int, count: int) -> pd.DataFrame:
        if symbol != self.signal_symbol:
            raise KeyError(f"No footprint candles for {symbol}")
        return self.frame().tail(count)

    def feed_age_seconds(self, now: datetime) -> float:
        return math.inf if self.last_tick_at is None else (now - self.last_tick_at).total_seconds()

    def frame(self) -> pd.DataFrame:
        key = (len(self.bars), self._ticks)
        if self._frame_cache and self._frame_cache[:2] == key:
            return self._frame_cache[2]
        bn, rows = self.bn, []
        for b in self.bars:
            poc, vah, val = value_area(b.levels)
            sb, ss = stacked_imbalances(b.levels, bn.footprint_row_size, bn.imbalance_ratio, bn.imbalance_min_qty)
            ab, ar = absorption(b, bn.absorption_share)
            flow = b.buy_vol + b.sell_vol
            rows.append({
                "open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume,
                "buy_vol": b.buy_vol, "sell_vol": b.sell_vol, "delta": b.buy_vol - b.sell_vol,
                "delta_pct": (b.buy_vol - b.sell_vol) / flow if flow else math.nan,
                "poc": poc, "vah": vah, "val": val, "stacked_buy": sb, "stacked_sell": ss,
                "absorb_bull": ab, "absorb_bear": ar,
                "book_imb": b.book_imb_sum / b.book_n if b.book_n else math.nan,
                "oi": b.oi_close or math.nan, "oi_chg": (b.oi_close - b.oi_open) if b.oi_open else math.nan,
                "has_flow": b.has_flow,
            })
        live = pd.DataFrame(rows, columns=FRAME_COLUMNS, index=pd.DatetimeIndex([b.start for b in self.bars]))
        parts = [p for p in (self.history, live) if not p.empty]
        out = pd.concat(parts) if len(parts) > 1 else (parts[0] if parts else live)
        self._frame_cache = (*key, out)
        return out
