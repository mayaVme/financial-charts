"""Simulated broker: real prices from a MarketData feed, fake money."""

from __future__ import annotations

import itertools
import logging
import time

import pandas as pd

from .base import Broker, MarketData, Order, OrderStatus, OrderType, Side

log = logging.getLogger(__name__)


class PaperBroker(Broker):
    def __init__(self, feed: MarketData, starting_cash: float, slippage_pct: float = 0.05, ltp_cache_seconds: float = 2.0):
        self.feed = feed
        self.cash = starting_cash
        self.slippage = slippage_pct / 100
        self.positions: dict[str, int] = {}
        self.orders: dict[str, Order] = {}
        self._ids = itertools.count(1)
        self._ltp_cache: dict[str, tuple[float, float]] = {}
        self._cache_s = ltp_cache_seconds

    # --- market data passthrough -------------------------------------------------
    def ltp(self, symbols: list[str]) -> dict[str, float]:
        now = time.monotonic()
        stale = [s for s in symbols if s not in self._ltp_cache or now - self._ltp_cache[s][0] > self._cache_s]
        if stale:
            for s, p in self.feed.ltp(stale).items():
                self._ltp_cache[s] = (now, p)
        return {s: self._ltp_cache[s][1] for s in symbols if s in self._ltp_cache}

    def candles(self, symbol: str, interval_minutes: int, count: int) -> pd.DataFrame:
        return self.feed.candles(symbol, interval_minutes, count)

    # --- orders -----------------------------------------------------------------
    def place_order(self, symbol, side, qty, order_type, price, trigger_price=None) -> str:
        if qty <= 0:
            raise ValueError("qty must be positive")
        oid = f"PAPER-{next(self._ids)}"
        status = OrderStatus.TRIGGER_PENDING if order_type is OrderType.SL else OrderStatus.OPEN
        self.orders[oid] = Order(oid, symbol, side, qty, order_type, price, trigger_price, status)
        self._process(oid)
        return oid

    def get_order(self, order_id: str) -> Order:
        self._process(order_id)
        return self.orders[order_id]

    def cancel_order(self, order_id: str) -> None:
        o = self.orders[order_id]
        self._process(order_id)
        if not o.status.is_final:
            o.status = OrderStatus.CANCELLED

    def available_cash(self) -> float:
        return self.cash

    def net_positions(self) -> dict[str, int]:
        return {s: q for s, q in self.positions.items() if q}

    def _process(self, order_id: str) -> None:
        o = self.orders[order_id]
        if o.status.is_final:
            return
        prices = self.ltp([o.symbol])
        if o.symbol not in prices:
            return
        px = prices[o.symbol]
        if o.status is OrderStatus.TRIGGER_PENDING:
            triggered = px <= o.trigger_price if o.side is Side.SELL else px >= o.trigger_price
            if not triggered:
                return
            o.status = OrderStatus.OPEN
        marketable = px <= o.price if o.side is Side.BUY else px >= o.price
        if not marketable:
            return
        fill = px * (1 + self.slippage) if o.side is Side.BUY else px * (1 - self.slippage)
        fill = min(fill, o.price) if o.side is Side.BUY else max(fill, o.price)
        o.status, o.filled_qty, o.avg_price = OrderStatus.COMPLETE, o.qty, round(fill, 2)
        sign = 1 if o.side is Side.BUY else -1
        self.positions[o.symbol] = self.positions.get(o.symbol, 0) + sign * o.qty
        self.cash -= sign * o.qty * o.avg_price
        log.info("[paper] filled %s %s %d @ %.2f (%s)", o.side.value, o.symbol, o.qty, o.avg_price, o.order_id)
