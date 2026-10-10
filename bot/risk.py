"""Position sizing, daily limits, kill switch, order-rate limiting and transaction costs."""

from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass

from .config import CostConfig, RiskConfig


@dataclass
class DayState:
    realized_pnl: float = 0.0
    trades_opened: int = 0
    halted: bool = False
    halt_reason: str = ""


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.day = DayState()

    def new_day(self) -> None:
        self.day = DayState()

    @property
    def max_daily_loss(self) -> float:
        return self.cfg.capital * self.cfg.max_daily_loss_pct / 100

    def position_size(self, price: float, stop_distance: float, available_cash: float | None = None) -> int:
        """Shares to buy so that hitting the stop loses ~risk_per_trade_pct of capital.

        Deliberately ignores intraday leverage: position value is capped by capital and cash.
        """
        if price <= 0 or stop_distance <= 0:
            return 0
        risk_amount = self.cfg.capital * self.cfg.risk_per_trade_pct / 100
        qty = math.floor(risk_amount / stop_distance)
        max_value = self.cfg.capital * self.cfg.max_position_value_pct / 100
        if available_cash is not None:
            max_value = min(max_value, available_cash)
        qty = min(qty, math.floor(max_value / price))
        return max(qty, 0)

    def can_open(self, open_positions: int) -> tuple[bool, str]:
        if self.day.halted:
            return False, f"halted: {self.day.halt_reason}"
        if open_positions >= self.cfg.max_open_positions:
            return False, "max open positions reached"
        if self.day.trades_opened >= self.cfg.max_trades_per_day:
            return False, "max trades per day reached"
        return True, ""

    def on_trade_opened(self) -> None:
        self.day.trades_opened += 1

    def on_trade_closed(self, pnl: float) -> None:
        self.day.realized_pnl += pnl

    def check_loss_limit(self, unrealized_pnl: float = 0.0) -> bool:
        """Returns True (and halts) if today's realized+unrealized loss breached the limit."""
        total = self.day.realized_pnl + unrealized_pnl
        if not self.day.halted and total <= -self.max_daily_loss:
            self.halt(f"daily loss limit hit (P&L {total:.2f} <= -{self.max_daily_loss:.2f})")
        return self.day.halted

    def halt(self, reason: str) -> None:
        self.day.halted = True
        self.day.halt_reason = reason


class RateLimiter:
    """Spaces calls at least 1/rate seconds apart."""

    def __init__(self, per_second: float, clock=_time.monotonic, sleep=_time.sleep):
        self.min_gap = 1.0 / per_second
        self.clock, self.sleep = clock, sleep
        self._last = -math.inf

    def wait(self) -> None:
        gap = self.clock() - self._last
        if gap < self.min_gap:
            self.sleep(self.min_gap - gap)
        self._last = self.clock()


def trade_costs(buy_value: float, sell_value: float, c: CostConfig) -> float:
    """Approximate total charges (brokerage, STT, exchange, SEBI, stamp duty, GST) for an intraday round trip."""
    brokerage = sum(min(v * c.brokerage_pct / 100, c.brokerage_cap) for v in (buy_value, sell_value))
    stt = sell_value * c.stt_sell_pct / 100
    exchange = (buy_value + sell_value) * c.exchange_txn_pct / 100
    sebi = (buy_value + sell_value) * c.sebi_pct / 100
    stamp = buy_value * c.stamp_buy_pct / 100
    gst = (brokerage + exchange + sebi) * c.gst_pct / 100
    return brokerage + stt + exchange + sebi + stamp + gst


def round_to_tick(price: float, tick: float) -> float:
    return round(round(price / tick) * tick, 2)
