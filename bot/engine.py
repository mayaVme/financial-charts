"""Live/paper trading loop.

Every entry is immediately protected by a broker-side stop order, so a crash or network loss
never leaves a position unprotected. Targets, signal exits, the daily-loss kill switch and the
end-of-day square-off are handled by the loop.
"""

from __future__ import annotations

import csv
import logging
import time as _time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .brokers.base import Broker, MarketData, OrderStatus, OrderType, Side
from .config import Config
from .market_hours import IST, MarketCalendar, last_closed_candle_end, now_ist
from .risk import RateLimiter, RiskManager, round_to_tick, trade_costs
from .strategies import Action, Signal, Strategy

log = logging.getLogger(__name__)

CANDLE_SETTLE_SECONDS = 5


@dataclass
class Position:
    symbol: str
    direction: int  # +1 long, -1 short
    qty: int
    entry_price: float
    stop_price: float
    target_price: float
    stop_order_id: str
    opened_at: datetime

    @property
    def exit_side(self) -> Side:
        return Side.SELL if self.direction > 0 else Side.BUY

    def pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.qty * self.direction


class TradingEngine:
    def __init__(self, cfg: Config, broker: Broker, strategy: Strategy, clock=now_ist, sleep=_time.sleep,
                 market_data: MarketData | None = None):
        self.cfg, self.broker, self.strategy = cfg, broker, strategy
        # Prices and candles can come from somewhere other than the order-routing broker (e.g. a tick feed).
        self.market_data = market_data or broker
        self.costs = cfg.costs
        self.clock, self.sleep = clock, sleep
        self.calendar = MarketCalendar(cfg.session)
        self.risk = RiskManager(cfg.risk)
        self.limiter = RateLimiter(cfg.risk.max_orders_per_second, sleep=sleep)
        self.positions: dict[str, Position] = {}
        self.blocked: set[str] = set()
        self._day = None
        self._last_candle_end: datetime | None = None
        self._squared_off = False
        self._reported = False
        self.closed_trades: list[dict] = []
        self.journal = Path(cfg.log_dir) / "trades.csv"
        self.journal.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ main loop
    def run(self, max_steps: int | None = None) -> None:
        log.info("Starting engine in %s mode for %s", self.cfg.mode.upper(), ", ".join(self.cfg.symbols))
        self.reconcile()
        steps = 0
        while max_steps is None or steps < max_steps:
            steps += 1
            now = self.clock()
            if not self.calendar.is_open(now):
                if self._day == now.date() and not self._reported:
                    self.square_off_all("market closed")
                    self._end_of_day_report()
                    self._reported = True
                wake = self.calendar.next_open(now)
                log.info("Market closed. Sleeping until %s", wake.strftime("%a %d %b %H:%M IST"))
                self.sleep(max((wake - now).total_seconds(), 1))
                continue
            try:
                self.step(now)
            except Exception:  # keep running; stops are already at the broker
                log.exception("Error during step")
            self.sleep(self.cfg.poll_seconds)

    def reconcile(self) -> None:
        """Refuse to trade symbols that already have positions the bot doesn't know about."""
        try:
            existing = self.broker.net_positions()
        except Exception:
            log.exception("Could not fetch broker positions; continuing")
            return
        for sym, qty in existing.items():
            if sym in self.cfg.symbols and sym not in self.positions:
                self.blocked.add(sym)
                log.warning("%s already has an open position (%d) not opened by this run; skipping it today", sym, qty)

    def step(self, now: datetime) -> None:
        if self._day != now.date():
            self._new_day(now)

        self._sync_stop_orders()
        prices = self.market_data.ltp(list(self.positions)) if self.positions else {}
        self._check_targets(prices)

        unrealized = sum(p.pnl(prices[s]) for s, p in self.positions.items() if s in prices)
        if self.risk.check_loss_limit(unrealized) and self.positions:
            log.error("KILL SWITCH: %s", self.risk.day.halt_reason)
            self.square_off_all("kill switch")

        if self.calendar.must_square_off(now):
            if not self._squared_off:
                self.square_off_all("end-of-day square-off")
                self._squared_off = True
            return

        candle_end = last_closed_candle_end(now, self.cfg.interval_minutes, self.cfg.session)
        if candle_end == self._last_candle_end or now < candle_end + timedelta(seconds=CANDLE_SETTLE_SECONDS):
            return
        self._last_candle_end = candle_end
        for sym in self.cfg.symbols:
            if sym in self.blocked:
                continue
            try:
                self._evaluate(sym, now)
            except Exception:
                log.exception("Failed to evaluate %s", sym)

    # ------------------------------------------------------------------ signals
    def _evaluate(self, sym: str, now: datetime) -> None:
        df = self.market_data.candles(sym, self.cfg.interval_minutes, self.cfg.lookback_candles + 1)
        if df is None or df.empty:
            log.warning("No candles for %s", sym)
            return
        df = df[df.index + timedelta(minutes=self.cfg.interval_minutes) <= now]  # closed candles only
        sig = self.strategy.on_candles(df)
        if sig.action is not Action.HOLD:
            log.info("%s signal %s (%s)", sym, sig.action.value, sig.reason)
        self.handle_signal(sym, sig, float(df["close"].iloc[-1]), now)

    def handle_signal(self, sym: str, sig: Signal, last_close: float, now: datetime) -> None:
        pos = self.positions.get(sym)
        a = sig.action
        if pos and ((a in (Action.EXIT_LONG, Action.SHORT) and pos.direction > 0) or (a in (Action.EXIT_SHORT, Action.BUY) and pos.direction < 0)):
            self.close_position(sym, f"signal {a.value}")
            pos = None
        if a in (Action.BUY, Action.SHORT) and pos is None:
            self.open_position(sym, +1 if a is Action.BUY else -1, sig, last_close, now)

    # ------------------------------------------------------------------ orders
    def _limit_price(self, ref: float, side: Side, buffer_pct: float) -> float:
        mult = 1 + buffer_pct / 100 if side is Side.BUY else 1 - buffer_pct / 100
        return round_to_tick(ref * mult, self.cfg.execution.tick_size)

    def _place(self, *args, **kw) -> str:
        self.limiter.wait()
        return self.broker.place_order(*args, **kw)

    def _wait_fill(self, order_id: str) -> tuple[int, float, OrderStatus]:
        for waited in range(self.cfg.execution.fill_timeout_seconds + 1):
            o = self.broker.get_order(order_id)
            if o.status.is_final or waited == self.cfg.execution.fill_timeout_seconds:
                break
            self.sleep(1)
        if not o.status.is_final:
            try:
                self.broker.cancel_order(order_id)
            except Exception:
                log.exception("Cancel failed for %s", order_id)
            o = self.broker.get_order(order_id)
        if o.status is OrderStatus.REJECTED:
            log.error("Order %s rejected: %s", order_id, o.message)
        return o.filled_qty, o.avg_price, o.status

    def open_position(self, sym: str, direction: int, sig: Signal, last_close: float, now: datetime) -> None:
        if not self.calendar.can_enter(now):
            log.info("%s: outside entry window, ignoring entry", sym)
            return
        ok, why = self.risk.can_open(len(self.positions))
        if not ok:
            log.info("%s: entry blocked (%s)", sym, why)
            return
        price = self.market_data.ltp([sym]).get(sym, last_close)
        qty = self.size_position(sym, price, sig.stop_distance)
        if qty <= 0:
            log.info("%s: position size is 0 (price %.2f, stop %.2f)", sym, price, sig.stop_distance)
            return

        ex = self.cfg.execution
        side = Side.BUY if direction > 0 else Side.SELL
        oid = self._place(sym, side, qty, OrderType.LIMIT, self._limit_price(price, side, ex.limit_buffer_pct))
        filled, avg, _ = self._wait_fill(oid)
        if filled <= 0:
            log.warning("%s: entry not filled", sym)
            return
        self.risk.on_trade_opened()

        stop = round_to_tick(avg - direction * sig.stop_distance, ex.tick_size)
        target = round_to_tick(avg + direction * sig.target_distance, ex.tick_size)
        exit_side = side.opposite
        try:
            stop_id = self._place(sym, exit_side, filled, OrderType.SL, self._limit_price(stop, exit_side, ex.stop_limit_buffer_pct), trigger_price=stop)
        except Exception:
            log.exception("%s: could not place protective stop; exiting immediately", sym)
            self._market_exit(sym, exit_side, filled, avg)
            return

        self.positions[sym] = Position(sym, direction, filled, avg, stop, target, stop_id, now)
        log.info("OPEN %s %s x%d @ %.2f | stop %.2f | target %.2f | %s", "LONG" if direction > 0 else "SHORT", sym, filled, avg, stop, target, sig.reason)

    def size_position(self, sym: str, price: float, stop_distance: float) -> int:
        return self.risk.position_size(price, stop_distance, self.broker.available_cash())

    def _market_exit(self, sym: str, side: Side, qty: int, ref_price: float) -> tuple[int, float]:
        """Marketable-limit exit with widening buffer; returns (filled_qty, avg_price)."""
        remaining, value = qty, 0.0
        for attempt in range(4):
            px = self.market_data.ltp([sym]).get(sym, ref_price)
            buffer = self.cfg.execution.limit_buffer_pct * (2**attempt)
            oid = self._place(sym, side, remaining, OrderType.LIMIT, self._limit_price(px, side, buffer))
            filled, avg, _ = self._wait_fill(oid)
            value += filled * avg
            remaining -= filled
            if remaining == 0:
                return qty, value / qty
        log.critical("%s: could not fully exit (%d of %d left). MANUAL ACTION REQUIRED.", sym, remaining, qty)
        self.risk.halt(f"failed to exit {sym}")
        done = qty - remaining
        return done, (value / done if done else 0.0)

    def close_position(self, sym: str, reason: str) -> None:
        pos = self.positions.get(sym)
        if not pos:
            return
        try:
            self.broker.cancel_order(pos.stop_order_id)
        except Exception:
            log.debug("cancel of stop %s failed (may already be done)", pos.stop_order_id)
        stop = self.broker.get_order(pos.stop_order_id)
        if stop.status is OrderStatus.COMPLETE:
            self._record_close(pos, stop.avg_price, "stop loss")
            return
        if stop.filled_qty:  # partially filled stop before cancel
            self._record_close(pos, stop.avg_price, "stop loss (partial)", qty=stop.filled_qty)
            pos.qty -= stop.filled_qty
        filled, avg = self._market_exit(sym, pos.exit_side, pos.qty, pos.entry_price)
        if filled:
            self._record_close(pos, avg, reason, qty=filled)
        if filled < pos.qty:
            pos.qty -= filled  # keep tracking what's left; engine is halted
            self.positions[sym] = pos

    def square_off_all(self, reason: str) -> None:
        for sym in list(self.positions):
            self.close_position(sym, reason)

    def _sync_stop_orders(self) -> None:
        for sym, pos in list(self.positions.items()):
            o = self.broker.get_order(pos.stop_order_id)
            if o.status is OrderStatus.COMPLETE:
                self._record_close(pos, o.avg_price, "stop loss")
            elif o.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                log.error("%s: protective stop %s is %s (%s); exiting", sym, o.order_id, o.status.value, o.message)
                self.close_position(sym, "stop order lost")

    def _check_targets(self, prices: dict[str, float]) -> None:
        for sym, pos in list(self.positions.items()):
            px = prices.get(sym)
            if px is not None and (px - pos.target_price) * pos.direction >= 0:
                self.close_position(sym, "target")

    # ------------------------------------------------------------------ bookkeeping
    def _record_close(self, pos: Position, exit_price: float, reason: str, qty: int | None = None) -> None:
        qty = qty or pos.qty
        gross = (exit_price - pos.entry_price) * qty * pos.direction
        buy_px, sell_px = (pos.entry_price, exit_price) if pos.direction > 0 else (exit_price, pos.entry_price)
        costs = trade_costs(buy_px * qty, sell_px * qty, self.costs)
        net = gross - costs
        self.risk.on_trade_closed(net)
        if qty >= pos.qty:
            self.positions.pop(pos.symbol, None)
        log.info("CLOSE %s x%d @ %.2f (%s) | net P&L %.2f | day %.2f", pos.symbol, qty, exit_price, reason, net, self.risk.day.realized_pnl)
        self.closed_trades.append(dict(symbol=pos.symbol, direction=pos.direction, qty=qty, entry_time=pos.opened_at, entry=pos.entry_price,
                                       exit_time=self.clock(), exit=exit_price, reason=reason, gross=gross, costs=costs, net=net))
        self.on_position_closed(pos, net, reason)
        new = not self.journal.exists()
        with self.journal.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["mode", "symbol", "direction", "qty", "entry_time", "entry", "exit_time", "exit", "reason", "gross", "costs", "net"])
            w.writerow([self.cfg.mode, pos.symbol, "LONG" if pos.direction > 0 else "SHORT", qty, pos.opened_at.isoformat(), pos.entry_price,
                        self.clock().isoformat(), exit_price, reason, round(gross, 2), round(costs, 2), round(net, 2)])

    def on_position_closed(self, pos: Position, net: float, reason: str) -> None:
        """Hook for subclasses."""

    def _new_day(self, now: datetime) -> None:
        self._day = now.date()
        self._squared_off = False
        self._reported = False
        self._last_candle_end = None
        self.risk.new_day()
        log.info("=== New trading day %s ===", now.astimezone(IST).date())

    def _end_of_day_report(self) -> None:
        d = self.risk.day
        log.info("=== Day summary: trades %d | realized net P&L %.2f%s ===", d.trades_opened, d.realized_pnl, f" | HALTED: {d.halt_reason}" if d.halted else "")
