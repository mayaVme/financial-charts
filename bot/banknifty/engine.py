"""BankNifty intraday engine: footprint signals on the future, executed in futures or ATM options.

Reuses the core TradingEngine for order handling (protective broker-side stops, target exits,
kill switch, end-of-day square-off) and adds:
- lot-based position sizing (futures margin or option premium),
- routing bullish/bearish signals to ATM CE/PE buys in options mode,
- moving the stop to breakeven at +1R,
- a cooldown after a losing trade,
- refusing new entries when the tick feed is stale.
"""

from __future__ import annotations

import csv
import logging
import math
import queue
import time as _time
from datetime import datetime, timedelta
from pathlib import Path

from ..brokers.base import Broker, OrderStatus, OrderType
from ..config import Config
from ..engine import Position, TradingEngine
from ..market_hours import now_ist
from ..risk import round_to_tick
from ..strategies.base import Action, Signal
from .contracts import Contracts
from .footprint import FootprintAggregator
from .strategy import OrderflowConfluenceStrategy
from .ticks import TickRecorder

log = logging.getLogger(__name__)


class BankNiftyEngine(TradingEngine):
    def __init__(self, cfg: Config, broker: Broker, contracts: Contracts, aggregator: FootprintAggregator,
                 index_symbol: str | None = None, clock=now_ist, sleep=_time.sleep):
        cfg.symbols = [contracts.future_symbol]
        super().__init__(cfg, broker, OrderflowConfluenceStrategy(cfg.banknifty, cfg.session), clock, sleep, market_data=aggregator)
        self.bn = cfg.banknifty
        self.contracts = contracts
        self.agg = aggregator
        self.index_symbol = index_symbol
        self.options_mode = self.bn.instrument == "options"
        self.costs = cfg.option_costs if self.options_mode else cfg.costs
        self.bias: dict[str, int] = {}  # traded symbol -> +1 bullish / -1 bearish view
        self.initial_risk: dict[str, float] = {}
        self.at_breakeven: set[str] = set()
        self.cooldown_until: datetime | None = None
        self.journal = Path(cfg.log_dir) / "banknifty-trades.csv"
        self.signal_log = Path(cfg.log_dir) / "banknifty-signals.csv"

    def reconcile(self) -> None:
        """Don't trade if the account already holds any position in this underlying."""
        try:
            existing = self.broker.net_positions()
        except Exception:
            log.exception("Could not fetch broker positions; continuing")
            return
        mine = {s: q for s, q in existing.items() if s.upper().startswith(self.bn.underlying + "-")}
        if mine:
            self.blocked.add(self.contracts.future_symbol)
            log.warning("Existing %s positions %s were not opened by this run; not trading today", self.bn.underlying, mine)

    # ------------------------------------------------------------------ signals
    def _evaluate(self, sym: str, now: datetime) -> None:
        df = self.agg.candles(sym, self.cfg.interval_minutes, self.cfg.lookback_candles + 1)
        df = df[df.index + timedelta(minutes=self.cfg.interval_minutes) <= now]
        if df.empty:
            return
        sig = self.strategy.on_candles(df)
        self._log_signal(df.index[-1], sig)
        if sig.action is not Action.HOLD:
            log.info("%s %s %s", df.index[-1].strftime("%H:%M"), sig.action.value, sig.reason)
        self.handle_signal(sym, sig, float(df["close"].iloc[-1]), now)

    def _log_signal(self, bar_time, sig: Signal) -> None:
        f = getattr(self.strategy, "last_factors", {})
        if not f:
            return
        new = not self.signal_log.exists()
        with self.signal_log.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["bar", "action", "score", *f.keys()])
            w.writerow([bar_time.isoformat(), sig.action.value, round(self.strategy.last_score, 3), *[round(v, 2) for v in f.values()]])

    def handle_signal(self, sym: str, sig: Signal, last_close: float, now: datetime) -> None:
        a = sig.action
        # Close positions whose view the signal contradicts.
        for traded in list(self.positions):
            view = self.bias.get(traded, self.positions[traded].direction)
            if (view > 0 and a in (Action.EXIT_LONG, Action.SHORT)) or (view < 0 and a in (Action.EXIT_SHORT, Action.BUY)):
                self.close_position(traded, f"signal {a.value}")
        if a not in (Action.BUY, Action.SHORT) or self.positions:
            return
        if self.cooldown_until and now < self.cooldown_until:
            log.info("Entry skipped: cooling down after a loss until %s", self.cooldown_until.strftime("%H:%M"))
            return
        age = self.agg.feed_age_seconds(now)
        if age > self.bn.stale_feed_seconds:
            log.warning("Entry skipped: tick feed is stale (%.0fs since last tick)", age)
            return

        view = 1 if a is Action.BUY else -1
        if not self.options_mode:
            self.bias[sym] = view
            self.open_position(sym, view, sig, last_close, now)
            return

        ref = self.agg.ltp([self.index_symbol]).get(self.index_symbol) if self.index_symbol else None
        ref = ref or last_close
        opt = self.contracts.pick_option(ref, "CE" if view > 0 else "PE", self.bn.strike_offset)
        if opt is None:
            log.warning("No %s option near %.0f in the subscribed chain", "CE" if view > 0 else "PE", ref)
            return
        premium = self.agg.ltp([opt.symbol]).get(opt.symbol)
        if not premium:
            log.warning("No live price for %s yet; skipping entry", opt.symbol)
            return
        delta = self.bn.option_delta
        stop = min(sig.stop_distance * delta, premium * self.bn.max_option_stop_pct / 100)
        target = stop * self.bn.reward_risk
        self.bias[opt.symbol] = view
        self.open_position(opt.symbol, +1, Signal(Action.BUY, stop, target, f"{opt.symbol} | {sig.reason}"), premium, now)

    # ------------------------------------------------------------------ sizing
    def size_position(self, sym: str, price: float, stop_distance: float) -> int:
        lot, r = self.contracts.lot_size, self.cfg.risk
        risk_amount = r.capital * r.risk_per_trade_pct / 100
        lots = math.floor(risk_amount / (stop_distance * lot)) if stop_distance > 0 else 0
        lots = min(lots, self.bn.max_lots)
        cash = self.broker.available_cash()
        if self.options_mode:
            budget = min(cash, r.capital * r.max_position_value_pct / 100)
            lots = min(lots, math.floor(budget / (price * lot)))
        else:
            lots = min(lots, math.floor(cash / self.bn.futures_margin_per_lot))
        if lots <= 0:
            log.info("Size 0 lots for %s: risk budget %.0f vs %.0f per lot at stop %.1f (cash %.0f)",
                     sym, risk_amount, stop_distance * lot, stop_distance, cash)
        return max(lots, 0) * lot

    # ------------------------------------------------------------------ position management
    def open_position(self, sym, direction, sig, last_close, now) -> None:
        super().open_position(sym, direction, sig, last_close, now)
        pos = self.positions.get(sym)
        if pos:
            self.initial_risk[sym] = abs(pos.entry_price - pos.stop_price)

    def step(self, now: datetime) -> None:
        super().step(now)
        self._manage_breakeven()

    def _manage_breakeven(self) -> None:
        if self.bn.breakeven_at_r <= 0:
            return
        for sym, pos in list(self.positions.items()):
            if sym in self.at_breakeven or sym not in self.initial_risk:
                continue
            px = self.agg.ltp([sym]).get(sym)
            if px is None or (px - pos.entry_price) * pos.direction < self.bn.breakeven_at_r * self.initial_risk[sym]:
                continue
            self._move_stop(pos, pos.entry_price)
            self.at_breakeven.add(sym)

    def _move_stop(self, pos: Position, new_stop: float) -> None:
        ex = self.cfg.execution
        try:
            self.broker.cancel_order(pos.stop_order_id)
        except Exception:
            log.debug("cancel of stop %s failed", pos.stop_order_id)
        old = self.broker.get_order(pos.stop_order_id)
        if old.status is OrderStatus.COMPLETE:
            self._record_close(pos, old.avg_price, "stop loss")
            return
        if old.filled_qty:
            self.close_position(pos.symbol, "stop partially filled")
            return
        new_stop = round_to_tick(new_stop, ex.tick_size)
        side = pos.exit_side
        try:
            pos.stop_order_id = self._place(pos.symbol, side, pos.qty, OrderType.SL,
                                            self._limit_price(new_stop, side, ex.stop_limit_buffer_pct), trigger_price=new_stop)
            pos.stop_price = new_stop
            log.info("%s stop moved to breakeven %.2f", pos.symbol, new_stop)
        except Exception:
            log.exception("%s: could not re-place stop; exiting", pos.symbol)
            self.close_position(pos.symbol, "stop replace failed")

    def on_position_closed(self, pos: Position, net: float, reason: str) -> None:
        self.bias.pop(pos.symbol, None)
        self.initial_risk.pop(pos.symbol, None)
        self.at_breakeven.discard(pos.symbol)
        if net < 0 and self.bn.cooldown_minutes_after_loss:
            self.cooldown_until = self.clock() + timedelta(minutes=self.bn.cooldown_minutes_after_loss)

    # ------------------------------------------------------------------ live loop
    def run_session(self, tick_queue: "queue.Queue", recorder: TickRecorder | None = None) -> None:
        """Trade one session from a live tick queue; returns after the market closes."""
        log.info("BankNifty %s mode: signals on %s, trading %s, lot size %d",
                 self.cfg.mode.upper(), self.contracts.future_symbol, self.bn.instrument, self.contracts.lot_size)
        self.reconcile()
        last_step = 0.0
        while True:
            now = self.clock()
            if not self.calendar.is_open(now):
                if self.calendar.is_trading_day(now) and now.time() < self.cfg.session.market_open:
                    self.sleep(min((self.calendar.next_open(now) - now).total_seconds(), 30))
                    continue
                self.square_off_all("market closed")
                self._end_of_day_report()
                return
            try:
                t = tick_queue.get(timeout=0.5)
                while True:
                    self.agg.on_tick(t)
                    if recorder:
                        recorder.write(t)
                    t = tick_queue.get_nowait()
            except queue.Empty:
                pass
            if _time.monotonic() - last_step >= 1.0:
                last_step = _time.monotonic()
                try:
                    self.step(self.clock())
                except Exception:
                    log.exception("Error during step")
