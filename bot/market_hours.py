"""NSE session calendar helpers (all times in IST)."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .config import SessionConfig

IST = ZoneInfo("Asia/Kolkata")


def now_ist() -> datetime:
    return datetime.now(IST)


class MarketCalendar:
    def __init__(self, session: SessionConfig):
        self.s = session
        self._holidays = set(session.holidays)

    def is_trading_day(self, ts: datetime) -> bool:
        d = ts.astimezone(IST).date()
        return d.weekday() < 5 and d not in self._holidays

    def _t(self, ts: datetime):
        return ts.astimezone(IST).time()

    def is_open(self, ts: datetime) -> bool:
        return self.is_trading_day(ts) and self.s.market_open <= self._t(ts) < self.s.market_close

    def can_enter(self, ts: datetime) -> bool:
        return self.is_trading_day(ts) and self.s.entry_start <= self._t(ts) < self.s.no_new_entries_after

    def must_square_off(self, ts: datetime) -> bool:
        return self.is_trading_day(ts) and self._t(ts) >= self.s.square_off_at

    def next_open(self, ts: datetime) -> datetime:
        ts = ts.astimezone(IST)
        candidate = ts.replace(hour=self.s.market_open.hour, minute=self.s.market_open.minute, second=0, microsecond=0)
        if candidate <= ts:
            candidate += timedelta(days=1)
        while not self.is_trading_day(candidate):
            candidate += timedelta(days=1)
        return candidate


def last_closed_candle_end(ts: datetime, interval_minutes: int, session: SessionConfig) -> datetime:
    """End time of the most recent fully-closed candle, with candles anchored at market open."""
    ts = ts.astimezone(IST)
    open_dt = ts.replace(hour=session.market_open.hour, minute=session.market_open.minute, second=0, microsecond=0)
    elapsed = int((ts - open_dt).total_seconds() // 60)
    n = max(elapsed // interval_minutes, 0)
    return open_dt + timedelta(minutes=n * interval_minutes)
