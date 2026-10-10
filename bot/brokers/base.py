from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    SL = "SL"  # stop-limit: triggers at trigger_price, then works as a limit at price


class OrderStatus(str, Enum):
    OPEN = "OPEN"
    TRIGGER_PENDING = "TRIGGER_PENDING"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

    @property
    def is_final(self) -> bool:
        return self in (OrderStatus.COMPLETE, OrderStatus.CANCELLED, OrderStatus.REJECTED)


@dataclass
class Order:
    order_id: str
    symbol: str
    side: Side
    qty: int
    order_type: OrderType
    price: float
    trigger_price: float | None = None
    status: OrderStatus = OrderStatus.OPEN
    filled_qty: int = 0
    avg_price: float = 0.0
    message: str = ""


class MarketData(ABC):
    @abstractmethod
    def ltp(self, symbols: list[str]) -> dict[str, float]:
        """Last traded price per symbol."""

    @abstractmethod
    def candles(self, symbol: str, interval_minutes: int, count: int) -> pd.DataFrame:
        """Recent OHLCV candles, indexed by IST candle-start time, oldest first.

        May include the currently-forming candle; callers filter it out.
        """


class Broker(MarketData):
    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: Side,
        qty: int,
        order_type: OrderType,
        price: float,
        trigger_price: float | None = None,
    ) -> str:
        """Places an intraday order and returns its id. Raises on rejection at submission."""

    @abstractmethod
    def get_order(self, order_id: str) -> Order: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> None: ...

    @abstractmethod
    def available_cash(self) -> float: ...

    @abstractmethod
    def net_positions(self) -> dict[str, int]:
        """Signed net intraday quantity per symbol (non-zero only)."""
