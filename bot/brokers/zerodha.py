"""Zerodha Kite Connect broker (live trading and market data).

Requires `pip install kiteconnect`, a Kite Connect app (api_key/api_secret) and a daily
access token (see `python -m bot login`). Tokens expire every morning around 6 AM IST.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import pandas as pd

from ..market_hours import IST
from .base import Broker, Order, OrderStatus, OrderType, Side

log = logging.getLogger(__name__)

_INTERVALS = {1: "minute", 3: "3minute", 5: "5minute", 10: "10minute", 15: "15minute", 30: "30minute", 60: "60minute"}
_STATUS = {
    "COMPLETE": OrderStatus.COMPLETE,
    "CANCELLED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "TRIGGER PENDING": OrderStatus.TRIGGER_PENDING,
}


def _kite_client(api_key: str, access_token: str | None = None):
    try:
        from kiteconnect import KiteConnect
    except ImportError as e:  # pragma: no cover
        raise SystemExit("kiteconnect is not installed: pip install kiteconnect") from e
    kite = KiteConnect(api_key=api_key)
    if access_token:
        kite.set_access_token(access_token)
    return kite


def login_url(api_key: str) -> str:
    return _kite_client(api_key).login_url()


def exchange_request_token(api_key: str, api_secret: str, request_token: str) -> str:
    session = _kite_client(api_key).generate_session(request_token, api_secret=api_secret)
    return session["access_token"]


class ZerodhaBroker(Broker):
    def __init__(self, api_key: str, access_token: str, exchange: str = "NSE", product: str = "MIS", tag: str = "algobot", algo_id: str = ""):
        self.kite = _kite_client(api_key, access_token)
        self.exchange, self.product, self.tag, self.algo_id = exchange, product, tag[:20], algo_id
        self._tokens: dict[str, int] | None = None

    @classmethod
    def from_env(cls, **kw) -> "ZerodhaBroker":
        key, token = os.environ.get("KITE_API_KEY"), os.environ.get("KITE_ACCESS_TOKEN")
        if not key or not token:
            raise SystemExit("Set KITE_API_KEY and KITE_ACCESS_TOKEN (run `python -m bot login` first).")
        return cls(key, token, **kw)

    def _key(self, symbol: str) -> str:
        return f"{self.exchange}:{symbol}"

    def _instrument_token(self, symbol: str) -> int:
        if self._tokens is None:
            self._tokens = {i["tradingsymbol"]: i["instrument_token"] for i in self.kite.instruments(self.exchange)}
        try:
            return self._tokens[symbol]
        except KeyError:
            raise ValueError(f"Unknown {self.exchange} symbol: {symbol}") from None

    def ltp(self, symbols: list[str]) -> dict[str, float]:
        data = self.kite.ltp([self._key(s) for s in symbols])
        return {s: data[self._key(s)]["last_price"] for s in symbols if self._key(s) in data}

    def candles(self, symbol: str, interval_minutes: int, count: int) -> pd.DataFrame:
        to_dt = datetime.now(IST)
        # ~75 five-minute candles per session; look back enough calendar days to cover weekends/holidays.
        days = max(5, int(count * interval_minutes / 375 * 1.6) + 4)
        rows = self.kite.historical_data(
            self._instrument_token(symbol), to_dt - timedelta(days=days), to_dt, _INTERVALS[interval_minutes]
        )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"]).dt.tz_convert(IST)
        return df.set_index("date")[["open", "high", "low", "close", "volume"]].tail(count)

    def place_order(self, symbol, side, qty, order_type, price, trigger_price=None) -> str:
        k = self.kite
        params = dict(
            variety=k.VARIETY_REGULAR,
            exchange=self.exchange,
            tradingsymbol=symbol,
            transaction_type=k.TRANSACTION_TYPE_BUY if side is Side.BUY else k.TRANSACTION_TYPE_SELL,
            quantity=int(qty),
            product=self.product,
            order_type=k.ORDER_TYPE_SL if order_type is OrderType.SL else k.ORDER_TYPE_LIMIT,
            price=price,
            validity=k.VALIDITY_DAY,
            tag=self.tag,
        )
        if order_type is OrderType.SL:
            params["trigger_price"] = trigger_price
        if self.algo_id:
            params["algo_id"] = self.algo_id
        order_id = k.place_order(**params)
        log.info("[kite] placed %s %s %s x%d @ %s trig=%s -> %s", order_type.value, side.value, symbol, qty, price, trigger_price, order_id)
        return str(order_id)

    def get_order(self, order_id: str) -> Order:
        h = self.kite.order_history(order_id)[-1]
        status = _STATUS.get(h["status"], OrderStatus.OPEN)
        return Order(
            order_id=order_id,
            symbol=h["tradingsymbol"],
            side=Side(h["transaction_type"]),
            qty=int(h["quantity"]),
            order_type=OrderType.SL if h["order_type"] == "SL" else OrderType.LIMIT,
            price=float(h.get("price") or 0),
            trigger_price=float(h.get("trigger_price") or 0) or None,
            status=status,
            filled_qty=int(h.get("filled_quantity") or 0),
            avg_price=float(h.get("average_price") or 0),
            message=h.get("status_message") or "",
        )

    def cancel_order(self, order_id: str) -> None:
        self.kite.cancel_order(variety=self.kite.VARIETY_REGULAR, order_id=order_id)

    def available_cash(self) -> float:
        return float(self.kite.margins(segment="equity")["net"])

    def net_positions(self) -> dict[str, int]:
        out = {}
        for p in self.kite.positions()["net"]:
            if p["exchange"] == self.exchange and p["product"] == self.product and p["quantity"]:
                out[p["tradingsymbol"]] = int(p["quantity"])
        return out
