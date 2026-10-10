"""Dhan (DhanHQ API v2) broker for live trading and market data.

Requires `pip install dhanhq`, your Dhan client ID and an access token. Generate the token at
web.dhan.co -> My Profile -> Access DhanHQ APIs; tokens are valid for 24 hours.

Dhan identifies instruments by numeric security ID, not symbol. IDs are looked up from Dhan's
scrip master CSV (cached locally for a day), or can be pinned in config under `dhan_security_ids`.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from ..data.feeds import resample
from ..market_hours import IST
from .base import Broker, Order, OrderStatus, OrderType, Side

log = logging.getLogger(__name__)

SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
_SEGMENTS = {"NSE": "NSE_EQ", "BSE": "BSE_EQ"}
_PRODUCTS = {"MIS": "INTRADAY", "INTRADAY": "INTRADAY", "CNC": "CNC"}
# Dhan serves 1/5/15/25/60-minute candles; fetch a base interval and resample so every candle
# is anchored at 09:15 like the rest of the bot.
_BASE_INTERVAL = {1: 1, 3: 1, 5: 5, 10: 5, 15: 15, 30: 15, 60: 15}
_STATUS = {
    "TRADED": OrderStatus.COMPLETE,
    "CANCELLED": OrderStatus.CANCELLED,
    "EXPIRED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
}


class DhanAPIError(RuntimeError):
    pass


def _unwrap(resp: dict, what: str):
    """dhanhq returns {'status': 'success'|'failure', 'remarks': ..., 'data': ...}."""
    if not isinstance(resp, dict) or resp.get("status") != "success":
        remarks = resp.get("remarks") if isinstance(resp, dict) else resp
        raise DhanAPIError(f"Dhan {what} failed: {remarks}")
    return resp.get("data")


def load_security_ids(exchange: str = "NSE", cache_dir: str | Path = "logs", max_age_hours: float = 20) -> dict[str, str]:
    """Map trading symbol -> Dhan security ID for cash-market equities, from the scrip master."""
    cache = Path(cache_dir) / "dhan-scrip-master.csv"
    if not cache.exists() or time.time() - cache.stat().st_mtime > max_age_hours * 3600:
        import requests

        log.info("Downloading Dhan scrip master")
        r = requests.get(SCRIP_MASTER_URL, timeout=60)
        r.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(r.content)
    df = pd.read_csv(cache, low_memory=False, dtype=str)
    df = df[(df["SEM_EXM_EXCH_ID"] == exchange) & (df["SEM_SEGMENT"] == "E")]
    if "SEM_SERIES" in df.columns:
        df = df[df["SEM_SERIES"].isin(["EQ", "BE"])]
    return dict(zip(df["SEM_TRADING_SYMBOL"].str.upper(), df["SEM_SMST_SECURITY_ID"]))


class DhanBroker(Broker):
    def __init__(
        self,
        client,
        exchange: str = "NSE",
        product: str = "MIS",
        tag: str = "algobot",
        security_ids: dict[str, str | int] | None = None,
        cache_dir: str = "logs",
        min_quote_interval: float = 1.1,
    ):
        self.client = client
        self.exchange = exchange
        self.segment = _SEGMENTS[exchange]
        self.product = _PRODUCTS[product.upper()]
        self.tag = tag[:25]
        self._cache_dir = cache_dir
        self._ids = {k.upper(): str(v) for k, v in (security_ids or {}).items()}
        self._master: dict[str, str] | None = None
        # Dhan's market quote API allows ~1 request/second.
        self._min_quote_interval = min_quote_interval
        self._last_quote = 0.0
        self._tag_seq = 0

    @classmethod
    def from_env(cls, **kw) -> "DhanBroker":
        client_id, token = os.environ.get("DHAN_CLIENT_ID"), os.environ.get("DHAN_ACCESS_TOKEN")
        if not client_id or not token:
            raise SystemExit("Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN (see README: Dhan setup).")
        try:
            from dhanhq import DhanContext, dhanhq
        except ImportError as e:  # pragma: no cover
            raise SystemExit("dhanhq is not installed: pip install dhanhq") from e
        return cls(dhanhq(DhanContext(client_id, token)), **kw)

    # ------------------------------------------------------------------ instruments
    def security_id(self, symbol: str) -> str:
        symbol = symbol.upper()
        if symbol in self._ids:
            return self._ids[symbol]
        if self._master is None:
            self._master = load_security_ids(self.exchange, self._cache_dir)
        try:
            self._ids[symbol] = self._master[symbol]
        except KeyError:
            raise ValueError(f"No Dhan security ID for {self.exchange} symbol {symbol}") from None
        return self._ids[symbol]

    # ------------------------------------------------------------------ market data
    def ltp(self, symbols: list[str]) -> dict[str, float]:
        if not symbols:
            return {}
        wait = self._min_quote_interval - (time.monotonic() - self._last_quote)
        if wait > 0:
            time.sleep(wait)
        ids = {self.security_id(s): s for s in symbols}
        data = _unwrap(self.client.ticker_data({self.segment: [int(i) for i in ids]}), "LTP")
        self._last_quote = time.monotonic()
        # The v2 payload is {"data": {"NSE_EQ": {"1594": {"last_price": ...}}}, "status": "success"}.
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        quotes = (data or {}).get(self.segment, {})
        return {ids[str(sid)]: float(q["last_price"]) for sid, q in quotes.items() if str(sid) in ids}

    def candles(self, symbol: str, interval_minutes: int, count: int) -> pd.DataFrame:
        base = _BASE_INTERVAL[interval_minutes]
        to_dt = datetime.now(IST) + timedelta(days=1)
        days = min(89, max(5, int(count * interval_minutes / 375 * 1.6) + 4))
        data = _unwrap(
            self.client.intraday_minute_data(
                self.security_id(symbol), self.segment, "EQUITY",
                (to_dt - timedelta(days=days)).strftime("%Y-%m-%d"), to_dt.strftime("%Y-%m-%d"), interval=base,
            ),
            "historical data",
        )
        if not data or not data.get("timestamp"):
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        idx = pd.to_datetime(pd.Series(data["timestamp"], dtype="int64"), unit="s", utc=True).dt.tz_convert(IST)
        df = pd.DataFrame({k: data[k] for k in ("open", "high", "low", "close", "volume")}, index=pd.DatetimeIndex(idx)).astype(float)
        return resample(df.sort_index(), interval_minutes).tail(count)

    # ------------------------------------------------------------------ orders
    def place_order(self, symbol, side, qty, order_type, price, trigger_price=None) -> str:
        self._tag_seq += 1
        resp = self.client.place_order(
            security_id=self.security_id(symbol),
            exchange_segment=self.segment,
            transaction_type="BUY" if side is Side.BUY else "SELL",
            quantity=int(qty),
            order_type="STOP_LOSS" if order_type is OrderType.SL else "LIMIT",
            product_type=self.product,
            price=float(price),
            trigger_price=float(trigger_price or 0),
            validity="DAY",
            tag=f"{self.tag}-{self._tag_seq}"[:25],
        )
        data = _unwrap(resp, f"place {side.value} {symbol}")
        if str(data.get("orderStatus", "")).upper() == "REJECTED":
            raise DhanAPIError(f"Dhan rejected {side.value} {symbol}: {data}")
        order_id = str(data["orderId"])
        log.info("[dhan] placed %s %s %s x%d @ %s trig=%s -> %s", order_type.value, side.value, symbol, qty, price, trigger_price, order_id)
        return order_id

    def get_order(self, order_id: str) -> Order:
        data = _unwrap(self.client.get_order_by_id(order_id), f"order {order_id}")
        if isinstance(data, list):
            data = data[-1]
        raw_status = str(data.get("orderStatus", "")).upper()
        is_sl = str(data.get("orderType", "")).upper() == "STOP_LOSS"
        status = _STATUS.get(raw_status)
        if status is None:
            status = OrderStatus.TRIGGER_PENDING if is_sl and raw_status == "PENDING" else OrderStatus.OPEN
        return Order(
            order_id=str(data.get("orderId", order_id)),
            symbol=str(data.get("tradingSymbol", "")),
            side=Side(str(data.get("transactionType", "BUY")).upper()),
            qty=int(data.get("quantity") or 0),
            order_type=OrderType.SL if is_sl else OrderType.LIMIT,
            price=float(data.get("price") or 0),
            trigger_price=float(data.get("triggerPrice") or 0) or None,
            status=status,
            filled_qty=int(data.get("filledQty") or 0),
            avg_price=float(data.get("averageTradedPrice") or 0),
            message=str(data.get("omsErrorDescription") or ""),
        )

    def cancel_order(self, order_id: str) -> None:
        _unwrap(self.client.cancel_order(order_id), f"cancel {order_id}")

    def available_cash(self) -> float:
        data = _unwrap(self.client.get_fund_limits(), "fund limits")
        # Dhan's field really is spelled "availabelBalance".
        return float(data.get("availabelBalance", data.get("availableBalance", 0)))

    def net_positions(self) -> dict[str, int]:
        data = _unwrap(self.client.get_positions(), "positions") or []
        out = {}
        for p in data:
            if p.get("exchangeSegment") == self.segment and p.get("productType") == self.product and int(p.get("netQty") or 0):
                out[str(p["tradingSymbol"]).upper()] = int(p["netQty"])
        return out
