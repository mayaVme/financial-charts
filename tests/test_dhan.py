import pandas as pd
import pytest

from bot.brokers.base import OrderStatus, OrderType, Side
from bot.brokers.dhan import DhanAPIError, DhanBroker
from bot.market_hours import IST


def ok(data):
    return {"status": "success", "remarks": "", "data": data}


class FakeDhan:
    """Mimics the dhanhq client's method names and response envelopes."""

    def __init__(self):
        self.placed = []
        self.orders = {}
        self.cancelled = []

    def ticker_data(self, securities):
        assert securities == {"NSE_EQ": [1594, 2885]}
        return ok({"data": {"NSE_EQ": {"1594": {"last_price": 1500.5}, "2885": {"last_price": 2900}}}, "status": "success"})

    def intraday_minute_data(self, security_id, exchange_segment, instrument_type, from_date, to_date, interval=1, oi=False):
        assert (security_id, exchange_segment, instrument_type, interval) == ("1594", "NSE_EQ", "EQUITY", 5)
        start = pd.Timestamp("2026-01-06 09:15", tz=IST)
        ts = [int((start + pd.Timedelta(minutes=5 * i)).timestamp()) for i in range(6)]
        return ok({"timestamp": ts, "open": [1, 2, 3, 4, 5, 6], "high": [2, 3, 4, 5, 6, 7], "low": [0.5] * 6, "close": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5], "volume": [10] * 6})

    def place_order(self, **kw):
        self.placed.append(kw)
        return ok({"orderId": "9001", "orderStatus": "PENDING"})

    def get_order_by_id(self, order_id):
        return ok(self.orders[order_id])

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        return ok({"orderId": order_id, "orderStatus": "CANCELLED"})

    def get_fund_limits(self):
        return ok({"availabelBalance": 98765.4})

    def get_positions(self):
        return ok([
            {"tradingSymbol": "INFY", "exchangeSegment": "NSE_EQ", "productType": "INTRADAY", "netQty": 10},
            {"tradingSymbol": "TCS", "exchangeSegment": "NSE_EQ", "productType": "CNC", "netQty": 5},
            {"tradingSymbol": "SBIN", "exchangeSegment": "NSE_EQ", "productType": "INTRADAY", "netQty": 0},
        ])


@pytest.fixture
def broker():
    return DhanBroker(FakeDhan(), security_ids={"INFY": 1594, "TCS": 2885}, min_quote_interval=0)


def test_ltp(broker):
    assert broker.ltp(["INFY", "TCS"]) == {"INFY": 1500.5, "TCS": 2900.0}


def test_candles_are_ist_and_anchored(broker):
    df = broker.candles("INFY", 5, 100)
    assert len(df) == 6 and df.index[0] == pd.Timestamp("2026-01-06 09:15", tz=IST)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_place_limit_and_stop_orders(broker):
    assert broker.place_order("INFY", Side.BUY, 10, OrderType.LIMIT, 1505.0) == "9001"
    broker.place_order("INFY", Side.SELL, 10, OrderType.SL, 1470.0, trigger_price=1480.0)
    entry, stop = broker.client.placed
    assert entry["security_id"] == "1594" and entry["exchange_segment"] == "NSE_EQ"
    assert entry["product_type"] == "INTRADAY" and entry["order_type"] == "LIMIT" and entry["transaction_type"] == "BUY"
    assert stop["order_type"] == "STOP_LOSS" and stop["trigger_price"] == 1480.0 and stop["price"] == 1470.0
    assert entry["tag"] != stop["tag"]


def test_order_status_mapping(broker):
    broker.client.orders = {
        "1": {"orderId": "1", "orderStatus": "TRADED", "transactionType": "BUY", "orderType": "LIMIT", "quantity": 10, "filledQty": 10, "averageTradedPrice": 1501.2, "tradingSymbol": "INFY"},
        "2": {"orderId": "2", "orderStatus": "PENDING", "transactionType": "SELL", "orderType": "STOP_LOSS", "quantity": 10, "filledQty": 0, "triggerPrice": 1480},
        "3": [{"orderId": "3", "orderStatus": "REJECTED", "transactionType": "BUY", "orderType": "LIMIT", "quantity": 5, "omsErrorDescription": "RMS: margin"}],
        "4": {"orderId": "4", "orderStatus": "PART_TRADED", "transactionType": "BUY", "orderType": "LIMIT", "quantity": 10, "filledQty": 4, "averageTradedPrice": 100},
    }
    o = broker.get_order("1")
    assert o.status is OrderStatus.COMPLETE and o.filled_qty == 10 and o.avg_price == 1501.2
    assert broker.get_order("2").status is OrderStatus.TRIGGER_PENDING
    rej = broker.get_order("3")
    assert rej.status is OrderStatus.REJECTED and "margin" in rej.message
    part = broker.get_order("4")
    assert part.status is OrderStatus.OPEN and part.filled_qty == 4


def test_funds_and_positions(broker):
    assert broker.available_cash() == 98765.4
    assert broker.net_positions() == {"INFY": 10}


def test_failure_raises(broker):
    broker.client.get_fund_limits = lambda: {"status": "failure", "remarks": {"error_code": "DH-901"}, "data": ""}
    with pytest.raises(DhanAPIError, match="DH-901"):
        broker.available_cash()


def test_symbol_lookup_from_scrip_master(tmp_path):
    (tmp_path / "dhan-scrip-master.csv").write_text(
        "SEM_EXM_EXCH_ID,SEM_SEGMENT,SEM_SMST_SECURITY_ID,SEM_INSTRUMENT_NAME,SEM_TRADING_SYMBOL,SEM_SERIES\n"
        "NSE,E,1594,EQUITY,INFY,EQ\n"
        "BSE,E,500209,EQUITY,INFY,A\n"
        "NSE,D,35001,FUTSTK,INFY-Jan2026-FUT,\n"
    )
    b = DhanBroker(FakeDhan(), cache_dir=str(tmp_path))
    assert b.security_id("infy") == "1594"
    with pytest.raises(ValueError):
        b.security_id("NOPE")


def test_real_sdk_has_the_methods_we_call():
    dhanhq = pytest.importorskip("dhanhq")
    client = dhanhq.dhanhq(dhanhq.DhanContext("1000000001", "token"))
    for m in ("ticker_data", "intraday_minute_data", "place_order", "get_order_by_id", "cancel_order", "get_fund_limits", "get_positions"):
        assert callable(getattr(client, m)), m
