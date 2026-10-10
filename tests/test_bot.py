from datetime import date, datetime, time

import numpy as np
import pandas as pd
import pytest

from bot.backtest import backtest
from bot.brokers.base import MarketData, OrderStatus, OrderType, Side
from bot.brokers.paper import PaperBroker
from bot.config import Config, SessionConfig, load_config
from bot.data.feeds import resample
from bot.engine import TradingEngine
from bot.indicators import atr, ema, rsi
from bot.market_hours import IST, MarketCalendar, last_closed_candle_end
from bot.risk import RiskManager, trade_costs
from bot.strategies import Action, EmaRsiStrategy, Signal


def ist(y, mo, d, h, mi, s=0):
    return datetime(y, mo, d, h, mi, s, tzinfo=IST)


def synthetic_candles(days=10, interval=5, seed=1):
    rng = np.random.default_rng(seed)
    idx = []
    for day in pd.bdate_range("2026-01-05", periods=days):
        start = pd.Timestamp(day.date()).tz_localize(IST) + pd.Timedelta(hours=9, minutes=15)
        idx.extend(start + pd.Timedelta(minutes=interval * k) for k in range(375 // interval))
    # Trending regimes so crossovers actually happen.
    drift = np.sin(np.arange(len(idx)) / 40) * 0.002
    close = 1000 * np.exp(np.cumsum(drift + rng.normal(0, 0.002, len(idx))))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.0015, len(idx))) * close
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) + spread, "low": np.minimum(open_, close) - spread, "close": close, "volume": 1000},
        index=pd.DatetimeIndex(idx),
    )


# ---------------------------------------------------------------- indicators & strategy
def test_indicators_basic():
    s = pd.Series(np.arange(1, 51, dtype=float))
    assert ema(s, 5).iloc[-1] < 50 and ema(s, 5).iloc[-1] > ema(s, 20).iloc[-1]
    assert rsi(s, 14).iloc[-1] == 100.0
    assert rsi(pd.Series([10.0] * 30), 14).iloc[-1] == 50.0
    df = pd.DataFrame({"high": s + 1, "low": s - 1, "close": s})
    assert atr(df, 14).iloc[-1] == pytest.approx(2.0, rel=0.05)


def test_strategy_buy_on_cross_up():
    cfg = Config().strategy
    cfg.rsi_buy_max = 100
    strat = EmaRsiStrategy(cfg)
    down = np.linspace(110, 100, 60)
    up = np.linspace(100, 104, 8)
    close = np.r_[down, up]
    df = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close})
    fast, slow = ema(df.close, cfg.fast_ema), ema(df.close, cfg.slow_ema)
    cross = next(i for i in range(1, len(df)) if fast[i - 1] <= slow[i - 1] and fast[i] > slow[i])
    sig = strat.on_candles(df.iloc[: cross + 1])
    assert sig.action is Action.BUY
    assert sig.stop_distance > 0 and sig.target_distance == pytest.approx(sig.stop_distance * cfg.reward_risk)


def test_strategy_warmup_holds():
    df = synthetic_candles(1).iloc[:10]
    assert EmaRsiStrategy(Config().strategy).on_candles(df).action is Action.HOLD


# ---------------------------------------------------------------- risk
def test_position_size_respects_risk_and_caps():
    rm = RiskManager(Config().risk)  # 1L capital, 1% risk, 20% max position
    assert rm.position_size(100, 2) == 200  # 1000 risk / 2 = 500, capped to 20000/100 = 200
    assert rm.position_size(1000, 10) == 20  # 100 by risk, capped 20 by value
    assert rm.position_size(100, 20) == 50  # risk-limited
    assert rm.position_size(100, 2, available_cash=5000) == 50
    assert rm.position_size(100, 0) == 0


def test_kill_switch():
    rm = RiskManager(Config().risk)
    rm.on_trade_closed(-1500)
    assert not rm.check_loss_limit(-400)
    assert rm.check_loss_limit(-600)
    assert rm.can_open(0) == (False, rm.can_open(0)[1]) and "halted" in rm.can_open(0)[1]
    rm.new_day()
    assert rm.can_open(0)[0]


def test_trade_costs_reasonable():
    cost = trade_costs(100_000, 100_500, Config().costs)
    # ~ 40 brokerage + 25 STT + ~6 exch + 3 stamp + GST ≈ 80
    assert 60 < cost < 110


# ---------------------------------------------------------------- calendar & config
def test_calendar():
    cal = MarketCalendar(SessionConfig(holidays=[date(2026, 1, 26)]))
    assert cal.is_open(ist(2026, 1, 6, 10, 0))
    assert not cal.is_open(ist(2026, 1, 6, 15, 30))
    assert not cal.is_open(ist(2026, 1, 10, 10, 0))  # Saturday
    assert not cal.is_trading_day(ist(2026, 1, 26, 10, 0))
    assert cal.next_open(ist(2026, 1, 23, 16, 0)) == ist(2026, 1, 27, 9, 15)  # Fri -> skip weekend + holiday
    assert cal.can_enter(ist(2026, 1, 6, 9, 30)) and not cal.can_enter(ist(2026, 1, 6, 15, 0))
    assert cal.must_square_off(ist(2026, 1, 6, 15, 12))
    assert last_closed_candle_end(ist(2026, 1, 6, 9, 27, 3), 5, SessionConfig()) == ist(2026, 1, 6, 9, 25)


def test_load_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("symbols: [infy, tcs]\nrisk:\n  capital: 50000\nsession:\n  square_off_at: '15:10'\n  holidays: [2026-11-09]\n")
    cfg = load_config(p)
    assert cfg.symbols == ["INFY", "TCS"] and cfg.risk.capital == 50000
    assert cfg.session.square_off_at == time(15, 10) and cfg.session.holidays == [date(2026, 11, 9)]
    p.write_text("risk:\n  max_orders_per_second: 12\n")
    with pytest.raises(ValueError, match="SEBI"):
        load_config(p)
    p.write_text("riskk: {}\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(p)


def test_example_config_is_valid():
    cfg = load_config("config.example.yaml")
    assert cfg.mode == "paper"


def test_resample_anchors_at_open():
    df = synthetic_candles(1, interval=1)
    out = resample(df, 15)
    assert out.index[0].time() == time(9, 15) and out.index[1].time() == time(9, 30)
    assert len(out) == 25


# ---------------------------------------------------------------- backtest
def test_backtest_runs_and_is_flat_overnight():
    cfg = Config()
    df = synthetic_candles(15)
    res = backtest(df, "TEST", EmaRsiStrategy(cfg.strategy), cfg)
    assert res.trades, "expected some trades on trending synthetic data"
    for t in res.trades:
        assert t.entry_time.date() == t.exit_time.date()
        assert t.exit_time.time() <= cfg.session.square_off_at or t.reason != "square-off"
        assert cfg.session.entry_start <= t.entry_time.time() < cfg.session.no_new_entries_after
    s = res.summary()
    assert s["trades"] == len(res.trades) and s["total_costs"] > 0
    assert res.equity.iloc[-1] == pytest.approx(cfg.risk.capital + s["net_pnl"], abs=0.05)


# ---------------------------------------------------------------- engine with paper broker
class FakeFeed(MarketData):
    def __init__(self, price=100.0):
        self.price = price

    def ltp(self, symbols):
        return {s: self.price for s in symbols}

    def candles(self, symbol, interval_minutes, count):
        return synthetic_candles(2)


def make_engine(tmp_path, price=100.0):
    cfg = Config(symbols=["ABC"], log_dir=str(tmp_path))
    cfg.costs.slippage_pct = 0
    feed = FakeFeed(price)
    broker = PaperBroker(feed, cfg.risk.capital, slippage_pct=0, ltp_cache_seconds=0)
    eng = TradingEngine(cfg, broker, EmaRsiStrategy(cfg.strategy), clock=lambda: ist(2026, 1, 6, 10, 0), sleep=lambda s: None)
    return eng, broker, feed


def test_engine_entry_places_protective_stop_and_hits_stop(tmp_path):
    eng, broker, feed = make_engine(tmp_path)
    now = ist(2026, 1, 6, 10, 0)
    eng.step(now)  # new day
    eng.handle_signal("ABC", Signal(Action.BUY, stop_distance=2, target_distance=4), 100, now)
    pos = eng.positions["ABC"]
    assert pos.qty == 200 and pos.stop_price == 98 and pos.target_price == 104
    stop = broker.orders[pos.stop_order_id]
    assert stop.order_type is OrderType.SL and stop.side is Side.SELL and stop.status is OrderStatus.TRIGGER_PENDING
    assert broker.net_positions() == {"ABC": 200}

    feed.price = 97.9
    eng.step(now)
    assert "ABC" not in eng.positions and broker.net_positions() == {}
    assert eng.risk.day.realized_pnl < -400
    assert "stop loss" in (tmp_path / "trades.csv").read_text()


def test_engine_target_exit_cancels_stop(tmp_path):
    eng, broker, feed = make_engine(tmp_path)
    now = ist(2026, 1, 6, 10, 0)
    eng.step(now)
    eng.handle_signal("ABC", Signal(Action.BUY, 2, 4), 100, now)
    stop_id = eng.positions["ABC"].stop_order_id
    feed.price = 104.5
    eng.step(now)
    assert "ABC" not in eng.positions and broker.net_positions() == {}
    assert broker.orders[stop_id].status is OrderStatus.CANCELLED
    assert eng.risk.day.realized_pnl > 0


def test_engine_short_and_square_off(tmp_path):
    eng, broker, feed = make_engine(tmp_path)
    eng.cfg.strategy.allow_short = True
    now = ist(2026, 1, 6, 10, 0)
    eng.step(now)
    eng.handle_signal("ABC", Signal(Action.SHORT, 2, 4), 100, now)
    assert broker.net_positions() == {"ABC": -200}
    eng.step(ist(2026, 1, 6, 15, 12, 30))
    assert broker.net_positions() == {} and not eng.positions


def test_engine_no_entries_late_or_when_halted(tmp_path):
    eng, broker, _ = make_engine(tmp_path)
    eng.step(ist(2026, 1, 6, 10, 0))
    eng.handle_signal("ABC", Signal(Action.BUY, 2, 4), 100, ist(2026, 1, 6, 15, 5))
    assert not eng.positions
    eng.risk.halt("test")
    eng.handle_signal("ABC", Signal(Action.BUY, 2, 4), 100, ist(2026, 1, 6, 10, 5))
    assert not eng.positions


def test_engine_kill_switch_squares_off(tmp_path):
    eng, broker, feed = make_engine(tmp_path)
    now = ist(2026, 1, 6, 10, 0)
    eng.step(now)
    eng.handle_signal("ABC", Signal(Action.BUY, 20, 40), 100, now)  # wide stop: 50 shares
    assert eng.positions["ABC"].qty == 50
    eng.risk.on_trade_closed(-1800)
    feed.price = 95  # -250 unrealized -> -2050 total > 2000 limit
    eng.step(now)
    assert eng.risk.day.halted and not eng.positions and broker.net_positions() == {}


def test_engine_skips_symbols_with_existing_positions(tmp_path):
    eng, broker, _ = make_engine(tmp_path)
    broker.positions["ABC"] = 10
    eng.reconcile()
    assert "ABC" in eng.blocked
