import csv
import math
from datetime import datetime, timedelta, time

import numpy as np
import pandas as pd
import pytest

from bot.banknifty.contracts import Contracts, OptionContract, resolve_contracts
from bot.banknifty.footprint import FootprintAggregator, FootprintBar, absorption, stacked_imbalances, value_area
from bot.banknifty.runner import INDEX_SYMBOL, replay, summarize
from bot.banknifty.strategy import OrderflowConfluenceStrategy
from bot.banknifty.ticks import TICK_FIELDS, DhanTickFeed, Tick, TickRecorder, TradeClassifier, load_ticks
from bot.config import Config, load_config
from bot.indicators import session_vwap, supertrend
from bot.market_hours import IST
from bot.strategies.base import Action

FUT = "BANKNIFTY-Oct2026-FUT"


def ts(h, m, s=0, day=12):
    return datetime(2026, 10, day, h, m, s, tzinfo=IST)


def tick(t, price, vol, bid=None, ask=None, sym=FUT, **kw):
    return Tick(t, sym, price, volume=vol, bid=bid if bid is not None else price - 0.05,
                ask=ask if ask is not None else price + 0.05, **kw)


# ---------------------------------------------------------------- classification & footprint
def test_classifier_quote_and_tick_rule():
    c = TradeClassifier()
    assert c.classify(tick(ts(9, 15), 100, 1000, 99.95, 100.05)) == (0, 0)  # first packet
    assert c.classify(tick(ts(9, 15, 1), 100.05, 1030, 100, 100.10)) == (30, 1)  # at prior ask
    assert c.classify(tick(ts(9, 15, 2), 100.00, 1050, 99.95, 100.05)) == (20, -1)  # at prior bid
    assert c.classify(tick(ts(9, 15, 3), 100.02, 1060, 0, 0)) == (10, 1)  # inside spread, uptick
    assert c.classify(tick(ts(9, 15, 4), 100.02, 1060)) == (0, 0)  # no new volume
    assert c.classify(tick(ts(9, 15, 0, day=13), 101, 50)) == (0, 0)  # new day resets


def test_value_area_and_poc():
    levels = {100.0: [10, 10], 105.0: [50, 50], 110.0: [20, 20], 115.0: [5, 5]}
    poc, vah, val = value_area(levels)
    assert (poc, vah, val) == (105.0, 110.0, 105.0)  # 100 + 40 of 170 already covers 70%


def test_stacked_imbalances():
    # ask volume at P vs bid volume one row below
    levels = {100.0: [10, 0], 105.0: [10, 300], 110.0: [10, 300], 115.0: [10, 300], 120.0: [10, 5]}
    assert stacked_imbalances(levels, 5, 3.0, 100) == (3, 0)
    levels = {100.0: [400, 10], 105.0: [400, 10], 110.0: [400, 10], 115.0: [0, 10]}
    assert stacked_imbalances(levels, 5, 3.0, 100)[1] == 3


def test_absorption():
    b = FootprintBar(ts(10, 0), 100, 120, 100, 115)
    b.levels = {100.0: [500, 50], 110.0: [100, 100], 115.0: [50, 100]}
    b.buy_vol, b.sell_vol = 250, 650
    assert absorption(b, 0.3) == (True, False)


def test_aggregator_builds_bars_with_flow():
    cfg = Config()
    agg = FootprintAggregator(FUT, 5, cfg.session, cfg.banknifty)
    vol, price = 0, 57000.0
    for i in range(600):  # 10 minutes of 1s ticks, buyers lifting the offer
        vol += 35
        price += 0.5
        agg.on_tick(tick(ts(9, 15) + timedelta(seconds=i), price, vol, bid=price - 0.05, ask=price, oi=1000 + i, bid_qty=900, ask_qty=300))
    df = agg.candles(FUT, 5, 10)
    assert len(df) == 2 and df.index[1] == ts(9, 20)
    b = df.iloc[0]
    assert b.buy_vol > 0 and b.sell_vol == 0 and b.delta_pct == 1.0 and b.has_flow
    assert b.book_imb == pytest.approx(0.5) and b.oi_chg > 0
    assert agg.feed_age_seconds(ts(9, 25, 30)) == pytest.approx(31)


def test_vwap_and_supertrend():
    idx = pd.date_range("2026-10-12 09:15", periods=60, freq="5min", tz=IST)
    c = np.linspace(100, 160, 60)
    df = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 10.0}, index=idx)
    assert session_vwap(df).iloc[-1] < df.close.iloc[-1]
    assert supertrend(df).iloc[-1] == 1
    assert supertrend(df.iloc[::-1].set_axis(idx)).iloc[-1] == -1


# ---------------------------------------------------------------- strategy
def flow_frame(direction=1, n=60):
    idx = []
    for d in (9, 12):
        start = pd.Timestamp(f"2026-10-{d:02d} 09:15", tz=IST)
        idx += [start + pd.Timedelta(minutes=5 * k) for k in range(n // 2)]
    c = 57000 + direction * np.arange(n) * 15.0
    df = pd.DataFrame({"open": c - direction * 10, "high": c + 20, "low": c - 20, "close": c, "volume": 5000.0}, index=pd.DatetimeIndex(idx))
    df["buy_vol"] = 3000.0 if direction > 0 else 1000.0
    df["sell_vol"] = 4000.0 - df["buy_vol"]
    df["delta"] = df.buy_vol - df.sell_vol
    df["delta_pct"] = df.delta / 4000
    df["poc"] = df.close - direction * 10
    df["vah"], df["val"] = df.high, df.low
    df["stacked_buy"], df["stacked_sell"] = (4, 0) if direction > 0 else (0, 4)
    df["absorb_bull"] = df["absorb_bear"] = False
    df["book_imb"] = 0.3 * direction
    df["oi"] = 1e6
    df["oi_chg"] = 500.0
    df["has_flow"] = True
    return df


@pytest.mark.parametrize("direction,action", [(1, Action.BUY), (-1, Action.SHORT)])
def test_strategy_enters_with_trend_and_flow(direction, action):
    s = OrderflowConfluenceStrategy(Config().banknifty, Config().session)
    sig = s.on_candles(flow_frame(direction))
    assert sig.action is action, sig.reason
    assert s.last_score * direction > 0.45
    assert 40 <= sig.stop_distance <= 150 and sig.target_distance == pytest.approx(2 * sig.stop_distance)
    assert s.last_factors["stacked_imbalance"] == direction and s.last_factors["vwap"] == direction


def test_strategy_needs_live_flow():
    df = flow_frame(1)
    df.loc[df.index[-1], "has_flow"] = False
    sig = OrderflowConfluenceStrategy(Config().banknifty, Config().session).on_candles(df)
    assert sig.action is not Action.BUY


def test_strategy_respects_required_filter():
    cfg = Config().banknifty
    cfg.required_filters = ["vwap", "delta_divergence"]  # divergence is 0 on a clean trend
    sig = OrderflowConfluenceStrategy(cfg, Config().session).on_candles(flow_frame(1))
    assert sig.action is Action.EXIT_SHORT and "blocked" in sig.reason


# ---------------------------------------------------------------- contracts, feed parsing, config
def test_resolve_contracts():
    rows = [
        ("NSE", "D", "51700", "FUTIDX", "BANKNIFTY-Oct2026-FUT", "35.0", "2026-10-27 14:30:00", "-0.01", "XX"),
        ("NSE", "D", "51800", "FUTIDX", "BANKNIFTY-Nov2026-FUT", "35.0", "2026-11-24 14:30:00", "-0.01", "XX"),
        ("NSE", "D", "70001", "FUTIDX", "FINNIFTY-Oct2026-FUT", "65.0", "2026-10-27 14:30:00", "-0.01", "XX"),
    ]
    for i, k in enumerate(range(56500, 57600, 100)):
        for ot in ("CE", "PE"):
            rows.append(("NSE", "D", str(80000 + 2 * i + (ot == "PE")), "OPTIDX", f"BANKNIFTY-Oct2026-{k}-{ot}", "35.0", "2026-10-27 14:30:00", f"{k}.0", ot))
    cols = ["SEM_EXM_EXCH_ID", "SEM_SEGMENT", "SEM_SMST_SECURITY_ID", "SEM_INSTRUMENT_NAME", "SEM_TRADING_SYMBOL",
            "SEM_LOT_UNITS", "SEM_EXPIRY_DATE", "SEM_STRIKE_PRICE", "SEM_OPTION_TYPE"]
    c = resolve_contracts(pd.DataFrame(rows, columns=cols), "BANKNIFTY", datetime(2026, 10, 12).date())
    assert (c.future_symbol, c.future_id, c.lot_size, c.strike_step) == (FUT, "51700", 35, 100.0)
    assert c.pick_option(57040, "CE").symbol == "BANKNIFTY-Oct2026-57000-CE"
    assert c.pick_option(57040, "PE", offset=1).strike == 56900
    assert len(c.options_near(57000, 2)) == 10
    with pytest.raises(ValueError):  # on expiry day with skip, no later option expiry in this master
        resolve_contracts(pd.DataFrame(rows, columns=cols), "BANKNIFTY", datetime(2026, 10, 27).date())


def test_dhan_feed_parse():
    feed = DhanTickFeed("1", "t", [(FUT, "NSE_FNO", 51700, "full")])
    msg = {"type": "Full Data", "exchange_segment": 2, "security_id": 51700, "LTP": "57010.50", "LTQ": 35, "volume": 123450,
           "total_buy_quantity": 9000, "total_sell_quantity": 7000, "OI": 2000000,
           "depth": [{"bid_quantity": 100, "ask_quantity": 50, "bid_price": "57010.00", "ask_price": "57010.50"}] * 5}
    t = feed.parse(msg)
    assert (t.symbol, t.ltp, t.volume, t.bid, t.ask, t.bid_qty, t.ask_qty, t.oi) == (FUT, 57010.5, 123450, 57010.0, 57010.5, 500, 250, 2000000)
    assert feed.parse({"type": "Full Data", "exchange_segment": 2, "security_id": 1, "LTP": "1"}) is None
    assert feed.instruments == [(2, "51700", 21)]


def test_example_config_loads():
    cfg = load_config("banknifty.example.yaml")
    assert cfg.banknifty.instrument == "options" and cfg.option_costs.stt_sell_pct > 0


def test_unknown_weight_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("banknifty:\n  weights:\n    magic: 1\n")
    with pytest.raises(ValueError, match="magic"):
        load_config(p)


# ---------------------------------------------------------------- end-to-end replay
def write_session(dirpath, day, seed, trend_sign=1):
    """Synthetic session: trending future with matching aggressor flow, index, and ATM options."""
    rng = np.random.default_rng(seed)
    contracts = Contracts("BANKNIFTY", FUT, "51700", "2026-10-27", 35, 100.0,
                          [OptionContract(f"BANKNIFTY-Oct2026-{k}-{ot}", str(k) + ot, float(k), ot, "2026-10-27")
                           for k in range(55000, 59100, 100) for ot in ("CE", "PE")])
    contracts.save(dirpath / f"2026-10-{day:02d}.contracts.json")
    rec = TickRecorder(dirpath)
    price, vol, oi = 57000.0, 0, 2_000_000
    t0 = datetime(2026, 10, day, 9, 15, tzinfo=IST)
    for i in range(0, 6 * 3600 + 15 * 60, 2):
        t = t0 + timedelta(seconds=i)
        regime = trend_sign * (1 if (i // 2400) % 2 == 0 else -1)  # 40-minute swings
        step = regime * 0.6 + rng.normal(0, 2.0)
        price = round(price + step, 1)
        q = int(rng.integers(1, 6)) * 35
        vol += q
        oi += q if regime > 0 else -q // 2
        aggressive_buy = (step > 0) if rng.random() < 0.85 else (step <= 0)
        bid, ask = price - 0.5, price + 0.5
        ltp = ask if aggressive_buy else bid
        rec.write(Tick(t, FUT, ltp, q, vol, bid, ask, 800 if regime > 0 else 400, 400 if regime > 0 else 800, 0, 0, oi))
        rec.write(Tick(t, INDEX_SYMBOL, price - 120))
        atm = round((price - 120) / 100) * 100
        for k in (atm - 100, atm, atm + 100):
            rec.write(Tick(t, f"BANKNIFTY-Oct2026-{k}-CE", round(max(400 + 0.5 * (price - 120 - k), 5), 1)))
            rec.write(Tick(t, f"BANKNIFTY-Oct2026-{k}-PE", round(max(400 - 0.5 * (price - 120 - k), 5), 1)))
    rec.close()
    return dirpath / f"2026-10-{day:02d}.csv"


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    d = tmp_path_factory.mktemp("ticks")
    return [write_session(d, day, seed) for day, seed in ((12, 1), (13, 2), (14, 3))]


def test_recorded_ticks_roundtrip(sessions):
    first = next(load_ticks(sessions[0]))
    assert first.symbol == FUT and first.ts.tzinfo is not None
    with open(sessions[0]) as f:
        assert next(csv.reader(f)) == TICK_FIELDS


@pytest.mark.parametrize("instrument", ["options", "futures"])
def test_replay_end_to_end(sessions, tmp_path, instrument):
    cfg = load_config("banknifty.example.yaml")
    cfg.log_dir = str(tmp_path)
    cfg.banknifty.instrument = instrument
    cfg.banknifty.record_ticks = False
    cfg.risk.capital = 1_000_000
    engine = replay(cfg, sessions)
    trades = engine.closed_trades
    assert trades, "expected the synthetic trends to produce trades"
    assert not engine.positions and engine.broker.net_positions() == {}
    lot = 35
    for t in trades:
        assert t["qty"] % lot == 0
        assert t["entry_time"].time() >= cfg.session.entry_start
        assert t["entry_time"].time() < cfg.session.no_new_entries_after
        assert t["exit_time"].date() == t["entry_time"].date()
        if instrument == "options":
            assert t["symbol"].endswith(("CE", "PE")) and t["direction"] == 1
        else:
            assert t["symbol"] == FUT
    per_day = pd.Series([t["entry_time"].date() for t in trades]).value_counts()
    assert per_day.max() <= cfg.risk.max_trades_per_day
    s = summarize(trades, cfg.risk.capital)
    assert s["trades"] == len(trades) and s["total_costs"] > 0
    assert (tmp_path / "banknifty-signals.csv").exists()


# ---------------------------------------------------------------- candle backtest
def test_candle_backtest_rules():
    from bot.banknifty.candle_backtest import SimParams, precompute_features, simulate, technical_only

    cfg = technical_only(load_config("banknifty.example.yaml"))
    assert cfg.banknifty.weights["bar_delta"] == 0 and not cfg.banknifty.require_orderflow
    rng = np.random.default_rng(3)
    idx, closes = [], []
    price = 57000.0
    for day in pd.bdate_range("2026-07-06", periods=6):
        start = pd.Timestamp(day.date()).tz_localize(IST) + pd.Timedelta(hours=9, minutes=15)
        for k in range(75):
            idx.append(start + pd.Timedelta(minutes=5 * k))
            price += 8 * np.sin(len(closes) / 12) + rng.normal(0, 15)
            closes.append(price)
    c = np.array(closes)
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) + 15, "low": np.minimum(o, c) - 15, "close": c, "volume": 1000.0},
                      index=pd.DatetimeIndex(idx))
    res = simulate(df, precompute_features(df, cfg), cfg, SimParams(lot_size=35, slippage_points=2))
    tr = res.frame()
    assert len(tr) > 0
    s = cfg.session
    assert all(s.entry_start <= t.time() < s.no_new_entries_after for t in tr.entry_time)
    assert all(t.exit_time.date() == t.entry_time.date() for t in tr.itertuples())
    assert tr.groupby("date").size().max() <= cfg.risk.max_trades_per_day
    for t in tr.itertuples():  # entries fill at the next candle's open plus slippage
        side = 1 if t.side == "LONG" else -1
        assert t.entry == pytest.approx(df.loc[t.entry_time, "open"] + 2 * side, abs=0.01)
    assert res.summary()["trades"] == len(tr)
