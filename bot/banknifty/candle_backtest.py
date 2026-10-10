"""Backtest the BankNifty strategy on plain OHLCV futures candles.

Historical tick data isn't available, so order-flow factors can't be reconstructed from candles.
This backtester runs the *technical* half of the confluence model (VWAP, EMA trend, Supertrend,
CPR, opening range, RSI) through the same decision code and trade management rules as the live
engine:

- signal on a candle's close, executed at the next candle's open (plus slippage);
- one position at a time, entries only inside the entry window, max trades per day,
  cooldown after a losing trade, daily loss kill switch, square-off at `square_off_at`;
- stop/target checked on each candle's high/low; if both are touched, the stop counts first;
- stop moves to breakeven after a candle reaches +`breakeven_at_r` R (from the next candle).

Results are per 1 futures lot so they don't depend on account size. An options estimate scales
the futures points by `option_delta` and ignores theta and IV changes, so it is optimistic.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace
from datetime import timedelta

import numpy as np
import pandas as pd

from ..config import Config
from ..indicators import atr
from ..risk import trade_costs
from ..strategies.base import Action
from .footprint import FRAME_COLUMNS
from .strategy import OrderflowConfluenceStrategy

ORDERFLOW_FACTORS = ["bar_delta", "cvd_slope", "stacked_imbalance", "absorption", "delta_divergence",
                     "poc_acceptance", "book_imbalance", "oi_buildup"]


@dataclass
class SimParams:
    lot_size: int = 35
    slippage_points: float = 2.0  # per side, futures
    option_cost_per_trade: float = 200.0  # rupees per lot round trip incl. option slippage (estimate)


@dataclass
class CandleTrade:
    direction: int
    entry_time: pd.Timestamp
    entry: float
    exit_time: pd.Timestamp
    exit: float
    reason: str
    stop_points: float
    points: float = 0.0
    costs: float = 0.0


@dataclass
class CandleResult:
    trades: list[CandleTrade]
    lot_size: int
    option_delta: float
    option_cost_per_trade: float
    skipped: dict = field(default_factory=dict)

    def frame(self) -> pd.DataFrame:
        rows = []
        for t in self.trades:
            gross = t.points * self.lot_size
            rows.append({
                "date": t.entry_time.date(), "entry_time": t.entry_time, "exit_time": t.exit_time,
                "side": "LONG" if t.direction > 0 else "SHORT", "entry": round(t.entry, 2), "exit": round(t.exit, 2),
                "stop_pts": round(t.stop_points, 1), "points": round(t.points, 2), "reason": t.reason,
                "gross_inr": round(gross, 2), "costs_inr": round(t.costs, 2), "net_inr": round(gross - t.costs, 2),
                "opt_est_inr": round(t.points * self.option_delta * self.lot_size - self.option_cost_per_trade, 2),
                "minutes": (t.exit_time - t.entry_time).total_seconds() / 60,
            })
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        df = self.frame()
        if df.empty:
            return {"trades": 0}
        net = df.net_inr
        eq = net.cumsum()
        dd = float((eq - np.maximum.accumulate(np.r_[0.0, eq.to_numpy()])[1:]).min())
        wins, losses = net[net > 0], net[net <= 0]
        daily = df.groupby("date").net_inr.sum()
        return {
            "trades": int(len(df)),
            "days_traded": int(df.date.nunique()),
            "win_rate_pct": round(100 * (df.points > 0).mean(), 1),
            "gross_points": round(float(df.points.sum()), 1),
            "avg_points_per_trade": round(float(df.points.mean()), 1),
            "net_inr_per_lot": round(float(net.sum()), 0),
            "costs_inr_per_lot": round(float(df.costs_inr.sum()), 0),
            "avg_win_inr": round(float(wins.mean()), 0) if len(wins) else 0,
            "avg_loss_inr": round(float(losses.mean()), 0) if len(losses) else 0,
            "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else None,
            "max_drawdown_inr": round(min(dd, 0.0), 0),
            "best_day_inr": round(float(daily.max()), 0),
            "worst_day_inr": round(float(daily.min()), 0),
            "options_estimate_inr_per_lot": round(float(df.opt_est_inr.sum()), 0),
            "long_trades": int((df.side == "LONG").sum()),
            "short_trades": int((df.side == "SHORT").sum()),
            "avg_minutes_held": round(float(df.minutes.mean()), 0),
            "exit_reasons": df.reason.value_counts().to_dict(),
        }


def technical_only(cfg: Config) -> Config:
    """Copy of cfg with order-flow factors switched off (they can't be computed from candles)."""
    cfg = copy.deepcopy(cfg)
    bn = cfg.banknifty
    bn.weights = {k: (0.0 if k in ORDERFLOW_FACTORS else v) for k, v in bn.weights.items()}
    bn.require_orderflow = False
    bn.required_filters = [f for f in bn.required_filters if f not in ORDERFLOW_FACTORS]
    return cfg


def with_flow_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["open", "high", "low", "close", "volume"]].astype(float).copy()
    for c in FRAME_COLUMNS:
        if c not in out.columns:
            out[c] = False if c in ("has_flow", "absorb_bull", "absorb_bear") else math.nan
    return out[FRAME_COLUMNS]


def precompute_features(df: pd.DataFrame, cfg: Config) -> list[tuple | None]:
    """Per-candle (factors, close, low, high, atr) computed on the trailing lookback window."""
    strat = OrderflowConfluenceStrategy(cfg.banknifty, cfg.session)
    data = with_flow_columns(df)
    a = atr(data[["high", "low", "close"]], 14)
    feats: list[tuple | None] = []
    look = cfg.lookback_candles
    for i in range(len(data)):
        window = data.iloc[max(0, i + 1 - look): i + 1]
        if len(window) < strat.min_candles():
            feats.append(None)
            continue
        bar = data.iloc[i]
        feats.append((strat.factors(window), float(bar.close), float(bar.low), float(bar.high), float(a.iloc[i])))
    return feats


def simulate(df: pd.DataFrame, feats: list, cfg: Config, sim: SimParams = SimParams(),
             direction_override: np.ndarray | None = None) -> CandleResult:
    """Run the trade-management rules over precomputed features.

    direction_override: optional array of +1/-1 per candle to replace each entry's direction
    (used for the random-direction control test).
    """
    bn, ss, r = cfg.banknifty, cfg.session, cfg.risk
    strat = OrderflowConfluenceStrategy(bn, ss)
    slip = sim.slippage_points
    max_daily_loss = r.capital * r.max_daily_loss_pct / 100
    trades: list[CandleTrade] = []
    skipped = {"outside_window": 0, "max_trades": 0, "cooldown": 0, "halted": 0}

    pos = None
    pending = None
    day = None
    day_trades = 0
    day_pnl = 0.0
    halted = False
    cooldown_until = None
    idx = df.index

    def close(t, px, reason):
        nonlocal pos, day_pnl, cooldown_until
        d = pos["dir"]
        fill = px - d * slip
        pts = (fill - pos["entry"]) * d
        notional_in = pos["entry"] * sim.lot_size
        notional_out = fill * sim.lot_size
        buy_v, sell_v = (notional_in, notional_out) if d > 0 else (notional_out, notional_in)
        costs = trade_costs(buy_v, sell_v, replace(cfg.costs, slippage_pct=0))
        trades.append(CandleTrade(d, pos["time"], pos["entry"], t, fill, reason, pos["risk"], pts, costs))
        net = pts * sim.lot_size - costs
        day_pnl += net
        if net < 0 and bn.cooldown_minutes_after_loss:
            cooldown_until = t + timedelta(minutes=bn.cooldown_minutes_after_loss)
        pos = None

    for i in range(len(df)):
        t = idx[i]
        o, h, l, c = (float(df.iloc[i][k]) for k in ("open", "high", "low", "close"))
        tod = t.time()
        if t.date() != day:
            day, day_trades, day_pnl, halted, cooldown_until, pending = t.date(), 0, 0.0, False, None, None

        # 1) act on the previous candle's signal at this open
        if pending is not None:
            sig = pending
            pending = None
            a = sig.action
            if pos and ((pos["dir"] > 0 and a in (Action.EXIT_LONG, Action.SHORT)) or (pos["dir"] < 0 and a in (Action.EXIT_SHORT, Action.BUY))):
                close(t, o, f"signal {a.value}")
            if a in (Action.BUY, Action.SHORT) and pos is None:
                if not (ss.entry_start <= tod < ss.no_new_entries_after):
                    skipped["outside_window"] += 1
                elif halted:
                    skipped["halted"] += 1
                elif day_trades >= r.max_trades_per_day:
                    skipped["max_trades"] += 1
                elif cooldown_until and t < cooldown_until:
                    skipped["cooldown"] += 1
                else:
                    d = 1 if a is Action.BUY else -1
                    if direction_override is not None:
                        d = int(direction_override[i])
                    entry = o + d * slip
                    pos = {"dir": d, "entry": entry, "time": t, "risk": sig.stop_distance,
                           "stop": entry - d * sig.stop_distance, "target": entry + d * sig.target_distance, "be": False}
                    day_trades += 1

        # 2) end-of-day square-off
        if pos and tod >= ss.square_off_at:
            close(t, o, "square-off")

        # 3) stop / target inside this candle
        if pos:
            d = pos["dir"]
            stop_hit = l <= pos["stop"] if d > 0 else h >= pos["stop"]
            target_hit = h >= pos["target"] if d > 0 else l <= pos["target"]
            if stop_hit:
                px = min(o, pos["stop"]) if d > 0 else max(o, pos["stop"])
                close(t, px, "breakeven stop" if pos["be"] else "stop loss")
            elif target_hit:
                px = max(o, pos["target"]) if d > 0 else min(o, pos["target"])
                close(t, px, "target")
            elif bn.breakeven_at_r > 0 and not pos["be"]:
                best = (h - pos["entry"]) if d > 0 else (pos["entry"] - l)
                if best >= bn.breakeven_at_r * pos["risk"]:
                    pos["stop"], pos["be"] = pos["entry"], True

        # 4) daily kill switch on the close
        if pos:
            unreal = (c - pos["entry"]) * pos["dir"] * sim.lot_size
            if day_pnl + unreal <= -max_daily_loss:
                close(t, c, "kill switch")
                halted = True
        elif day_pnl <= -max_daily_loss:
            halted = True

        # 5) signal on this close
        if tod < ss.square_off_at and feats[i] is not None:
            f, fc, fl, fh, fa = feats[i]
            sig = strat.decide(f, fc, fl, fh, False, fa)
            if sig.action is not Action.HOLD:
                pending = sig

    if pos:
        close(idx[-1], float(df.iloc[-1].close), "end of data")
    return CandleResult(trades, sim.lot_size, bn.option_delta, sim.option_cost_per_trade, skipped)
