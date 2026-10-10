"""Bar-by-bar intraday backtester using the same strategy, sizing and session rules as the live engine.

Fill model: signals on a candle's close are executed at the next candle's open (plus slippage).
Stops/targets are checked against each later candle's high/low; if both are touched in the same
candle the stop is assumed to have hit first. Open positions are squared off at the first candle
starting at or after `square_off_at`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .risk import RiskManager, trade_costs
from .strategies import Action, Strategy


@dataclass
class Trade:
    symbol: str
    direction: int
    qty: int
    entry_time: pd.Timestamp
    entry: float
    exit_time: pd.Timestamp
    exit: float
    reason: str
    gross: float
    costs: float

    @property
    def net(self) -> float:
        return self.gross - self.costs


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity: pd.Series
    starting_capital: float

    def summary(self) -> dict:
        nets = np.array([t.net for t in self.trades])
        wins, losses = nets[nets > 0], nets[nets <= 0]
        eq = self.equity
        dd = (eq - eq.cummax()).min() if len(eq) else 0.0
        daily = eq.groupby(eq.index.date).last().diff().dropna() if len(eq) else pd.Series(dtype=float)
        sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if len(daily) > 1 and daily.std() > 0 else 0.0
        return {
            "trades": len(nets),
            "win_rate_pct": round(100 * len(wins) / len(nets), 1) if len(nets) else 0.0,
            "net_pnl": round(float(nets.sum()), 2),
            "return_pct": round(100 * float(nets.sum()) / self.starting_capital, 2),
            "total_costs": round(sum(t.costs for t in self.trades), 2),
            "avg_win": round(float(wins.mean()), 2) if len(wins) else 0.0,
            "avg_loss": round(float(losses.mean()), 2) if len(losses) else 0.0,
            "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if len(losses) and losses.sum() < 0 else float("inf") if len(wins) else 0.0,
            "max_drawdown": round(float(dd), 2),
            "max_drawdown_pct": round(100 * float(dd) / self.starting_capital, 2),
            "daily_sharpe_annualized": round(sharpe, 2),
        }


def backtest(df: pd.DataFrame, symbol: str, strategy: Strategy, cfg: Config) -> BacktestResult:
    s, c = cfg.session, cfg.costs
    slip = c.slippage_pct / 100
    risk = RiskManager(cfg.risk)
    trades: list[Trade] = []
    equity, eq_idx = [], []
    realized = 0.0
    pos = None  # dict(direction, qty, entry, stop, target, time)
    pending = None  # (Action, Signal) to execute at next open
    current_day = None
    lookback = cfg.lookback_candles

    def close(t, price, reason):
        nonlocal pos, realized
        d, q = pos["direction"], pos["qty"]
        fill = price * (1 - slip * d)
        gross = (fill - pos["entry"]) * q * d
        buy_v, sell_v = (pos["entry"] * q, fill * q) if d > 0 else (fill * q, pos["entry"] * q)
        cost = trade_costs(buy_v, sell_v, c)
        trades.append(Trade(symbol, d, q, pos["time"], pos["entry"], t, round(fill, 2), reason, gross, cost))
        risk.on_trade_closed(gross - cost)
        realized += gross - cost
        pos = None

    times = df.index
    for i in range(len(df)):
        t = times[i]
        bar = df.iloc[i]
        tod = t.time()
        if t.date() != current_day:
            current_day = t.date()
            risk.new_day()
            pending = None

        # 1) execute the previous candle's signal at this candle's open
        if pending is not None:
            action, sig = pending
            pending = None
            if pos and ((action in (Action.EXIT_LONG, Action.SHORT) and pos["direction"] > 0) or (action in (Action.EXIT_SHORT, Action.BUY) and pos["direction"] < 0)):
                close(t, bar.open, f"signal {action.value}")
            if action in (Action.BUY, Action.SHORT) and pos is None and s.entry_start <= tod < s.no_new_entries_after and risk.can_open(0)[0]:
                d = 1 if action is Action.BUY else -1
                entry = bar.open * (1 + slip * d)
                qty = risk.position_size(entry, sig.stop_distance)
                if qty > 0:
                    risk.on_trade_opened()
                    pos = dict(direction=d, qty=qty, entry=entry, time=t,
                               stop=entry - d * sig.stop_distance, target=entry + d * sig.target_distance)

        # 2) end-of-day square-off
        if pos and tod >= s.square_off_at:
            close(t, bar.open, "square-off")

        # 3) stops / targets within this candle
        if pos:
            d = pos["direction"]
            stop_hit = bar.low <= pos["stop"] if d > 0 else bar.high >= pos["stop"]
            target_hit = bar.high >= pos["target"] if d > 0 else bar.low <= pos["target"]
            if stop_hit:
                # Gap through the stop fills at the open.
                px = min(bar.open, pos["stop"]) if d > 0 else max(bar.open, pos["stop"])
                close(t, px, "stop loss")
            elif target_hit:
                px = max(bar.open, pos["target"]) if d > 0 else min(bar.open, pos["target"])
                close(t, px, "target")

        # 4) daily loss kill switch (on close of this candle)
        unreal = (bar.close - pos["entry"]) * pos["qty"] * pos["direction"] if pos else 0.0
        if risk.check_loss_limit(unreal) and pos:
            close(t, bar.close, "kill switch")
            unreal = 0.0

        # 5) signal on this candle's close
        if tod < s.square_off_at:
            sig = strategy.on_candles(df.iloc[max(0, i + 1 - lookback): i + 1])
            if sig.action is not Action.HOLD:
                pending = (sig.action, sig)

        equity.append(cfg.risk.capital + realized + unreal)
        eq_idx.append(t)

    if pos:
        close(times[-1], df["close"].iloc[-1], "end of data")
        equity[-1] = cfg.risk.capital + realized
    return BacktestResult(trades, pd.Series(equity, index=eq_idx, dtype=float), cfg.risk.capital)
