"""EMA crossover with RSI confirmation and ATR-based stop/target."""

from __future__ import annotations

import math

import pandas as pd

from ..config import StrategyConfig
from ..indicators import atr, ema, rsi
from .base import Action, Signal, Strategy


class EmaRsiStrategy(Strategy):
    name = "ema_rsi"

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    def min_candles(self) -> int:
        return max(self.cfg.slow_ema * 3, self.cfg.rsi_period + 2, self.cfg.atr_period + 2)

    def on_candles(self, df: pd.DataFrame) -> Signal:
        c = self.cfg
        if len(df) < self.min_candles():
            return Signal(Action.HOLD, reason="warming up")

        close = df["close"]
        fast, slow = ema(close, c.fast_ema), ema(close, c.slow_ema)
        r = rsi(close, c.rsi_period).iloc[-1]
        a = atr(df, c.atr_period).iloc[-1]
        if math.isnan(r) or math.isnan(a) or a <= 0:
            return Signal(Action.HOLD, reason="indicators not ready")

        crossed_up = fast.iloc[-2] <= slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]
        crossed_down = fast.iloc[-2] >= slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1]
        stop = a * c.atr_stop_mult
        target = stop * c.reward_risk

        if crossed_up:
            if c.rsi_buy_min <= r <= c.rsi_buy_max:
                return Signal(Action.BUY, stop, target, f"EMA{c.fast_ema}>EMA{c.slow_ema}, RSI {r:.1f}")
            return Signal(Action.EXIT_SHORT, reason=f"bullish cross (RSI {r:.1f} outside entry band)")
        if crossed_down:
            if c.allow_short and c.rsi_sell_min <= r <= c.rsi_sell_max:
                return Signal(Action.SHORT, stop, target, f"EMA{c.fast_ema}<EMA{c.slow_ema}, RSI {r:.1f}")
            return Signal(Action.EXIT_LONG, reason=f"bearish cross, RSI {r:.1f}")
        return Signal(Action.HOLD)
