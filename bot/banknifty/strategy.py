"""Order-flow + technical confluence strategy for BankNifty futures candles.

Every factor scores the latest closed candle from -1 (bearish) to +1 (bullish). The weighted
average of all factors is the confluence score. Factors that can't be computed count as 0, so
missing data makes the bot *less* likely to trade, never more.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..config import BankNiftyConfig, SessionConfig
from ..indicators import atr, ema, rsi, session_vwap, supertrend
from ..strategies.base import Action, Signal, Strategy


def _sign(x: float, dead: float = 0.0) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    return 1.0 if x > dead else -1.0 if x < -dead else 0.0


def _clip(x: float) -> float:
    return 0.0 if x is None or math.isnan(x) else float(max(-1.0, min(1.0, x)))


class OrderflowConfluenceStrategy(Strategy):
    name = "banknifty_orderflow"

    def __init__(self, bn: BankNiftyConfig, session: SessionConfig):
        self.bn, self.session = bn, session
        self.last_factors: dict[str, float] = {}
        self.last_score: float = 0.0

    def min_candles(self) -> int:
        return 30

    # ------------------------------------------------------------------ factors
    def factors(self, df: pd.DataFrame) -> dict[str, float]:
        bn = self.bn
        last = df.iloc[-1]
        close = df["close"].astype(float)
        day = df.index[-1].date()
        today = df[df.index.date == day]
        f: dict[str, float] = {}

        # --- technical context
        ohlcv = df[["open", "high", "low", "close", "volume"]].astype(float)
        vwap = session_vwap(ohlcv).iloc[-1]
        f["vwap"] = _sign((last.close - vwap) / vwap if vwap and not math.isnan(vwap) else math.nan, 0.0002)
        f["ema_trend"] = _sign(ema(close, 9).iloc[-1] - ema(close, 21).iloc[-1])
        f["supertrend"] = float(supertrend(ohlcv).iloc[-1])

        prev = df[df.index.date < day]
        if not prev.empty:
            pd_ = prev[prev.index.date == prev.index[-1].date()]
            H, L, C = pd_["high"].max(), pd_["low"].min(), pd_["close"].iloc[-1]
            pivot, bc = (H + L + C) / 3, (H + L) / 2
            tc = 2 * pivot - bc
            top, bottom = max(tc, bc), min(tc, bc)
            f["cpr"] = 1.0 if last.close > top else -1.0 if last.close < bottom else 0.0
        else:
            f["cpr"] = 0.0

        or_end = today.index[0].normalize() + pd.Timedelta(hours=self.session.market_open.hour,
                                                           minutes=self.session.market_open.minute + bn.opening_range_minutes)
        orng = today[today.index < or_end]
        if not orng.empty and df.index[-1] >= or_end:
            f["opening_range"] = 1.0 if last.close > orng["high"].max() else -1.0 if last.close < orng["low"].min() else 0.0
        else:
            f["opening_range"] = 0.0

        r = rsi(close, 14).iloc[-1]
        f["rsi"] = 0.0 if math.isnan(r) else 1.0 if r > 55 else -1.0 if r < 45 else 0.0

        # --- order flow
        flow = today[today["has_flow"].astype(bool)] if "has_flow" in today else today.iloc[0:0]
        has_flow = bool(last.get("has_flow", False))
        f["bar_delta"] = _clip(last.delta_pct / 0.25) if has_flow else 0.0

        if has_flow and len(flow) >= 2:
            cvd = flow["delta"].astype(float).cumsum()
            k = min(bn.cvd_slope_bars, len(cvd) - 1)
            window = flow.iloc[-k:]
            vol = float((window["buy_vol"] + window["sell_vol"]).sum())
            f["cvd_slope"] = _clip((cvd.iloc[-1] - cvd.iloc[-1 - k]) / vol / 0.15) if vol else 0.0
            L = min(bn.divergence_lookback, len(flow) - 1)
            hi_prev, lo_prev = flow["high"].iloc[-1 - L:-1].max(), flow["low"].iloc[-1 - L:-1].min()
            cvd_hi, cvd_lo = cvd.iloc[-1 - L:-1].max(), cvd.iloc[-1 - L:-1].min()
            if last.high >= hi_prev and cvd.iloc[-1] < cvd_hi:
                f["delta_divergence"] = -1.0  # new high on weaker buying
            elif last.low <= lo_prev and cvd.iloc[-1] > cvd_lo:
                f["delta_divergence"] = 1.0  # new low on weaker selling
            else:
                f["delta_divergence"] = 0.0
        else:
            f["cvd_slope"] = f["delta_divergence"] = 0.0

        if has_flow:
            sb, ss = int(last.stacked_buy or 0), int(last.stacked_sell or 0)
            f["stacked_imbalance"] = 1.0 if sb >= bn.stacked_levels and sb > ss else -1.0 if ss >= bn.stacked_levels and ss > sb else 0.0
            f["absorption"] = 1.0 if last.absorb_bull and not last.absorb_bear else -1.0 if last.absorb_bear and not last.absorb_bull else 0.0
            f["poc_acceptance"] = _sign(last.close - last.poc) if not math.isnan(last.poc) else 0.0
        else:
            f["stacked_imbalance"] = f["absorption"] = f["poc_acceptance"] = 0.0
        f["book_imbalance"] = _clip(last.book_imb / 0.3) if not pd.isna(last.get("book_imb", np.nan)) else 0.0

        oi_chg, px_chg = last.get("oi_chg", np.nan), last.close - last.open
        if pd.isna(oi_chg) or oi_chg == 0 or px_chg == 0:
            f["oi_buildup"] = 0.0
        elif oi_chg > 0:
            f["oi_buildup"] = 1.0 if px_chg > 0 else -1.0  # long build-up / short build-up
        else:
            f["oi_buildup"] = 0.5 if px_chg > 0 else -0.5  # short covering / long unwinding
        return f

    def score(self, f: dict[str, float]) -> float:
        w = self.bn.weights
        total = sum(abs(v) for v in w.values())
        return sum(w[k] * f.get(k, 0.0) for k in w) / total if total else 0.0

    # ------------------------------------------------------------------ decision
    def on_candles(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.min_candles():
            return Signal(Action.HOLD, reason="warming up")
        f = self.factors(df)
        last = df.iloc[-1]
        a = atr(df[["high", "low", "close"]].astype(float), 14).iloc[-1]
        return self.decide(f, float(last.close), float(last.low), float(last.high), bool(last.get("has_flow", False)), float(a))

    def decide(self, f: dict[str, float], close: float, low: float, high: float, has_flow: bool, a: float) -> Signal:
        """Turn factor scores for the latest candle into an action (separate from `factors` so it can be re-run cheaply)."""
        bn = self.bn
        s = self.score(f)
        self.last_factors, self.last_score = f, s
        top = ", ".join(f"{k} {v:+.1f}" for k, v in sorted(f.items(), key=lambda kv: -abs(kv[1] * bn.weights[kv[0]]))[:5] if v)
        reason = f"score {s:+.2f} [{top}]"

        if s >= bn.entry_threshold or s <= -bn.entry_threshold:
            d = 1 if s > 0 else -1
            blocked = [k for k in bn.required_filters if f.get(k, 0.0) * d <= 0]
            if bn.require_orderflow and not has_flow:
                blocked.append("no live order flow")
            if math.isnan(a) or a < bn.min_atr_points:
                blocked.append(f"ATR {a:.0f} < {bn.min_atr_points:.0f}")
            if not blocked:
                extreme = low if d > 0 else high
                stop = abs(close - extreme) + bn.stop_buffer_atr * a
                stop = max(stop, bn.min_stop_points)
                if stop <= bn.max_stop_points:
                    return Signal(Action.BUY if d > 0 else Action.SHORT, stop, stop * bn.reward_risk, reason)
                reason += f" | stop {stop:.0f} > max {bn.max_stop_points:.0f}"
            else:
                reason += " | blocked: " + ", ".join(blocked)
            # A strong score that can't be traded still argues against the opposite position.
            return Signal(Action.EXIT_SHORT if d > 0 else Action.EXIT_LONG, reason=reason)
        if s <= -bn.exit_threshold:
            return Signal(Action.EXIT_LONG, reason=reason)
        if s >= bn.exit_threshold:
            return Signal(Action.EXIT_SHORT, reason=reason)
        return Signal(Action.HOLD, reason=reason)
