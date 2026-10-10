"""Configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, time
from pathlib import Path

import yaml


@dataclass
class StrategyConfig:
    name: str = "ema_rsi"
    fast_ema: int = 9
    slow_ema: int = 21
    rsi_period: int = 14
    rsi_buy_min: float = 50.0
    rsi_buy_max: float = 70.0
    rsi_sell_min: float = 30.0
    rsi_sell_max: float = 50.0
    atr_period: int = 14
    atr_stop_mult: float = 1.5
    reward_risk: float = 2.0
    allow_short: bool = False


@dataclass
class RiskConfig:
    capital: float = 100_000.0
    risk_per_trade_pct: float = 1.0
    max_position_value_pct: float = 20.0
    max_open_positions: int = 3
    max_daily_loss_pct: float = 2.0
    max_trades_per_day: int = 10
    max_orders_per_second: float = 5.0


@dataclass
class SessionConfig:
    market_open: time = time(9, 15)
    market_close: time = time(15, 30)
    # Don't take signals from the opening auction noise.
    entry_start: time = time(9, 30)
    no_new_entries_after: time = time(15, 0)
    # Brokers auto square-off MIS around 15:20-15:25 (with a fee); beat them.
    square_off_at: time = time(15, 12)
    holidays: list[date] = field(default_factory=list)


@dataclass
class ExecutionConfig:
    exchange: str = "NSE"
    product: str = "MIS"
    # Entries/exits go out as marketable LIMIT orders this far through the LTP.
    limit_buffer_pct: float = 0.3
    # Protective stops go out as SL (stop-limit) with this limit buffer past the trigger.
    stop_limit_buffer_pct: float = 1.0
    tick_size: float = 0.05
    fill_timeout_seconds: int = 20
    order_tag: str = "algobot"
    # Exchange-assigned algo ID, if your broker requires one under SEBI's retail algo framework.
    algo_id: str = ""


@dataclass
class CostConfig:
    """Approximate intraday equity charges (Zerodha-style). Check your broker's current schedule."""

    brokerage_pct: float = 0.03
    brokerage_cap: float = 20.0
    stt_sell_pct: float = 0.025
    exchange_txn_pct: float = 0.00297
    sebi_pct: float = 0.0001
    stamp_buy_pct: float = 0.003
    gst_pct: float = 18.0
    slippage_pct: float = 0.05


@dataclass
class Config:
    mode: str = "paper"  # paper | live
    broker: str = "zerodha"
    data_source: str = "yfinance"  # used by paper mode when no broker credentials
    symbols: list[str] = field(default_factory=lambda: ["RELIANCE", "INFY", "HDFCBANK"])
    interval_minutes: int = 5
    lookback_candles: int = 100
    poll_seconds: int = 15
    log_dir: str = "logs"
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    costs: CostConfig = field(default_factory=CostConfig)

    def validate(self) -> None:
        if self.mode not in ("paper", "live"):
            raise ValueError(f"mode must be 'paper' or 'live', got {self.mode!r}")
        if self.interval_minutes not in (1, 3, 5, 10, 15, 30, 60):
            raise ValueError("interval_minutes must be one of 1, 3, 5, 10, 15, 30, 60")
        if not self.symbols:
            raise ValueError("symbols must not be empty")
        r = self.risk
        if not 0 < r.risk_per_trade_pct <= 5:
            raise ValueError("risk_per_trade_pct must be in (0, 5]")
        if not 0 < r.max_daily_loss_pct <= 20:
            raise ValueError("max_daily_loss_pct must be in (0, 20]")
        # SEBI's retail algo framework requires exchange registration above 10 orders/sec.
        if not 0 < r.max_orders_per_second < 10:
            raise ValueError("max_orders_per_second must be below 10 (SEBI retail algo threshold)")
        s = self.strategy
        if s.fast_ema >= s.slow_ema:
            raise ValueError("fast_ema must be smaller than slow_ema")
        ss = self.session
        if not (ss.market_open <= ss.entry_start < ss.no_new_entries_after < ss.square_off_at < ss.market_close):
            raise ValueError("session times must satisfy open <= entry_start < no_new_entries_after < square_off_at < close")


def _parse_time(v) -> time:
    if isinstance(v, time):
        return v
    if isinstance(v, int):  # YAML parses 09:15 as sexagesimal minutes
        return time(v // 60, v % 60)
    h, m = str(v).split(":")
    return time(int(h), int(m))


def _parse_date(v) -> date:
    return v if isinstance(v, date) else date.fromisoformat(str(v))


def _build(cls, raw: dict | None):
    raw = raw or {}
    known = cls.__dataclass_fields__
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | os.PathLike | None) -> Config:
    raw: dict = {}
    if path:
        raw = yaml.safe_load(Path(path).read_text()) or {}

    session_raw = dict(raw.pop("session", {}) or {})
    for k in ("market_open", "market_close", "entry_start", "no_new_entries_after", "square_off_at"):
        if k in session_raw:
            session_raw[k] = _parse_time(session_raw[k])
    if "holidays" in session_raw:
        session_raw["holidays"] = [_parse_date(d) for d in session_raw["holidays"] or []]

    cfg = Config(
        strategy=_build(StrategyConfig, raw.pop("strategy", None)),
        risk=_build(RiskConfig, raw.pop("risk", None)),
        session=_build(SessionConfig, session_raw),
        execution=_build(ExecutionConfig, raw.pop("execution", None)),
        costs=_build(CostConfig, raw.pop("costs", None)),
    )
    for k, v in raw.items():
        if k not in Config.__dataclass_fields__:
            raise ValueError(f"Unknown config key: {k}")
        setattr(cfg, k, v)
    cfg.symbols = [s.strip().upper() for s in cfg.symbols]
    cfg.validate()
    return cfg
