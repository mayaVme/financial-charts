"""Wiring for BankNifty paper/live sessions and tick-replay backtests."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from ..brokers.paper import PaperBroker
from ..config import Config
from ..market_hours import now_ist
from .contracts import Contracts, resolve_contracts
from .engine import BankNiftyEngine
from .footprint import FootprintAggregator
from .ticks import DhanTickFeed, TickRecorder, load_ticks

log = logging.getLogger(__name__)
INDEX_SYMBOL = "NIFTY BANK"


def run_session(cfg: Config, live: bool) -> BankNiftyEngine:
    from ..brokers.dhan import DhanBroker, scrip_master

    bn = cfg.banknifty
    dhan = DhanBroker.from_env(exchange="NSE", product=cfg.execution.product, tag=cfg.execution.order_tag, cache_dir=cfg.log_dir)
    today = now_ist().date()
    contracts = resolve_contracts(scrip_master(cfg.log_dir), bn.underlying, today, bn.skip_expiry_day_options)
    dhan.register(INDEX_SYMBOL, bn.index_security_id, "IDX_I", "INDEX")
    dhan.register(contracts.future_symbol, contracts.future_id, "NSE_FNO", "FUTIDX")

    ref = dhan.ltp([INDEX_SYMBOL]).get(INDEX_SYMBOL) or dhan.ltp([contracts.future_symbol]).get(contracts.future_symbol)
    if not ref:
        raise SystemExit("Could not fetch a BankNifty price from Dhan")
    contracts.options = contracts.options_near(ref, bn.strikes_to_subscribe)
    for o in contracts.options:
        dhan.register(o.symbol, o.security_id, "NSE_FNO", "OPTIDX")
    log.info("Future %s (expiry %s, lot %d); %d options around %.0f expiring %s", contracts.future_symbol,
             contracts.future_expiry, contracts.lot_size, len(contracts.options), ref,
             contracts.options[0].expiry if contracts.options else "-")

    agg = FootprintAggregator(contracts.future_symbol, cfg.interval_minutes, cfg.session, bn)
    try:
        agg.seed(dhan.candles(contracts.future_symbol, cfg.interval_minutes, cfg.lookback_candles))
    except Exception:
        log.exception("Could not load futures history; indicators will warm up from live ticks (takes hours)")

    subs = [(INDEX_SYMBOL, "IDX_I", bn.index_security_id, "ticker"),
            (contracts.future_symbol, "NSE_FNO", contracts.future_id, "full")]
    subs += [(o.symbol, "NSE_FNO", o.security_id, "quote") for o in contracts.options]
    feed = DhanTickFeed(os.environ["DHAN_CLIENT_ID"], os.environ["DHAN_ACCESS_TOKEN"], subs)
    recorder = None
    if bn.record_ticks:
        recorder = TickRecorder(bn.tick_dir)
        contracts.save(Path(bn.tick_dir) / f"{today.isoformat()}.contracts.json")

    broker = dhan if live else PaperBroker(agg, cfg.risk.capital, cfg.costs.slippage_pct, ltp_cache_seconds=0)
    engine = BankNiftyEngine(cfg, broker, contracts, agg, INDEX_SYMBOL)
    feed.start()
    try:
        engine.run_session(feed.queue, recorder)
    finally:
        feed.stop()
        if recorder:
            recorder.close()
    return engine


def replay(cfg: Config, tick_files: list[str | Path], contracts_file: str | Path | None = None) -> BankNiftyEngine:
    """Backtest by replaying recorded ticks through the exact live engine with simulated fills."""
    files = sorted(Path(f) for f in tick_files)
    if not files:
        raise SystemExit("No tick files given")
    state = {"now": None}
    engine = agg = None

    for path in files:
        sidecar = Path(contracts_file) if contracts_file else path.with_name(path.stem + ".contracts.json")
        if not sidecar.exists():
            raise SystemExit(f"Missing contracts file {sidecar} (written by the recorder next to each tick file)")
        contracts = Contracts.load(sidecar)
        if engine is None:
            agg = FootprintAggregator(contracts.future_symbol, cfg.interval_minutes, cfg.session, cfg.banknifty)
            broker = PaperBroker(agg, cfg.risk.capital, cfg.costs.slippage_pct, ltp_cache_seconds=0)
            engine = BankNiftyEngine(cfg, broker, contracts, agg, INDEX_SYMBOL, clock=lambda: state["now"], sleep=lambda s: None)
        else:  # new day: possibly a rolled-over future or a different strike band
            agg.signal_symbol = contracts.future_symbol
            engine.cfg.symbols = [contracts.future_symbol]
            engine.contracts = contracts

        last_sec, last_ts = None, None
        for t in load_ticks(path):
            state["now"] = last_ts = t.ts
            agg.on_tick(t)
            sec = t.ts.replace(microsecond=0)
            if sec != last_sec and engine.calendar.is_open(t.ts):
                last_sec = sec
                engine.step(t.ts)
        if last_ts is not None:
            sq = cfg.session.square_off_at
            end = last_ts.replace(hour=sq.hour, minute=sq.minute, second=1, microsecond=0)
            state["now"] = max(end, last_ts)
            engine.step(state["now"])
            engine.square_off_all("end of data")
    return engine


def summarize(trades: list[dict], capital: float) -> dict:
    nets = np.array([t["net"] for t in trades], dtype=float)
    if not len(nets):
        return {"trades": 0}
    equity = np.cumsum(nets)
    dd = float((equity - np.maximum.accumulate(np.r_[0.0, equity])[1:]).min())
    wins, losses = nets[nets > 0], nets[nets <= 0]
    return {
        "trades": int(len(nets)),
        "win_rate_pct": round(100 * len(wins) / len(nets), 1),
        "net_pnl": round(float(nets.sum()), 2),
        "return_pct": round(100 * float(nets.sum()) / capital, 2),
        "total_costs": round(float(sum(t["costs"] for t in trades)), 2),
        "avg_win": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else None,
        "max_drawdown": round(min(dd, 0.0), 2),
        "exit_reasons": {r: sum(1 for t in trades if t["reason"] == r) for r in sorted({t["reason"] for t in trades})},
    }
