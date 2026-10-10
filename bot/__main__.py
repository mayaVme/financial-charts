"""Command-line entry point: python -m bot {backtest,paper,live,login}."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .config import load_config


def _setup_logging(log_dir: str, verbose: bool) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(Path(log_dir) / "bot.log")],
    )


def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def cmd_backtest(args, cfg) -> None:
    from .backtest import backtest
    from .data.feeds import YFinanceFeed, load_csv, resample
    from .strategies import build_strategy

    if args.csv:
        df = resample(load_csv(args.csv), cfg.interval_minutes)
        symbol = args.symbol or Path(args.csv).stem.upper()
    else:
        if not args.symbol:
            raise SystemExit("--symbol is required without --csv")
        df = YFinanceFeed().history(args.symbol, cfg.interval_minutes)
        symbol = args.symbol
    if df.empty:
        raise SystemExit("No data to backtest")

    result = backtest(df, symbol, build_strategy(cfg.strategy), cfg)
    print(f"\nBacktest {symbol} | {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d} | {len(df)} candles of {cfg.interval_minutes}m")
    print(json.dumps(result.summary(), indent=2))
    if args.trades_out:
        import pandas as pd

        pd.DataFrame([{**t.__dict__, "net": t.net} for t in result.trades]).to_csv(args.trades_out, index=False)
        print(f"Trades written to {args.trades_out}")


def _build_broker(cfg, live: bool):
    from .brokers.zerodha import ZerodhaBroker

    if live:
        return ZerodhaBroker.from_env(exchange=cfg.execution.exchange, product=cfg.execution.product,
                                    tag=cfg.execution.order_tag, algo_id=cfg.execution.algo_id)

    from .brokers.paper import PaperBroker
    from .data.feeds import KiteFeed, YFinanceFeed

    if cfg.data_source == "kite":
        feed = KiteFeed(ZerodhaBroker.from_env(exchange=cfg.execution.exchange))
    elif cfg.data_source == "yfinance":
        feed = YFinanceFeed()
    else:
        raise SystemExit(f"Unknown data_source {cfg.data_source!r} (use 'yfinance' or 'kite')")
    return PaperBroker(feed, cfg.risk.capital, cfg.costs.slippage_pct)


def cmd_run(args, cfg, live: bool) -> None:
    from .engine import TradingEngine
    from .strategies import build_strategy

    if live:
        if cfg.mode != "live" or not args.i_accept_the_risk:
            raise SystemExit(
                "Refusing to trade real money. Set `mode: live` in the config AND pass --i-accept-the-risk.\n"
                "Paper trade first: python -m bot paper"
            )
        print("*** LIVE TRADING WITH REAL MONEY. Ctrl+C to stop (broker-side stops stay in place). ***")
    else:
        cfg.mode = "paper"
    engine = TradingEngine(cfg, _build_broker(cfg, live), build_strategy(cfg.strategy))
    try:
        engine.run()
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Interrupted. Open positions: %s", list(engine.positions) or "none")
        if engine.positions:
            print("Positions are still open with stop orders at the broker. Square them off manually if needed.")


def cmd_login(args, cfg) -> None:
    from .brokers.zerodha import exchange_request_token, login_url

    key, secret = os.environ.get("KITE_API_KEY"), os.environ.get("KITE_API_SECRET")
    if not key or not secret:
        raise SystemExit("Set KITE_API_KEY and KITE_API_SECRET in your environment or .env")
    print("1) Open this URL, log in, and copy `request_token` from the redirect URL:\n   " + login_url(key))
    token = args.request_token or input("2) Paste request_token: ").strip()
    access = exchange_request_token(key, secret, token)
    print(f"\nAdd this to .env (valid until ~6 AM IST tomorrow):\nKITE_ACCESS_TOKEN={access}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bot", description="Intraday trading bot for NSE")
    ap.add_argument("-c", "--config", default="config.yaml" if Path("config.yaml").exists() else None)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backtest", help="Backtest on CSV or recent Yahoo Finance data")
    b.add_argument("--csv", help="OHLCV CSV file")
    b.add_argument("--symbol", help="NSE symbol, e.g. INFY")
    b.add_argument("--trades-out", help="Write trades to this CSV")

    sub.add_parser("paper", help="Paper trade with live prices, simulated orders")
    lv = sub.add_parser("live", help="Trade real money via Zerodha Kite Connect")
    lv.add_argument("--i-accept-the-risk", action="store_true")
    lg = sub.add_parser("login", help="Generate a Kite access token for today")
    lg.add_argument("--request-token")

    args = ap.parse_args(argv)
    _load_dotenv()
    cfg = load_config(args.config)
    _setup_logging(cfg.log_dir, args.verbose)

    if args.cmd == "backtest":
        cmd_backtest(args, cfg)
    elif args.cmd == "paper":
        cmd_run(args, cfg, live=False)
    elif args.cmd == "live":
        cmd_run(args, cfg, live=True)
    elif args.cmd == "login":
        cmd_login(args, cfg)


if __name__ == "__main__":
    main()
