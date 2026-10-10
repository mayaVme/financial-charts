# NSE Intraday Trading Bot

An automated intraday trading bot for Indian equities (NSE cash market). It ships with a
backtester, a paper-trading mode that uses live prices with simulated orders, and live trading
through **Zerodha Kite Connect**.

> **Read this first.** Automated trading can lose money fast, and most retail intraday traders
> lose money. The bundled strategy is a simple, well-known template. It is **not** a proven
> money-maker. Backtest it, paper trade it for weeks, and only then risk capital you can afford
> to lose. Nothing here is investment advice.

## What it does

- **Strategy:** EMA 9/21 crossover with an RSI filter on 5-minute candles. Stop = 1.5 × ATR,
  target = 2 × stop. Long-only by default; set `allow_short: true` to also short intraday.
- **Risk controls** (always on):
  - Position size is set so a stopped-out trade loses about 1% of capital.
  - Each position is capped at 20% of capital, and the bot never uses intraday leverage.
  - At most 3 open positions and 10 trades a day.
  - **Daily kill switch:** if the day's realized plus open loss reaches 2% of capital, the bot
    squares off every position and stops for the day.
  - **A broker-side stop order is placed right after every entry.** If the bot crashes or loses
    its connection, your position is still protected.
  - The bot opens no new trades after 15:00 and squares off everything at 15:12, ahead of the
    broker's own MIS auto-square-off and its fee.
  - It uses marketable limit orders instead of raw market orders.
  - Orders are throttled to 5 per second, safely below SEBI's 10 orders/sec retail threshold.
  - At startup the bot skips any symbol that already has a position it didn't open.
- **Session rules:** IST trading hours, weekends, and an NSE holiday list that you maintain.
- **Costs:** P&L and backtests include brokerage, STT, exchange and SEBI fees, stamp duty, GST
  and slippage.
- **Records:** every closed trade is logged to `logs/trades.csv`, and the full log goes to
  `logs/bot.log`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml      # edit symbols, capital, risk, holidays
cp .env.example .env                    # Kite credentials (only for kite data / live)
```

## 1. Backtest

```bash
python -m bot backtest --symbol INFY                 # ~60 days of 5m data from Yahoo Finance
python -m bot backtest --csv data/INFY.csv --trades-out trades.csv
```

The CSV needs these columns: `datetime, open, high, low, close[, volume]`. Times without a
time zone are treated as IST, and any candle size gets resampled to `interval_minutes`.

How fills are simulated: a signal on one candle's close is executed at the next candle's open.
Stops and targets are checked against each candle's high and low. If both are hit in the same
candle, the backtest assumes the stop hit first.

## 2. Paper trade (do this for weeks)

```bash
python -m bot paper
```

This uses live prices and simulated fills. By default prices come from Yahoo Finance, which is
free but can lag the exchange by a few minutes. For real-time prices, set `data_source: kite`
(this needs Kite credentials and places no real orders). Leave it running through market hours;
it sleeps outside them.

## 3. Live trading (Zerodha)

1. Create a Kite Connect app at <https://developers.kite.trade>. The API is a paid
   subscription. Put `KITE_API_KEY` and `KITE_API_SECRET` in `.env`.
2. Every trading morning, generate an access token (tokens expire around 6 AM IST):
   ```bash
   python -m bot login     # follow the prompts, then put KITE_ACCESS_TOKEN in .env
   ```
3. Set `mode: live` in `config.yaml`, then start trading:
   ```bash
   python -m bot live --i-accept-the-risk
   ```

Run it on an always-on machine or a cloud VM with a stable connection, not a laptop that sleeps.

### Regulatory notes (India)

SEBI's retail algo-trading framework (circular of Feb 2025, phased in by brokers since late
2025) applies to API trading. In practice:

- API orders must come from a **static IP registered with your broker**.
- Strategies that place more than 10 orders per second must be registered with the exchange.
  This bot is capped at 5 per second.
- Orders are tagged as algo orders. If your broker issues an exchange algo ID, set
  `execution.algo_id`.

Brokers' implementations and deadlines differ, so check Zerodha's current requirements before
going live.

## Project layout

```
bot/
  __main__.py        CLI: backtest | paper | live | login
  config.py          typed config + validation
  engine.py          live/paper loop: signals, entries with protective stops, exits, kill switch
  backtest.py        bar-by-bar backtester with the same rules and costs
  risk.py            sizing, daily limits, rate limiter, Indian cost model
  market_hours.py    IST session calendar
  indicators.py      EMA, RSI, ATR
  strategies/        Strategy interface + EMA/RSI strategy
  brokers/           Broker interface, PaperBroker, ZerodhaBroker
  data/feeds.py      Yahoo Finance / Kite / CSV data
tests/               pytest suite (python -m pytest)
```

## Adding a strategy or broker

- **Strategy:** subclass `Strategy` in `bot/strategies/`, return a `Signal` from
  `on_candles(df)`, and register it in `STRATEGIES`. The backtester and the live engine both
  pick it up.
- **Broker** (Upstox, Angel One, Dhan, etc.): implement the `Broker` interface in
  `bot/brokers/base.py`: `ltp`, `candles`, `place_order`, `get_order`, `cancel_order`,
  `available_cash`, and `net_positions`.
