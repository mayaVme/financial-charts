# NSE Intraday Trading Bot

An automated intraday trading bot for Indian equities (NSE cash market). It ships with a
backtester, a paper-trading mode that uses live prices with simulated orders, and live trading
through **Dhan** (the default) or **Zerodha Kite Connect**.

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
cp .env.example .env                    # broker credentials (only for broker data / live)
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
free but can lag the exchange by a few minutes. For real-time prices, set `data_source: broker`
(this needs your Dhan credentials and places no real orders). Leave it running through market hours;
it sleeps outside them.

## 3. Live trading (Dhan)

1. In Dhan, open <https://web.dhan.co> → **My Profile → Access DhanHQ APIs**.
   - Generate an **access token**. It is valid for 24 hours, so you'll need a new one each day.
   - Note your **Client ID**.
   - Register the **static IP** of the machine that will run the bot. Dhan rejects API orders
     from any other IP.
2. Put `DHAN_CLIENT_ID` and `DHAN_ACCESS_TOKEN` in `.env`. Running `python -m bot login`
   repeats these steps.
3. Set `mode: live` in `config.yaml` (`broker: dhan` is already the default), then start trading:
   ```bash
   python -m bot live --i-accept-the-risk
   ```

How the bot talks to Dhan:
- **Instrument IDs:** Dhan identifies stocks by security ID, not by symbol. The bot downloads
  Dhan's scrip master once a day and looks the IDs up. If a symbol can't be found, pin its ID
  under `dhan_security_ids` in the config.
- **Order types:** intraday orders are sent with the `INTRADAY` product. Entries and exits are
  `LIMIT` orders, and protective stops are `STOP_LOSS` (stop-limit) orders.
- **Quote limit:** Dhan's quote API allows about one request per second, and the bot paces its
  price checks to match.

### Zerodha instead

Set `broker: zerodha`. Put `KITE_API_KEY` and `KITE_API_SECRET` in `.env`, then run
`python -m bot login` each morning to get `KITE_ACCESS_TOKEN`. Kite Connect is a paid
subscription.

Run it on an always-on machine or a cloud VM with a stable connection and a static IP, not a
laptop that sleeps.

### Regulatory notes (India)

SEBI's retail algo-trading framework (circular of Feb 2025, phased in by brokers since late
2025) applies to API trading. In practice:

- API orders must come from a **static IP registered with your broker**.
- Strategies that place more than 10 orders per second must be registered with the exchange.
  This bot is capped at 5 per second.
- Orders are tagged as algo orders. With Zerodha, if you're issued an exchange algo ID, set
  `execution.algo_id`.

Brokers' implementations and deadlines differ, so check your broker's current requirements before
going live.

## BankNifty order-flow bot (Dhan)

A separate intraday bot for BankNifty. It reads order flow from the current-month BankNifty
**future** (the index itself has no volume). It then trades either **ATM options** (the
default: buy a CE on bullish signals, a PE on bearish ones) or the **future** itself.

```bash
cp banknifty.example.yaml banknifty.yaml
python -m bot -c banknifty.yaml banknifty paper                       # live Dhan data, simulated fills
python -m bot -c banknifty.yaml banknifty backtest --csv futures_5m.csv     # technical half, on candles
python -m bot -c banknifty.yaml banknifty replay --ticks data/ticks/*.csv   # full model, on recorded ticks
python -m bot -c banknifty.yaml banknifty live --i-accept-the-risk    # real money (needs mode: live)
```

Start it each morning before 09:15 with a fresh Dhan token. It trades one session and exits
after the close.

### How a decision is made

Ticks from Dhan's WebSocket are built into **5-minute footprint candles**. Each candle records
volume traded at the bid and at the ask for every 5-point price row. On every closed candle,
14 factors each score from -1 (bearish) to +1 (bullish), and the confluence score is their
weighted average:

| Group | Factors |
|---|---|
| Order flow | bar delta, CVD slope, stacked diagonal imbalances (300%, 3+ rows), absorption at the bar's extremes, delta divergence vs new highs/lows, close vs footprint POC, 5-level order-book imbalance, OI build-up/unwinding |
| Technicals | session VWAP, EMA 9/21, Supertrend (10, 3), CPR from the previous day, 15-minute opening-range breakout, RSI |

How the score is used:

- **Entry:** the bot enters when |score| ≥ 0.45 and all of these hold:
  - the required filters agree (VWAP by default);
  - the candle has live order-flow data;
  - ATR shows the market is actually moving;
  - the stop fits within 40–150 points.
- **Stop:** beyond the signal candle's extreme plus 0.25 × ATR, placed at Dhan as a stop-loss
  order.
- **Target:** 2R. The stop moves to breakeven at +1R.
- **Exit on reversal:** an opposite score of 0.35 or more closes the trade.
- **Options mode:** stops and targets are converted from underlying points to premium points
  using delta (0.5 by default). The premium stop is capped at 35%.
- **Guards:**
  - 1 position at a time and 4 trades a day;
  - a 15-minute cooldown after a loss;
  - no entries while the tick feed is stale;
  - the 2% daily-loss kill switch;
  - square-off at 15:12.

Every candle's factor scores are written to `logs/banknifty-signals.csv`, and every trade to
`logs/banknifty-trades.csv`, so you can see exactly why the bot did what it did. Weights,
thresholds and filters are all set in `banknifty.yaml`.

### Backtest results (Jul 6 – Oct 9, 2026)

The bot was tested on 68 sessions of real 5-minute BankNifty continuous-futures candles
(5,196 bars from TradingView). Each data chunk was checked against the source's own bar count,
total volume and high/low. Candles can't reproduce order flow, so **only the technical half of
the model was tested**: VWAP, EMA, Supertrend, CPR, opening range and RSI. The entry rules,
stops, breakeven, cooldown, kill switch and square-off were the same as live. Results are per
1 lot (35), with 2 points of slippage per side and futures STT at 0.02%.

| | Result |
|---|---|
| Trades / win rate | 229 / 21% |
| Gross points (before costs) | **–2,105** (–1,189 even with zero slippage) |
| Net per lot after charges | **–₹2.05L** (–₹3.23L with STT at 0.05%) |
| Profit factor / max drawdown | 0.48 / –₹2.07L |
| Every month | July, August, September and October all lost |
| Random-direction control | strategy at the 29th percentile, so **no better than a coin flip** |
| Parameter sweep | **0 of 48** threshold/R:R/breakeven/stop combinations profitable after costs |
| Walk-forward | best settings on Jul–Aug lost –₹52k on Sep–Oct |
| ATM options estimate (delta 0.5, no theta) | –₹83k per lot, and the real figure would be worse |

**What this means:** the technical signals have no edge on BankNifty at this timeframe in this
period. Round-trip costs of about 16–20 points per lot make any no-edge strategy lose steadily.
The order-flow factors are still untested, because no historical tick data exists. Don't trade
this live unless paper-traded tick recordings (`banknifty replay`) show a clear, cost-adjusted
edge over many weeks.

Rerun it on your own candles:

```bash
python -m bot -c banknifty.yaml banknifty backtest --csv BANKNIFTY_FUT_5m.csv --trades-out trades.csv
```

### Limits you should know about

- **No historical order-flow data:** Dhan doesn't sell historical tick data, so the footprint
  can't be backtested on past years. The bot **records every tick** to `data/ticks/` while it
  runs (paper or live), and `banknifty replay` backtests on those recordings using the exact
  live engine. Paper trade for several weeks first; that builds your own tick history. Vendor
  tick data converted to the same CSV columns also works.
- **Footprint is approximate:** Dhan's feed sends snapshots, not every individual trade. The
  bot estimates traded volume from changes in cumulative volume, and classifies it as buying
  or selling against the previous best bid/ask. It's a good approximation, but not
  exchange-grade tick-by-tick data.
- **Starting mid-session:** indicators are warmed up from Dhan's 5-minute futures history, but
  order-flow factors only score candles built from live ticks.
- **Capital:** one BankNifty lot (lot size is read from Dhan's contract list) needs roughly
  ₹2.5L margin as a future. Options need only the premium, which is why options are the
  default. If the risk budget can't cover one lot at the planned stop, the bot skips the trade
  and logs why.
- **Charges:** the STT and exchange charges in the example config are deliberately
  conservative. Check them against your Dhan contract note.
- **No edge is proven:** none of these factors has a verified edge on BankNifty. The default
  weights and thresholds are reasoned starting points, not optimised values. Validate them on
  your own recorded data before trusting them with money.

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
  brokers/           Broker interface, PaperBroker, DhanBroker, ZerodhaBroker
  data/feeds.py      Yahoo Finance / broker / CSV data
  banknifty/         BankNifty order-flow bot: ticks, footprint, confluence strategy, engine, replay
tests/               pytest suite (python -m pytest)
```

## Adding a strategy or broker

- **Strategy:** subclass `Strategy` in `bot/strategies/`, return a `Signal` from
  `on_candles(df)`, and register it in `STRATEGIES`. The backtester and the live engine both
  pick it up.
- **Broker** (Upstox, Angel One, etc.): implement the `Broker` interface in
  `bot/brokers/base.py`: `ltp`, `candles`, `place_order`, `get_order`, `cancel_order`,
  `available_cash`, and `net_positions`.
