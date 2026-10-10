"""Tick data: Dhan WebSocket feed, trade-side classification, recording and replay."""

from __future__ import annotations

import csv
import logging
import queue
import threading
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

from ..market_hours import IST, now_ist

log = logging.getLogger(__name__)


@dataclass
class Tick:
    ts: datetime
    symbol: str
    ltp: float
    ltq: int = 0
    volume: int = 0  # cumulative day volume
    bid: float = 0.0
    ask: float = 0.0
    bid_qty: int = 0  # sum of top-N bid quantities
    ask_qty: int = 0
    total_buy_qty: int = 0  # all pending buy quantity on the exchange
    total_sell_qty: int = 0
    oi: int = 0


TICK_FIELDS = [f.name for f in fields(Tick)]


class TradeClassifier:
    """Turns successive snapshot packets into (traded_qty, aggressor_side).

    Dhan's feed sends snapshots, not individual trades, so the traded quantity is the change in
    cumulative volume. The side uses the quote rule against the *previous* best bid/ask (trade at
    or above the ask = buyer-initiated, at or below the bid = seller-initiated), falling back to
    the tick rule when the trade printed inside the spread.
    """

    def __init__(self):
        self._prev: dict[str, Tick] = {}
        self._last_side: dict[str, int] = {}

    def classify(self, t: Tick) -> tuple[int, int]:
        prev = self._prev.get(t.symbol)
        self._prev[t.symbol] = t
        if prev is None or t.volume < prev.volume or t.ts.date() != prev.ts.date():
            return 0, 0  # first packet of the day / volume reset
        qty = t.volume - prev.volume
        if qty <= 0:
            return 0, 0
        bid, ask = (prev.bid, prev.ask) if prev.bid and prev.ask else (t.bid, t.ask)
        if ask and t.ltp >= ask:
            side = 1
        elif bid and t.ltp <= bid:
            side = -1
        elif t.ltp > prev.ltp:
            side = 1
        elif t.ltp < prev.ltp:
            side = -1
        else:
            side = self._last_side.get(t.symbol, 0)
        self._last_side[t.symbol] = side
        return qty, side


# ---------------------------------------------------------------------------- recording / replay
class TickRecorder:
    """Appends ticks to data/ticks/YYYY-MM-DD.csv so sessions can be replayed later."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer = None
        self._day = None
        self._n = 0

    def write(self, t: Tick) -> None:
        day = t.ts.date()
        if day != self._day:
            self.close()
            path = self.dir / f"{day.isoformat()}.csv"
            new = not path.exists()
            self._file = path.open("a", newline="")
            self._writer = csv.writer(self._file)
            if new:
                self._writer.writerow(TICK_FIELDS)
            self._day = day
        row = asdict(t)
        row["ts"] = t.ts.isoformat()
        self._writer.writerow([row[k] for k in TICK_FIELDS])
        self._n += 1
        if self._n % 500 == 0:
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


def load_ticks(path: str | Path):
    """Yields Ticks from a recorded CSV (or a vendor file converted to the same columns)."""
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            ts = datetime.fromisoformat(row["ts"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=IST)
            yield Tick(
                ts=ts.astimezone(IST),
                symbol=row["symbol"],
                ltp=float(row["ltp"]),
                **{k: (float(row[k]) if k in ("bid", "ask") else int(float(row[k])))
                   for k in TICK_FIELDS if k not in ("ts", "symbol", "ltp") and row.get(k) not in (None, "")},
            )


# ---------------------------------------------------------------------------- live Dhan feed
SEGMENT_CODES = {"IDX_I": 0, "NSE_EQ": 1, "NSE_FNO": 2}
_MODES = {"ticker": 15, "quote": 17, "full": 21}


class DhanTickFeed:
    """Streams Dhan v2 market-feed packets on a background thread into a queue of Ticks.

    subscriptions: list of (symbol, segment, security_id, mode) with mode in ticker|quote|full.
    Reconnects with exponential backoff if the socket drops.
    """

    def __init__(self, client_id: str, access_token: str, subscriptions, depth_levels: int = 5):
        self.client_id, self.token = client_id, access_token
        self.instruments = [(SEGMENT_CODES[seg], str(sid), _MODES[mode]) for _, seg, sid, mode in subscriptions]
        self.symbol_of = {(SEGMENT_CODES[seg], int(sid)): sym for sym, seg, sid, _ in subscriptions}
        self.depth_levels = depth_levels
        self.queue: queue.Queue[Tick] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="dhan-feed", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        from dhanhq import DhanContext, MarketFeed

        backoff = 1
        while not self._stop.is_set():
            try:
                feed = MarketFeed(DhanContext(self.client_id, self.token), self.instruments, "v2")
                log.info("Connecting Dhan market feed (%d instruments)", len(self.instruments))
                while not self._stop.is_set():
                    feed.run_forever()
                    msg = feed.get_data()
                    self.connected = True
                    tick = self.parse(msg)
                    if tick:
                        self.queue.put(tick)
                    backoff = 1
            except Exception as e:  # network drops, auth errors
                self.connected = False
                log.warning("Market feed error: %s; reconnecting in %ds", e, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def parse(self, msg) -> Tick | None:
        if not isinstance(msg, dict) or "LTP" not in msg:
            return None
        sym = self.symbol_of.get((int(msg.get("exchange_segment", -1)), int(msg.get("security_id", -1))))
        if sym is None:
            return None
        t = Tick(ts=now_ist(), symbol=sym, ltp=float(msg["LTP"]))
        t.ltq = int(msg.get("LTQ") or 0)
        t.volume = int(msg.get("volume") or 0)
        t.total_buy_qty = int(msg.get("total_buy_quantity") or 0)
        t.total_sell_qty = int(msg.get("total_sell_quantity") or 0)
        t.oi = int(msg.get("OI") or 0)
        depth = msg.get("depth") or []
        if depth:
            t.bid, t.ask = float(depth[0]["bid_price"]), float(depth[0]["ask_price"])
            top = depth[: self.depth_levels]
            t.bid_qty = sum(int(d["bid_quantity"]) for d in top)
            t.ask_qty = sum(int(d["ask_quantity"]) for d in top)
        return t
