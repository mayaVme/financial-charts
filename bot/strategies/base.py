from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Action(str, Enum):
    HOLD = "HOLD"
    BUY = "BUY"  # open long
    SHORT = "SHORT"  # open short (intraday only)
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"


@dataclass
class Signal:
    action: Action
    stop_distance: float = 0.0  # in price units, for entries
    target_distance: float = 0.0
    reason: str = ""


class Strategy:
    """Strategies see only fully-closed candles and return one Signal for the latest one."""

    name = "base"

    def min_candles(self) -> int:
        raise NotImplementedError

    def on_candles(self, df: pd.DataFrame) -> Signal:
        raise NotImplementedError
