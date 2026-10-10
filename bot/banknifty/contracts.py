"""Resolve the BankNifty future and option contracts to trade from Dhan's scrip master."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd


@dataclass
class OptionContract:
    symbol: str
    security_id: str
    strike: float
    opt_type: str  # CE | PE
    expiry: str  # ISO date


@dataclass
class Contracts:
    underlying: str
    future_symbol: str
    future_id: str
    future_expiry: str
    lot_size: int
    strike_step: float
    options: list[OptionContract]

    def pick_option(self, underlying_price: float, opt_type: str, offset: int = 0) -> OptionContract | None:
        """ATM (offset 0) or offset strikes OTM (positive) / ITM (negative) for CE or PE."""
        atm = round(underlying_price / self.strike_step) * self.strike_step
        strike = atm + offset * self.strike_step * (1 if opt_type == "CE" else -1)
        return next((o for o in self.options if o.opt_type == opt_type and abs(o.strike - strike) < 1e-6), None)

    def options_near(self, price: float, n: int) -> list[OptionContract]:
        atm = round(price / self.strike_step) * self.strike_step
        return [o for o in self.options if abs(o.strike - atm) <= n * self.strike_step + 1e-6]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Contracts":
        d = json.loads(Path(path).read_text())
        d["options"] = [OptionContract(**o) for o in d["options"]]
        return cls(**d)


def resolve_contracts(master: pd.DataFrame, underlying: str, today: date, skip_expiry_day_options: bool = True) -> Contracts:
    m = master
    need = {"SEM_EXM_EXCH_ID", "SEM_SEGMENT", "SEM_SMST_SECURITY_ID", "SEM_INSTRUMENT_NAME", "SEM_TRADING_SYMBOL",
            "SEM_LOT_UNITS", "SEM_EXPIRY_DATE", "SEM_STRIKE_PRICE", "SEM_OPTION_TYPE"}
    missing = need - set(m.columns)
    if missing:
        raise ValueError(f"Dhan scrip master is missing columns {sorted(missing)}; the format may have changed")
    d = m[(m["SEM_EXM_EXCH_ID"] == "NSE") & (m["SEM_SEGMENT"] == "D") & m["SEM_TRADING_SYMBOL"].str.startswith(f"{underlying}-")].copy()
    d["expiry"] = pd.to_datetime(d["SEM_EXPIRY_DATE"].str[:10]).dt.date

    fut = d[(d["SEM_INSTRUMENT_NAME"] == "FUTIDX") & (d["expiry"] >= today)].sort_values("expiry")
    if fut.empty:
        raise ValueError(f"No live {underlying} futures in the scrip master")
    f = fut.iloc[0]

    opts = d[d["SEM_INSTRUMENT_NAME"] == "OPTIDX"]
    opts = opts[opts["expiry"] > today] if skip_expiry_day_options else opts[opts["expiry"] >= today]
    if opts.empty:
        raise ValueError(f"No live {underlying} options in the scrip master")
    expiry = opts["expiry"].min()
    chain = opts[opts["expiry"] == expiry]
    strikes = sorted(chain["SEM_STRIKE_PRICE"].astype(float).unique())
    diffs = pd.Series(strikes).diff().dropna()
    step = float(diffs.mode().iloc[0]) if not diffs.empty else 100.0

    return Contracts(
        underlying=underlying,
        future_symbol=f["SEM_TRADING_SYMBOL"],
        future_id=str(f["SEM_SMST_SECURITY_ID"]),
        future_expiry=f["expiry"].isoformat(),
        lot_size=int(float(f["SEM_LOT_UNITS"])),
        strike_step=step,
        options=[
            OptionContract(r.SEM_TRADING_SYMBOL, str(r.SEM_SMST_SECURITY_ID), float(r.SEM_STRIKE_PRICE), r.SEM_OPTION_TYPE, expiry.isoformat())
            for r in chain.itertuples()
        ],
    )
