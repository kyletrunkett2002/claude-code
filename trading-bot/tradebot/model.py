"""Core data types shared across the framework."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Signal(Enum):
    """What a strategy wants to do after seeing the latest candle."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class Candle:
    """A single OHLCV bar.

    ``timestamp`` is the bar's open time in milliseconds since the Unix epoch,
    matching the convention used by most exchange APIs.
    """

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_binance(cls, row: list) -> "Candle":
        """Build a Candle from a Binance kline row.

        Binance returns rows shaped like::

            [openTime, open, high, low, close, volume, closeTime, ...]
        """
        return cls(
            timestamp=int(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )


@dataclass
class Trade:
    """A filled order recorded by the broker."""

    timestamp: int
    side: Signal          # BUY or SELL
    price: float          # fill price after slippage
    quantity: float       # units of the base asset
    fee: float            # quote-currency fee paid
    equity_after: float   # total account equity right after the fill
