"""Keltner Channel strategy with selectable mode.

Keltner Channels are an EMA midline wrapped by ATR-scaled bands. Like the
Bollinger strategy, the same indicator supports two opposite ideas:

- mode="breakout" (default): a close above the upper band signals a volatility
  expansion to the upside — go long; a close below the lower band — exit/short.
  Keltner breakouts are a classic trend-entry because ATR bands lag less than
  standard-deviation bands.
- mode="reversion": fade the bands instead — buy the lower, sell the upper.

Which one works is empirical; backtest before believing either.
"""

from __future__ import annotations

from typing import List

from ..indicators import keltner
from ..model import Candle, Signal
from ..strategy import Strategy


class Keltner(Strategy):
    name = "keltner"

    def __init__(self, period: int = 20, atr_period: int = 10,
                 multiplier: float = 2.0, mode: str = "breakout"):
        if mode not in ("breakout", "reversion"):
            raise ValueError("mode must be 'breakout' or 'reversion'")
        self.period = period
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.mode = mode

    def warmup(self) -> int:
        return max(self.period, self.atr_period) + 2

    def evaluate(self, history: List[Candle]) -> Signal:
        highs = [c.high for c in history]
        lows = [c.low for c in history]
        closes = [c.close for c in history]
        lower, _, upper = keltner(highs, lows, closes, self.period,
                                  self.atr_period, self.multiplier)
        if lower[-1] is None or upper[-1] is None:
            return Signal.HOLD
        price = closes[-1]
        above = price > upper[-1]
        below = price < lower[-1]
        if self.mode == "breakout":
            if above:
                return Signal.BUY
            if below:
                return Signal.SELL
        else:  # reversion
            if below:
                return Signal.BUY
            if above:
                return Signal.SELL
        return Signal.HOLD
