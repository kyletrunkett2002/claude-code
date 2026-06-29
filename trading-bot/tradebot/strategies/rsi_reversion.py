"""RSI mean-reversion — a counter-trend strategy.

Buy when RSI drops into oversold territory (the asset has fallen "too far, too
fast" and tends to bounce); sell when it climbs back into overbought territory.
Mean-reversion is the mirror image of trend-following: it wins in choppy,
range-bound markets and gets hurt by strong sustained trends.
"""

from __future__ import annotations

from typing import List

from ..indicators import rsi
from ..model import Candle, Signal
from ..strategy import Strategy


class RsiReversion(Strategy):
    name = "rsi_reversion"

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        if not 0 < oversold < overbought < 100:
            raise ValueError("require 0 < oversold < overbought < 100")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def warmup(self) -> int:
        return self.period + 1

    def evaluate(self, history: List[Candle]) -> Signal:
        closes = [c.close for c in history]
        values = rsi(closes, self.period)
        current = values[-1]
        if current is None:
            return Signal.HOLD
        if current <= self.oversold:
            return Signal.BUY
        if current >= self.overbought:
            return Signal.SELL
        return Signal.HOLD
