"""SuperTrend — a popular ATR-based trend follower.

Go long when the SuperTrend direction flips up, exit/short when it flips down.
Because the band is built from ATR, it automatically gives trends more room in
volatile markets and tightens in calm ones — which is why it's a trader
favourite for riding sustained moves.
"""

from __future__ import annotations

from typing import List

from ..indicators import supertrend
from ..model import Candle, Signal
from ..strategy import Strategy


class SuperTrend(Strategy):
    name = "supertrend"

    def __init__(self, period: int = 10, multiplier: float = 3.0):
        self.period = period
        self.multiplier = multiplier

    def warmup(self) -> int:
        return self.period + 2

    def evaluate(self, history: List[Candle]) -> Signal:
        highs = [c.high for c in history]
        lows = [c.low for c in history]
        closes = [c.close for c in history]
        _, direction = supertrend(highs, lows, closes, self.period, self.multiplier)
        if direction[-1] is None or direction[-2] is None:
            return Signal.HOLD
        if direction[-2] == -1 and direction[-1] == 1:
            return Signal.BUY
        if direction[-2] == 1 and direction[-1] == -1:
            return Signal.SELL
        return Signal.HOLD
