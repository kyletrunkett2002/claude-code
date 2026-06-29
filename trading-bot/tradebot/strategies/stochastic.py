"""Stochastic oscillator strategy.

Buy when %K crosses up through %D while in oversold territory (momentum turning
up from a low), sell when %K crosses down through %D while overbought. The
crossover requirement makes it less twitchy than a raw threshold cross.
"""

from __future__ import annotations

from typing import List

from ..indicators import stochastic
from ..model import Candle, Signal
from ..strategy import Strategy


class Stochastic(Strategy):
    name = "stochastic"

    def __init__(self, k_period: int = 14, d_period: int = 3,
                 oversold: float = 20.0, overbought: float = 80.0):
        if not 0 < oversold < overbought < 100:
            raise ValueError("require 0 < oversold < overbought < 100")
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold
        self.overbought = overbought

    def warmup(self) -> int:
        return self.k_period + self.d_period + 1

    def evaluate(self, history: List[Candle]) -> Signal:
        if len(history) < self.warmup():
            return Signal.HOLD
        highs = [c.high for c in history]
        lows = [c.low for c in history]
        closes = [c.close for c in history]
        k, d = stochastic(highs, lows, closes, self.k_period, self.d_period)
        if None in (k[-1], k[-2], d[-1], d[-2]):
            return Signal.HOLD
        crossed_up = k[-2] <= d[-2] and k[-1] > d[-1]
        crossed_down = k[-2] >= d[-2] and k[-1] < d[-1]
        if crossed_up and k[-1] < self.overbought and d[-2] <= self.oversold + 15:
            return Signal.BUY
        if crossed_down and k[-1] > self.oversold and d[-2] >= self.overbought - 15:
            return Signal.SELL
        return Signal.HOLD
