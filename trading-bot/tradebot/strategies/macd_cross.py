"""MACD crossover — momentum via the difference of two EMAs.

Go long when the MACD line crosses above its signal line (momentum turning up),
exit when it crosses below. Smoother than a raw price crossover because both
inputs are already averaged.
"""

from __future__ import annotations

from typing import List

from ..indicators import macd
from ..model import Candle, Signal
from ..strategy import Strategy


class MacdCross(Strategy):
    name = "macd_cross"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        if fast >= slow:
            raise ValueError("fast must be shorter than slow")
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def warmup(self) -> int:
        return self.slow + self.signal + 1

    def evaluate(self, history: List[Candle]) -> Signal:
        closes = [c.close for c in history]
        _, _, hist = macd(closes, self.fast, self.slow, self.signal)
        if hist[-1] is None or hist[-2] is None:
            return Signal.HOLD
        if hist[-2] <= 0 < hist[-1]:
            return Signal.BUY
        if hist[-2] >= 0 > hist[-1]:
            return Signal.SELL
        return Signal.HOLD
