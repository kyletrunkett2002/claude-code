"""Simple moving-average crossover — a classic trend-following strategy.

Go long when the fast SMA crosses above the slow SMA (momentum turning up);
exit when it crosses back below. Trend strategies win by catching big moves and
cutting losers early; they lose in choppy, sideways markets. Backtest before
trusting it.
"""

from __future__ import annotations

from typing import List

from ..indicators import sma
from ..model import Candle, Signal
from ..strategy import Strategy


class SmaCrossover(Strategy):
    name = "sma_crossover"

    def __init__(self, fast: int = 10, slow: int = 30):
        if fast >= slow:
            raise ValueError("fast period must be shorter than slow period")
        self.fast = fast
        self.slow = slow

    def warmup(self) -> int:
        return self.slow + 1

    def evaluate(self, history: List[Candle]) -> Signal:
        if len(history) < self.slow + 1:
            return Signal.HOLD

        closes = [c.close for c in history]
        fast = sma(closes, self.fast)
        slow = sma(closes, self.slow)

        if None in (fast[-1], fast[-2], slow[-1], slow[-2]):
            return Signal.HOLD

        crossed_up = fast[-2] <= slow[-2] and fast[-1] > slow[-1]
        crossed_down = fast[-2] >= slow[-2] and fast[-1] < slow[-1]

        if crossed_up:
            return Signal.BUY
        if crossed_down:
            return Signal.SELL
        return Signal.HOLD
