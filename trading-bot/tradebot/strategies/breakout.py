"""Donchian channel breakout — the classic trend-following 'turtle' entry.

Buy when price breaks above the highest high of the last ``entry`` bars; exit
when it breaks below the lowest low of the last ``exit_period`` bars. Trend
followers make their money from rare, huge moves and accept many small losses
in between — so expect a low win rate but large average winners.
"""

from __future__ import annotations

from typing import List

from ..indicators import donchian
from ..model import Candle, Signal
from ..strategy import Strategy


class Breakout(Strategy):
    name = "breakout"

    def __init__(self, entry: int = 20, exit_period: int = 10):
        self.entry = entry
        self.exit_period = exit_period

    def warmup(self) -> int:
        return max(self.entry, self.exit_period) + 1

    def evaluate(self, history: List[Candle]) -> Signal:
        if len(history) < self.warmup():
            return Signal.HOLD
        highs = [c.high for c in history]
        lows = [c.low for c in history]
        price = history[-1].close

        # Use channels computed up to the *previous* bar to avoid comparing the
        # current bar's high/low against a window that already includes it.
        _, entry_hi = donchian(highs[:-1], lows[:-1], self.entry)
        entry_lo, _ = donchian(highs[:-1], lows[:-1], self.exit_period)
        if entry_hi[-1] is None or entry_lo[-1] is None:
            return Signal.HOLD

        if price > entry_hi[-1]:
            return Signal.BUY
        if price < entry_lo[-1]:
            return Signal.SELL
        return Signal.HOLD
