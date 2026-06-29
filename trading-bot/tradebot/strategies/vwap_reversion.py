"""VWAP mean-reversion.

Volume-Weighted Average Price is the average price the market actually paid over
the window. When price stretches a chosen fraction *below* VWAP it's "cheap
relative to where volume traded" — buy, expecting a snap back. When it stretches
above, sell. A reversion strategy: it earns in choppy markets and bleeds in
strong trends, so pair it with the regime filter or backtest before trusting it.
"""

from __future__ import annotations

from typing import List

from ..indicators import vwap
from ..model import Candle, Signal
from ..strategy import Strategy


class VwapReversion(Strategy):
    name = "vwap_reversion"

    def __init__(self, period: int = 20, band: float = 0.02):
        if band <= 0:
            raise ValueError("band must be positive")
        self.period = period
        self.band = band

    def warmup(self) -> int:
        return self.period + 1

    def evaluate(self, history: List[Candle]) -> Signal:
        highs = [c.high for c in history]
        lows = [c.low for c in history]
        closes = [c.close for c in history]
        vols = [c.volume for c in history]
        vw = vwap(highs, lows, closes, vols, self.period)
        if vw[-1] is None:
            return Signal.HOLD
        price = closes[-1]
        if price < vw[-1] * (1 - self.band):
            return Signal.BUY
        if price > vw[-1] * (1 + self.band):
            return Signal.SELL
        return Signal.HOLD
