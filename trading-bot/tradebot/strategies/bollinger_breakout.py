"""Bollinger Band strategy with selectable mode.

- mode="reversion" (default): buy when price closes below the lower band
  (stretched cheap), sell when it closes above the upper band. Wins in ranges.
- mode="breakout": buy when price closes above the upper band (volatility
  expansion to the upside), sell when it closes below the lower band. Wins in
  trends.

Same indicator, opposite trades — which one works is an empirical question you
answer with the backtester, not a matter of opinion.
"""

from __future__ import annotations

from typing import List

from ..indicators import bollinger
from ..model import Candle, Signal
from ..strategy import Strategy


class BollingerBreakout(Strategy):
    name = "bollinger"

    def __init__(self, period: int = 20, num_std: float = 2.0, mode: str = "reversion"):
        if mode not in ("reversion", "breakout"):
            raise ValueError("mode must be 'reversion' or 'breakout'")
        self.period = period
        self.num_std = num_std
        self.mode = mode

    def warmup(self) -> int:
        return self.period + 1

    def evaluate(self, history: List[Candle]) -> Signal:
        closes = [c.close for c in history]
        lower, _, upper = bollinger(closes, self.period, self.num_std)
        if lower[-1] is None or upper[-1] is None:
            return Signal.HOLD
        price = closes[-1]
        below = price < lower[-1]
        above = price > upper[-1]
        if self.mode == "reversion":
            if below:
                return Signal.BUY
            if above:
                return Signal.SELL
        else:  # breakout
            if above:
                return Signal.BUY
            if below:
                return Signal.SELL
        return Signal.HOLD
