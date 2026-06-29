"""Regime-adaptive meta-strategy — use the right tool for the market.

No single strategy works in all conditions: trend-followers win in trends and
bleed in chop, mean-reverters do the opposite. This strategy reads the market's
*regime* with ADX (trend strength) and routes the decision accordingly:

* ADX high (strong trend)  -> defer to a trend strategy (default: SuperTrend)
* ADX low  (choppy range)  -> defer to a reversion strategy (default: RSI)

Adapting to the regime is one of the most robust ideas in systematic trading —
but it's only as good as its two members, so validate the whole thing with
walk-forward and Monte Carlo like anything else.
"""

from __future__ import annotations

from typing import List, Optional

from ..indicators import adx
from ..model import Candle, Signal
from ..strategy import Strategy


class RegimeAdaptive(Strategy):
    name = "regime_adaptive"

    def __init__(self, adx_period: int = 14, adx_threshold: float = 25.0,
                 trend: Optional[Strategy] = None,
                 reversion: Optional[Strategy] = None):
        if trend is None:
            from .supertrend import SuperTrend
            trend = SuperTrend()
        if reversion is None:
            from .rsi_reversion import RsiReversion
            reversion = RsiReversion()
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.trend = trend
        self.reversion = reversion

    def warmup(self) -> int:
        return max(self.adx_period * 2 + 2, self.trend.warmup(),
                   self.reversion.warmup())

    def evaluate(self, history: List[Candle]) -> Signal:
        highs = [c.high for c in history]
        lows = [c.low for c in history]
        closes = [c.close for c in history]
        strength = adx(highs, lows, closes, self.adx_period)
        current = strength[-1]
        if current is None:
            return Signal.HOLD
        if current >= self.adx_threshold:
            return self.trend.evaluate(history)
        return self.reversion.evaluate(history)
