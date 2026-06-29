"""Multi-timeframe trend filter.

"Trade in the direction of the higher timeframe" is one of the oldest pieces of
trading wisdom, and it's genuinely effective: it stops a fast strategy from
fighting the dominant trend. This wraps any base strategy and only lets its
signals through when they agree with a slower, higher-timeframe trend:

* higher-timeframe trend up   -> allow BUY,  block SELL
* higher-timeframe trend down -> allow SELL, block BUY

The higher timeframe is built by resampling the candles by ``htf_factor`` (e.g.
4 turns 1h bars into 4h bars), then checking whether a moving average there is
rising or falling.
"""

from __future__ import annotations

from typing import List, Optional

from ..data import resample
from ..indicators import sma
from ..model import Candle, Signal
from ..strategy import Strategy


class MtfTrend(Strategy):
    name = "mtf_trend"

    def __init__(self, htf_factor: int = 4, htf_period: int = 20,
                 base: Optional[Strategy] = None):
        if htf_factor < 2:
            raise ValueError("htf_factor must be >= 2 to be a *higher* timeframe")
        if base is None:
            from .sma_crossover import SmaCrossover
            base = SmaCrossover()
        self.htf_factor = htf_factor
        self.htf_period = htf_period
        self.base = base

    def warmup(self) -> int:
        # Need enough base-timeframe bars to form htf_period higher-TF bars.
        return max(self.base.warmup(),
                   (self.htf_period + 2) * self.htf_factor)

    def lookback(self) -> int:
        # Opt out of windowing: resample() groups from the start of whatever
        # list it gets, so a moving window would shift the higher-timeframe bar
        # boundaries every bar. Use the full history to keep that anchoring
        # stable and the signals consistent.
        return 0

    def _htf_direction(self, history: List[Candle]) -> int:
        """+1 if the higher-timeframe MA is rising, -1 if falling, 0 if unknown."""
        htf = resample(history, self.htf_factor)
        closes = [c.close for c in htf]
        ma = sma(closes, self.htf_period)
        if len(ma) < 2 or ma[-1] is None or ma[-2] is None:
            return 0
        if ma[-1] > ma[-2]:
            return 1
        if ma[-1] < ma[-2]:
            return -1
        return 0

    def evaluate(self, history: List[Candle]) -> Signal:
        signal = self.base.evaluate(history)
        if signal is Signal.HOLD:
            return Signal.HOLD
        direction = self._htf_direction(history)
        if direction == 0:
            return Signal.HOLD          # no clear higher-TF trend -> stand aside
        if signal is Signal.BUY and direction > 0:
            return Signal.BUY
        if signal is Signal.SELL and direction < 0:
            return Signal.SELL
        return Signal.HOLD              # signal fights the higher-TF trend -> skip
