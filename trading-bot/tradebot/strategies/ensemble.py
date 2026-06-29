"""Ensemble — combine several strategies by majority vote.

Each member votes BUY (+1), SELL (-1), or HOLD (0). If the net vote meets a
threshold the ensemble acts, otherwise it holds. Diversifying across uncorrelated
signals tends to smooth the equity curve: when one strategy is wrong, others may
be right. It is not magic — a basket of bad strategies is still bad.
"""

from __future__ import annotations

from typing import List, Optional

from ..model import Candle, Signal
from ..strategy import Strategy


class Ensemble(Strategy):
    name = "ensemble"

    def __init__(self, members: Optional[List[Strategy]] = None, threshold: int = 1):
        if members is None:
            # Sensible default basket spanning trend + reversion.
            from .sma_crossover import SmaCrossover
            from .macd_cross import MacdCross
            from .rsi_reversion import RsiReversion
            members = [SmaCrossover(), MacdCross(), RsiReversion()]
        if not members:
            raise ValueError("ensemble needs at least one member")
        self.members = members
        self.threshold = max(1, threshold)

    def warmup(self) -> int:
        return max(m.warmup() for m in self.members)

    def evaluate(self, history: List[Candle]) -> Signal:
        score = 0
        for m in self.members:
            sig = m.evaluate(history)
            if sig is Signal.BUY:
                score += 1
            elif sig is Signal.SELL:
                score -= 1
        if score >= self.threshold:
            return Signal.BUY
        if score <= -self.threshold:
            return Signal.SELL
        return Signal.HOLD
