"""Strategy interface.

A strategy looks at the history of candles seen so far and emits a Signal.
The same object works in backtest, paper, and live modes — the engine only
ever calls :meth:`Strategy.evaluate`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from .model import Candle, Signal


class Strategy(ABC):
    """Base class for all strategies.

    Subclasses implement :meth:`evaluate`, which receives the full list of
    candles seen so far (the most recent is ``history[-1]``) and returns a
    :class:`Signal`. Returning ``HOLD`` means "do nothing"; the engine, not the
    strategy, decides position sizing.
    """

    name: str = "strategy"

    @abstractmethod
    def evaluate(self, history: List[Candle]) -> Signal:
        ...

    def warmup(self) -> int:
        """How many candles are needed before signals are meaningful."""
        return 0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} {self.name}>"
