"""Built-in example strategies and a registry for looking them up by name."""

from __future__ import annotations

from typing import Dict, Type

from ..strategy import Strategy
from .sma_crossover import SmaCrossover
from .rsi_reversion import RsiReversion
from .macd_cross import MacdCross
from .bollinger_breakout import BollingerBreakout
from .breakout import Breakout
from .ensemble import Ensemble

REGISTRY: Dict[str, Type[Strategy]] = {
    "sma": SmaCrossover,
    "rsi": RsiReversion,
    "macd": MacdCross,
    "bollinger": BollingerBreakout,
    "breakout": Breakout,
    "ensemble": Ensemble,
}


def build(name: str, **params) -> Strategy:
    """Instantiate a registered strategy by short name."""
    try:
        cls = REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown strategy {name!r}; available: {available}")
    return cls(**params)


__all__ = [
    "SmaCrossover", "RsiReversion", "MacdCross", "BollingerBreakout",
    "Breakout", "Ensemble", "REGISTRY", "build",
]
