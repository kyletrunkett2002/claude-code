"""Performance metrics for an equity curve.

These are the numbers that tell you whether a strategy is actually worth real
money. Return alone is a trap — a strategy that doubles your account and then
loses 70% along the way is usually untradeable. Always read return *next to*
max drawdown and Sharpe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List

# Approximate number of each bar interval in a 365-day year, for annualizing.
BARS_PER_YEAR = {
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "1h": 8_760,
    "4h": 2_190,
    "1d": 365,
}


@dataclass
class Metrics:
    start_equity: float
    end_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe: float
    num_trades: int
    win_rate_pct: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute(equity_curve: List[float], num_trades: int, wins: int,
            interval: str = "1d") -> Metrics:
    """Compute summary metrics from an equity curve.

    Args:
        equity_curve: account equity sampled once per bar.
        num_trades: number of *round-trip* sells (closed positions).
        wins: how many of those closed trades were profitable.
        interval: bar size, used to annualize the Sharpe ratio.
    """
    if len(equity_curve) < 2:
        start = equity_curve[0] if equity_curve else 0.0
        return Metrics(start, start, 0.0, 0.0, 0.0, num_trades, 0.0)

    start = equity_curve[0]
    end = equity_curve[-1]
    total_return = (end / start - 1) * 100 if start else 0.0

    max_dd = _max_drawdown(equity_curve)
    sharpe = _sharpe(equity_curve, interval)
    win_rate = (wins / num_trades * 100) if num_trades else 0.0

    return Metrics(
        start_equity=round(start, 2),
        end_equity=round(end, 2),
        total_return_pct=round(total_return, 2),
        max_drawdown_pct=round(max_dd, 2),
        sharpe=round(sharpe, 2),
        num_trades=num_trades,
        win_rate_pct=round(win_rate, 2),
    )


def _max_drawdown(curve: List[float]) -> float:
    """Largest peak-to-trough decline, as a positive percentage."""
    peak = curve[0]
    worst = 0.0
    for v in curve:
        peak = max(peak, v)
        if peak > 0:
            dd = (v - peak) / peak
            worst = min(worst, dd)
    return abs(worst) * 100


def _sharpe(curve: List[float], interval: str) -> float:
    """Annualized Sharpe ratio (risk-free rate assumed 0)."""
    returns = []
    for prev, cur in zip(curve, curve[1:]):
        if prev > 0:
            returns.append(cur / prev - 1)
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    periods = BARS_PER_YEAR.get(interval, 365)
    return (mean / std) * math.sqrt(periods)
