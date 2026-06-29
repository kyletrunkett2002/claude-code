"""Performance metrics for an equity curve.

These are the numbers that tell you whether a strategy is actually worth real
money. Return alone is a trap — a strategy that doubles your account and then
loses 70% along the way is usually untradeable. Always read return *next to*
max drawdown, Sortino, and Calmar.
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
    cagr_pct: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    calmar: float
    profit_factor: float
    exposure_pct: float
    num_trades: int
    win_rate_pct: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute(equity_curve: List[float], num_trades: int, wins: int,
            interval: str = "1d", gross_profit: float = 0.0,
            gross_loss: float = 0.0, exposure_pct: float = 0.0) -> Metrics:
    """Compute summary metrics from an equity curve.

    Args:
        equity_curve: account equity sampled once per bar.
        num_trades: number of closed round-trips.
        wins: how many closed trades were profitable.
        interval: bar size, used to annualize ratios and the CAGR.
        gross_profit / gross_loss: summed PnL of winning / losing trades
            (gross_loss as a positive number), used for the profit factor.
        exposure_pct: share of bars spent holding a position.
    """
    if len(equity_curve) < 2:
        start = equity_curve[0] if equity_curve else 0.0
        return Metrics(start, start, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       exposure_pct, num_trades, 0.0)

    start = equity_curve[0]
    end = equity_curve[-1]
    total_return = (end / start - 1) * 100 if start else 0.0

    periods = BARS_PER_YEAR.get(interval, 365)
    years = (len(equity_curve) - 1) / periods
    if start > 0 and end > 0 and years > 0:
        cagr = ((end / start) ** (1 / years) - 1) * 100
    else:
        cagr = 0.0

    max_dd = _max_drawdown(equity_curve)
    returns = _bar_returns(equity_curve)
    sharpe = _sharpe(returns, periods)
    sortino = _sortino(returns, periods)
    calmar = (cagr / max_dd) if max_dd > 0 else 0.0
    profit_factor = _profit_factor(gross_profit, gross_loss)
    win_rate = (wins / num_trades * 100) if num_trades else 0.0

    return Metrics(
        start_equity=round(start, 2),
        end_equity=round(end, 2),
        total_return_pct=round(total_return, 2),
        cagr_pct=round(cagr, 2),
        max_drawdown_pct=round(max_dd, 2),
        sharpe=round(sharpe, 2),
        sortino=round(sortino, 2),
        calmar=round(calmar, 2),
        profit_factor=round(profit_factor, 2),
        exposure_pct=round(exposure_pct, 1),
        num_trades=num_trades,
        win_rate_pct=round(win_rate, 2),
    )


def _bar_returns(curve: List[float]) -> List[float]:
    out = []
    for prev, cur in zip(curve, curve[1:]):
        if prev > 0:
            out.append(cur / prev - 1)
    return out


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


def _sharpe(returns: List[float], periods: int) -> float:
    """Annualized Sharpe ratio (risk-free rate assumed 0)."""
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(periods)


def _sortino(returns: List[float], periods: int) -> float:
    """Annualized Sortino ratio — like Sharpe but penalizes only downside."""
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    downside = [r for r in returns if r < 0]
    if not downside:
        return 0.0
    dd = math.sqrt(sum(r * r for r in downside) / len(returns))
    if dd == 0:
        return 0.0
    return (mean / dd) * math.sqrt(periods)


def _profit_factor(gross_profit: float, gross_loss: float) -> float:
    """Gross profit / gross loss. >1 is profitable; <1 loses money."""
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss
