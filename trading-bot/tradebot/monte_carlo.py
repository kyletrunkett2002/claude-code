"""Monte Carlo robustness testing.

A single backtest gives you *one* history — but the future will deal the same
trades in a different order, with different luck. A strategy can look great
purely because its winners happened to land in a lucky sequence.

Monte Carlo stress-tests that. It takes your strategy's realized per-trade
returns and resamples them thousands of times (bootstrap, with replacement) to
build a *distribution* of outcomes. Instead of one return and one drawdown you
get ranges: "5% of the time this strategy loses money" or "1-in-20 chance of a
40% drawdown." That is far more honest than a single backtest number, and it is
how risk desks actually think about a strategy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List


@dataclass
class MonteCarloResult:
    simulations: int
    trades_per_sim: int
    return_p5: float       # 5th percentile final return %
    return_p50: float      # median final return %
    return_p95: float      # 95th percentile final return %
    drawdown_p50: float    # median max drawdown %
    drawdown_p95: float    # 95th percentile (bad) max drawdown %
    prob_loss_pct: float   # % of simulations that ended below the starting stake
    prob_big_dd_pct: float # % of simulations breaching `big_drawdown`
    big_drawdown_pct: float

    def summary(self) -> str:
        return "\n".join([
            f"  Simulations          : {self.simulations:,} × {self.trades_per_sim} trades",
            f"  Final return  p5/p50/p95 : "
            f"{self.return_p5:+.1f}% / {self.return_p50:+.1f}% / {self.return_p95:+.1f}%",
            f"  Max drawdown  p50/p95    : "
            f"{self.drawdown_p50:.1f}% / {self.drawdown_p95:.1f}%",
            f"  Probability of net loss  : {self.prob_loss_pct:.1f}%",
            f"  Probability of >{self.big_drawdown_pct:.0f}% drawdown : "
            f"{self.prob_big_dd_pct:.1f}%",
        ])


def _max_drawdown_pct(curve: List[float]) -> float:
    peak = curve[0]
    worst = 0.0
    for v in curve:
        peak = max(peak, v)
        if peak > 0:
            worst = min(worst, (v - peak) / peak)
    return abs(worst) * 100


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def bootstrap(trade_returns: List[float], simulations: int = 2000,
              big_drawdown_pct: float = 30.0, seed: int = 7) -> MonteCarloResult:
    """Bootstrap per-trade returns into a distribution of outcomes.

    Args:
        trade_returns: realized round-trip return fractions from a backtest
            (e.g. 0.05 for a +5% trade), as found on ``BacktestResult.trade_returns``.
        simulations: number of resampled equity paths to generate.
        big_drawdown_pct: the drawdown threshold to report a probability for.
        seed: RNG seed for reproducibility.

    Each simulation draws ``len(trade_returns)`` trades with replacement and
    compounds them into an equity path starting at 1.0.
    """
    n = len(trade_returns)
    if n == 0:
        raise ValueError("no trades to bootstrap; the strategy never traded")

    rng = random.Random(seed)
    finals: List[float] = []
    drawdowns: List[float] = []
    losses = 0
    big_dd = 0

    for _ in range(simulations):
        equity = 1.0
        curve = [equity]
        for _ in range(n):
            r = trade_returns[rng.randrange(n)]
            equity *= (1 + r)
            if equity <= 0:        # wiped out
                equity = 0.0
                curve.append(equity)
                break
            curve.append(equity)
        final_return = (curve[-1] - 1) * 100
        dd = _max_drawdown_pct(curve)
        finals.append(final_return)
        drawdowns.append(dd)
        if curve[-1] < 1.0:
            losses += 1
        if dd >= big_drawdown_pct:
            big_dd += 1

    finals.sort()
    drawdowns.sort()
    return MonteCarloResult(
        simulations=simulations,
        trades_per_sim=n,
        return_p5=round(_percentile(finals, 5), 2),
        return_p50=round(_percentile(finals, 50), 2),
        return_p95=round(_percentile(finals, 95), 2),
        drawdown_p50=round(_percentile(drawdowns, 50), 2),
        drawdown_p95=round(_percentile(drawdowns, 95), 2),
        prob_loss_pct=round(losses / simulations * 100, 2),
        prob_big_dd_pct=round(big_dd / simulations * 100, 2),
        big_drawdown_pct=big_drawdown_pct,
    )
