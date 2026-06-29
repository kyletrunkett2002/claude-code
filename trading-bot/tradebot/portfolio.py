"""Multi-asset portfolio backtesting.

Running one strategy across several uncorrelated coins is the closest thing to a
free lunch in trading: the same edge, spread over more bets, usually means a
smoother equity curve and shallower drawdowns than betting everything on one
coin. This module allocates capital across symbols, backtests each
independently, and stitches the results into one portfolio equity curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .backtest import BacktestResult, run_backtest
from .metrics import Metrics, compute
from .model import Candle
from .risk import RiskConfig
from .strategy import Strategy


@dataclass
class PortfolioResult:
    metrics: Metrics
    equity_curve: List[float]
    per_symbol: Dict[str, BacktestResult]
    buy_and_hold_return_pct: float

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"  Symbols          : {len(self.per_symbol)}  "
            f"({', '.join(self.per_symbol)})",
            f"  Start equity     : {m.start_equity:,.2f}",
            f"  End equity       : {m.end_equity:,.2f}",
            f"  Total return     : {m.total_return_pct:+.2f}%",
            f"  Buy & hold return: {self.buy_and_hold_return_pct:+.2f}%  (equal-weight basket)",
            f"  CAGR             : {m.cagr_pct:+.2f}%",
            f"  Max drawdown     : {m.max_drawdown_pct:.2f}%",
            f"  Sharpe (annual)  : {m.sharpe:.2f}",
            f"  Calmar           : {m.calmar:.2f}",
            f"  Trades (closed)  : {m.num_trades}",
        ]
        lines.append("  Per-symbol return:")
        for sym, res in self.per_symbol.items():
            lines.append(f"    {sym:<12} {res.metrics.total_return_pct:+8.2f}%  "
                         f"(maxDD {res.metrics.max_drawdown_pct:.1f}%)")
        return "\n".join(lines)


def run_portfolio(make_strategy: Callable[[], Strategy],
                  symbol_candles: Dict[str, List[Candle]],
                  cash: float = 10_000.0, fee_rate: float = 0.001,
                  slippage: float = 0.0005, interval: str = "1d",
                  allow_short: bool = False,
                  risk: Optional[RiskConfig] = None) -> PortfolioResult:
    """Backtest one strategy across many symbols with equal capital allocation.

    Args:
        make_strategy: a zero-arg factory returning a *fresh* strategy instance
            per symbol (so stateful strategies don't bleed across markets).
        symbol_candles: maps symbol -> its candle history.

    Series are aligned by index after trimming to the shortest one, so pass
    candles of the same interval. Each symbol receives ``cash / N`` and the
    per-symbol equity curves are summed into the portfolio curve.
    """
    if not symbol_candles:
        raise ValueError("need at least one symbol")

    n = min(len(c) for c in symbol_candles.values())
    if n < 2:
        raise ValueError("not enough overlapping candles across symbols")

    per_symbol_alloc = cash / len(symbol_candles)
    per_symbol: Dict[str, BacktestResult] = {}
    combined = [0.0] * n
    bnh_total = 0.0

    for sym, candles in symbol_candles.items():
        trimmed = candles[-n:]                      # align lengths from the right
        result = run_backtest(make_strategy(), trimmed, cash=per_symbol_alloc,
                              fee_rate=fee_rate, slippage=slippage,
                              interval=interval, allow_short=allow_short, risk=risk)
        per_symbol[sym] = result
        for i, v in enumerate(result.equity_curve):
            combined[i] += v
        bnh_total += result.buy_and_hold_return_pct

    total_closed = sum(r.metrics.num_trades for r in per_symbol.values())
    total_wins = sum(int(round(r.metrics.win_rate_pct / 100 * r.metrics.num_trades))
                     for r in per_symbol.values())
    metrics = compute(combined, num_trades=total_closed, wins=total_wins,
                      interval=interval)

    return PortfolioResult(
        metrics=metrics,
        equity_curve=combined,
        per_symbol=per_symbol,
        buy_and_hold_return_pct=round(bnh_total / len(symbol_candles), 2),
    )
