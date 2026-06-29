"""Optimization and validation.

Two tools that, used together, are the difference between a strategy that makes
money and one that only *looks* like it does:

* ``grid_search`` brute-forces parameter combinations and ranks them by a metric.
  Powerful and dangerous: the top result is often **overfit** — tuned to noise
  in your sample that won't repeat.

* ``walk_forward`` is the antidote. It optimizes on one slice of history, then
  measures performance on the *next, unseen* slice, and repeats. The combined
  out-of-sample equity curve is a far more honest estimate of live performance.
  If a strategy is great in-sample but falls apart walk-forward, it was overfit —
  better to learn that here than with real money.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from .backtest import BacktestResult, run_backtest
from .metrics import compute
from .model import Candle
from .risk import RiskConfig
from .strategies import build


def _score(result: BacktestResult, metric: str) -> float:
    value = getattr(result.metrics, metric, None)
    if value is None:
        raise ValueError(f"unknown metric {metric!r}")
    return float(value)


@dataclass
class GridResult:
    params: Dict
    score: float
    result: BacktestResult


def grid_search(strategy_name: str, candles: List[Candle],
                param_grid: Dict[str, Sequence], metric: str = "sharpe",
                interval: str = "1h", risk: RiskConfig = None,
                **bt_kwargs) -> List[GridResult]:
    """Backtest every combination in ``param_grid``; return results best-first.

    ``param_grid`` maps a strategy kwarg to the values to try, e.g.
    ``{"fast": [5, 10], "slow": [20, 30]}``. Combinations that raise (invalid
    parameter pairs) are skipped.
    """
    keys = list(param_grid)
    combos = list(itertools.product(*(param_grid[k] for k in keys)))
    out: List[GridResult] = []
    for combo in combos:
        params = dict(zip(keys, combo))
        try:
            strat = build(strategy_name, **params)
            result = run_backtest(strat, candles, interval=interval, risk=risk,
                                  **bt_kwargs)
        except (ValueError, TypeError):
            continue
        out.append(GridResult(params, _score(result, metric), result))
    out.sort(key=lambda g: g.score, reverse=True)
    return out


@dataclass
class WalkForwardResult:
    folds: int
    in_sample_metric: float
    out_sample_metric: float
    out_sample_return_pct: float
    out_sample_max_drawdown_pct: float
    chosen_params: List[Dict]
    oos_equity_curve: List[float]

    def summary(self) -> str:
        decay = self.in_sample_metric - self.out_sample_metric
        verdict = ("looks robust" if self.out_sample_metric > 0 and decay < self.in_sample_metric * 0.5
                   else "likely OVERFIT — be skeptical")
        return "\n".join([
            f"  Folds                : {self.folds}",
            f"  In-sample  metric    : {self.in_sample_metric:+.2f}  (optimized)",
            f"  Out-of-sample metric : {self.out_sample_metric:+.2f}  (honest)",
            f"  OOS total return     : {self.out_sample_return_pct:+.2f}%",
            f"  OOS max drawdown     : {self.out_sample_max_drawdown_pct:.2f}%",
            f"  Verdict              : {verdict}",
        ])


def walk_forward(strategy_name: str, candles: List[Candle],
                 param_grid: Dict[str, Sequence], folds: int = 4,
                 train_ratio: float = 0.7, metric: str = "sharpe",
                 interval: str = "1h", risk: RiskConfig = None,
                 **bt_kwargs) -> WalkForwardResult:
    """Anchored walk-forward optimization.

    The data is split into ``folds`` windows. In each window the parameters are
    optimized on the first ``train_ratio`` of bars and then evaluated on the
    remaining out-of-sample bars. Out-of-sample results are stitched together
    into one honest equity curve.
    """
    if folds < 1:
        raise ValueError("folds must be >= 1")
    n = len(candles)
    window = n // folds
    if window < 20:
        raise ValueError("not enough candles for this many folds")

    in_scores: List[float] = []
    out_scores: List[float] = []
    chosen: List[Dict] = []
    oos_equity: List[float] = []
    oos_wins = oos_closed = 0
    oos_gp = oos_gl = 0.0

    for f in range(folds):
        start = f * window
        end = n if f == folds - 1 else (f + 1) * window
        chunk = candles[start:end]
        split = int(len(chunk) * train_ratio)
        train, test = chunk[:split], chunk[split:]
        if len(train) < 10 or len(test) < 5:
            continue

        ranked = grid_search(strategy_name, train, param_grid, metric=metric,
                             interval=interval, risk=risk, **bt_kwargs)
        if not ranked:
            continue
        best = ranked[0]
        chosen.append(best.params)
        in_scores.append(best.score)

        strat = build(strategy_name, **best.params)
        oos = run_backtest(strat, test, interval=interval, risk=risk, **bt_kwargs)
        out_scores.append(_score(oos, metric))

        # Stitch OOS curves, rebasing each onto the running equity.
        base = oos_equity[-1] if oos_equity else (oos.equity_curve[0] if oos.equity_curve else 0.0)
        if oos.equity_curve:
            scale = base / oos.equity_curve[0] if oos.equity_curve[0] else 1.0
            oos_equity.extend(v * scale for v in oos.equity_curve)
        oos_wins += int(round(oos.metrics.win_rate_pct / 100 * oos.metrics.num_trades))
        oos_closed += oos.metrics.num_trades

    if not out_scores:
        raise ValueError("walk-forward produced no usable folds")

    oos_metrics = compute(oos_equity, num_trades=oos_closed, wins=oos_wins,
                          interval=interval)
    return WalkForwardResult(
        folds=len(out_scores),
        in_sample_metric=round(sum(in_scores) / len(in_scores), 2),
        out_sample_metric=round(sum(out_scores) / len(out_scores), 2),
        out_sample_return_pct=oos_metrics.total_return_pct,
        out_sample_max_drawdown_pct=oos_metrics.max_drawdown_pct,
        chosen_params=chosen,
        oos_equity_curve=oos_equity,
    )


# Reasonable default search grids per strategy, used by the CLI optimizer.
DEFAULT_GRIDS: Dict[str, Dict[str, Sequence]] = {
    "sma": {"fast": [5, 10, 15, 20], "slow": [30, 50, 80, 120]},
    "rsi": {"period": [7, 14, 21], "oversold": [20, 30], "overbought": [70, 80]},
    "macd": {"fast": [8, 12], "slow": [21, 26], "signal": [9, 12]},
    "bollinger": {"period": [14, 20, 30], "num_std": [1.5, 2.0, 2.5],
                  "mode": ["reversion", "breakout"]},
    "breakout": {"entry": [10, 20, 40, 55], "exit_period": [5, 10, 20]},
    "supertrend": {"period": [7, 10, 14], "multiplier": [2.0, 3.0, 4.0]},
    "vwap": {"period": [10, 20, 30], "band": [0.01, 0.02, 0.03]},
    "stochastic": {"k_period": [9, 14, 21], "oversold": [15, 20],
                   "overbought": [80, 85]},
}
