"""Backtesting engine.

Replays historical candles one at a time, feeding each strategy only the data
it would have had at that moment (no look-ahead), and simulates fills through a
:class:`PaperBroker`. The result is an equity curve plus the metrics that tell
you whether the strategy is worth real money.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .broker import PaperBroker
from .metrics import Metrics, compute
from .model import Candle, Signal, Trade
from .strategy import Strategy


@dataclass
class BacktestResult:
    metrics: Metrics
    equity_curve: List[float]
    trades: List[Trade]
    buy_and_hold_return_pct: float

    def summary(self) -> str:
        m = self.metrics
        edge = m.total_return_pct - self.buy_and_hold_return_pct
        lines = [
            f"  Start equity     : {m.start_equity:,.2f}",
            f"  End equity       : {m.end_equity:,.2f}",
            f"  Total return     : {m.total_return_pct:+.2f}%",
            f"  Buy & hold return: {self.buy_and_hold_return_pct:+.2f}%",
            f"  Edge vs hold     : {edge:+.2f}%",
            f"  Max drawdown     : {m.max_drawdown_pct:.2f}%",
            f"  Sharpe (annual)  : {m.sharpe:.2f}",
            f"  Trades (closed)  : {m.num_trades}",
            f"  Win rate         : {m.win_rate_pct:.2f}%",
        ]
        return "\n".join(lines)


def run_backtest(strategy: Strategy, candles: List[Candle],
                 cash: float = 10_000.0, fee_rate: float = 0.001,
                 slippage: float = 0.0005, position_fraction: float = 1.0,
                 interval: str = "1d") -> BacktestResult:
    """Run ``strategy`` over ``candles`` and return performance results.

    The strategy is long/flat: a BUY deploys ``position_fraction`` of cash, a
    SELL closes the whole position. Win rate is measured on closed round-trips
    by comparing each SELL's proceeds against the average cost of the position.
    """
    broker = PaperBroker(cash=cash, fee_rate=fee_rate, slippage=slippage)
    equity_curve: List[float] = []

    # Track average entry cost so we can label each closed trade win/loss.
    cost_basis = 0.0      # total quote spent on the open position (incl. fees)
    wins = 0
    closed = 0

    for i in range(len(candles)):
        history = candles[: i + 1]
        price = history[-1].close
        signal = strategy.evaluate(history)

        if signal is Signal.BUY and broker.position == 0:
            trade = broker.buy(history[-1].timestamp, price, position_fraction)
            if trade:
                cost_basis = trade.price * trade.quantity + trade.fee
        elif signal is Signal.SELL and broker.position > 0:
            qty_before = broker.position
            trade = broker.sell(history[-1].timestamp, price, 1.0)
            if trade:
                proceeds = trade.price * trade.quantity - trade.fee
                if proceeds > cost_basis:
                    wins += 1
                closed += 1
                cost_basis = 0.0

        equity_curve.append(broker.equity(price))

    metrics = compute(equity_curve, num_trades=closed, wins=wins, interval=interval)

    first = candles[0].close
    last = candles[-1].close
    bnh = (last / first - 1) * 100 if first else 0.0

    return BacktestResult(
        metrics=metrics,
        equity_curve=equity_curve,
        trades=broker.trades,
        buy_and_hold_return_pct=round(bnh, 2),
    )
