"""Backtesting engine.

Replays historical candles one at a time, feeding each strategy only the data
it would have had at that moment (no look-ahead), and simulates fills through a
:class:`PaperBroker`. An optional :class:`RiskManager` enforces stop-loss,
take-profit and a max-drawdown circuit breaker, and sizes positions. The result
is an equity curve plus the metrics that tell you whether the strategy is worth
real money.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .broker import PaperBroker
from .metrics import Metrics, compute
from .model import Candle, Signal, Trade
from .risk import RiskConfig, RiskManager
from .strategy import Strategy


@dataclass
class BacktestResult:
    metrics: Metrics
    equity_curve: List[float]
    trades: List[Trade]
    buy_and_hold_return_pct: float
    halted: bool = False

    def summary(self) -> str:
        m = self.metrics
        edge = m.total_return_pct - self.buy_and_hold_return_pct
        lines = [
            f"  Start equity     : {m.start_equity:,.2f}",
            f"  End equity       : {m.end_equity:,.2f}",
            f"  Total return     : {m.total_return_pct:+.2f}%",
            f"  Buy & hold return: {self.buy_and_hold_return_pct:+.2f}%",
            f"  Edge vs hold     : {edge:+.2f}%",
            f"  CAGR             : {m.cagr_pct:+.2f}%",
            f"  Max drawdown     : {m.max_drawdown_pct:.2f}%",
            f"  Sharpe (annual)  : {m.sharpe:.2f}",
            f"  Sortino (annual) : {m.sortino:.2f}",
            f"  Calmar           : {m.calmar:.2f}",
            f"  Profit factor    : {m.profit_factor:.2f}",
            f"  Exposure         : {m.exposure_pct:.1f}%",
            f"  Trades (closed)  : {m.num_trades}",
            f"  Win rate         : {m.win_rate_pct:.2f}%",
        ]
        if self.halted:
            lines.append("  ** Max-drawdown circuit breaker tripped: trading halted. **")
        return "\n".join(lines)


def run_backtest(strategy: Strategy, candles: List[Candle],
                 cash: float = 10_000.0, fee_rate: float = 0.001,
                 slippage: float = 0.0005, position_fraction: float = 1.0,
                 interval: str = "1d", allow_short: bool = False,
                 risk: Optional[RiskConfig] = None) -> BacktestResult:
    """Run ``strategy`` over ``candles`` and return performance results.

    By default the strategy is long/flat. With ``allow_short=True`` it becomes
    a stop-and-reverse system: a SELL with no long open opens a short, and a BUY
    covers a short before (optionally) going long.

    Each bar, in order: protective stop/target exits are checked against the
    bar's high/low first, then the strategy's signal is applied, then the
    drawdown circuit breaker. Round-trip PnL is measured as the change in
    account equity between entry and exit, which works for longs and shorts
    alike. Win rate and profit factor are measured on those closed round-trips.
    """
    if risk is None:
        risk = RiskConfig(position_fraction=position_fraction)
    manager = RiskManager(risk)
    broker = PaperBroker(cash=cash, fee_rate=fee_rate, slippage=slippage)

    equity_curve: List[float] = []
    entry_equity = 0.0     # account equity right after the position was opened
    wins = 0
    closed = 0
    gross_profit = 0.0
    gross_loss = 0.0
    bars_in_market = 0
    halted = False

    def close_position(ts: int, price: float) -> None:
        nonlocal wins, closed, gross_profit, gross_loss
        if broker.position > 0:
            trade = broker.sell(ts, price, 1.0)
        elif broker.position < 0:
            trade = broker.cover(ts, price)
        else:
            return
        if not trade:
            return
        pnl = broker.equity(price) - entry_equity   # position is now flat
        if pnl > 0:
            wins += 1
            gross_profit += pnl
        else:
            gross_loss += -pnl
        closed += 1
        manager.on_exit()

    def open_position(ts: int, price: float, side: int) -> None:
        nonlocal entry_equity
        if side > 0:
            trade = broker.buy(ts, price, risk.entry_fraction())
        else:
            trade = broker.sell_short(ts, price, risk.entry_fraction())
        if trade:
            entry_equity = broker.equity(price)
            manager.on_entry(trade.price, side=side)

    for i in range(len(candles)):
        bar = candles[i]
        history = candles[: i + 1]
        price = bar.close

        # 1) Protective exits (intrabar, before acting on new signals).
        if broker.position != 0:
            exit_price = manager.protective_exit(bar.high, bar.low)
            if exit_price is not None:
                close_position(bar.timestamp, exit_price)

        # 2) Strategy signal (skipped once the circuit breaker has tripped).
        if not halted:
            signal = strategy.evaluate(history)
            if signal is Signal.BUY:
                if broker.position < 0:
                    close_position(bar.timestamp, price)
                if broker.position == 0:
                    open_position(bar.timestamp, price, side=1)
            elif signal is Signal.SELL:
                if broker.position > 0:
                    close_position(bar.timestamp, price)
                if allow_short and broker.position == 0:
                    open_position(bar.timestamp, price, side=-1)

        if broker.position != 0:
            bars_in_market += 1

        equity = broker.equity(price)
        equity_curve.append(equity)

        # 3) Circuit breaker: if drawdown limit breached, liquidate and halt.
        if manager.update_equity(equity) and not halted:
            halted = True
            if broker.position != 0:
                close_position(bar.timestamp, price)
                equity_curve[-1] = broker.equity(price)

    exposure = (bars_in_market / len(candles) * 100) if candles else 0.0
    metrics = compute(equity_curve, num_trades=closed, wins=wins,
                      interval=interval, gross_profit=gross_profit,
                      gross_loss=gross_loss, exposure_pct=exposure)

    first = candles[0].close
    last = candles[-1].close
    bnh = (last / first - 1) * 100 if first else 0.0

    return BacktestResult(
        metrics=metrics,
        equity_curve=equity_curve,
        trades=broker.trades,
        buy_and_hold_return_pct=round(bnh, 2),
        halted=halted,
    )
