# tradebot

A small but serious **crypto trading bot framework** in pure Python — zero
third-party dependencies, fully offline-capable, 32 tests.

The honest path to making money with an automated strategy isn't a magic
algorithm — it's a disciplined workflow:

1. **Backtest** a strategy on history. Does it beat just holding, after fees and slippage?
2. **Optimize** its parameters — then **walk-forward validate** to check the result
   isn't overfit garbage. *This step is where most retail traders fool themselves.*
3. **Paper-trade** it live (real prices, fake money) on data it has never seen.
4. **Only then** risk real money — small, with stop-losses and a drawdown circuit breaker.

`tradebot` is built around that workflow, with real risk management baked in.

> ⚠️ **Reality check.** Most trading strategies lose money after costs, and no
> tool can guarantee profit — anyone who tells you otherwise is selling something.
> This is a tool for *finding out whether your idea actually has an edge*, and for
> not blowing up your account while you find out. Never trade money you can't
> afford to lose.

## Requirements

Python 3.8+. **No third-party packages** — standard library only.

## Quickstart

```bash
cd trading-bot

# 1. Compare every built-in strategy on offline synthetic data (no network):
python -m tradebot compare --synthetic 800 --interval 1h

# 2. Backtest one, with risk management and an ASCII equity chart:
python -m tradebot backtest -s sma --synthetic 800 \
    --stop-loss 0.05 --take-profit 0.15 --risk-per-trade 0.02 \
    --max-drawdown 0.25 --plot

# 3. Search for the best parameters:
python -m tradebot optimize -s sma --synthetic 800 --metric sharpe --top 10

# 4. Validate that result out-of-sample (the anti-overfitting check):
python -m tradebot walkforward -s sma --synthetic 800 --folds 4 --plot

# On a real machine (no proxy), swap --synthetic for live Binance data, no key:
python -m tradebot backtest -s breakout --symbol BTCUSDT --interval 4h --limit 1000

# 5. Paper-trade live prices with fake money (Ctrl-C to stop):
python -m tradebot paper -s sma --symbol BTCUSDT --interval 1h --poll 60
```

## Strategies

| Name | Style | Idea |
|---|---|---|
| `sma` | trend | Fast/slow moving-average crossover |
| `macd` | trend | MACD line crossing its signal line |
| `breakout` | trend | Donchian channel breakout (the "turtle" entry) |
| `rsi` | mean-reversion | Buy oversold, sell overbought |
| `bollinger` | both | Band reversion *or* breakout (`-p mode=breakout`) |
| `ensemble` | meta | Majority vote across several strategies |

Pass parameters with `-p key=value`, e.g. `-s sma -p fast=5 -p slow=30`.

## Risk management (the part that keeps you solvent)

Strategies decide *when* to trade; risk rules decide *how much* and *when to bail*.
All of these compose, on any command:

| Flag | What it does |
|---|---|
| `--stop-loss 0.05` | Exit if price falls 5% below entry (checked intrabar against the low) |
| `--take-profit 0.15` | Exit if price rises 15% above entry |
| `--risk-per-trade 0.02` | Size each position so a stop-out loses ~2% of equity (needs `--stop-loss`). Tighter stop → bigger size, constant dollar risk |
| `--max-drawdown 0.25` | Circuit breaker: liquidate and halt trading if equity falls 25% from its peak |
| `--fraction 0.5` | Fixed: deploy 50% of cash per entry |
| `--max-position 0.3` | Hard cap on cash deployed per trade |

`--risk-per-trade` is volatility-aware position sizing — the single most
important habit separating traders who survive from those who don't.

## Finding a *real* edge (not a mirage)

A great backtest is easy to fake by accident: try enough parameters and one will
look brilliant purely by luck. Two commands guard against this.

```bash
# Grid-search parameters, ranked by a metric:
python -m tradebot optimize -s breakout --synthetic 1000 --metric calmar

# Walk-forward: optimize on one slice, test on the NEXT, unseen slice, repeat.
python -m tradebot walkforward -s breakout --synthetic 1000 --folds 5
```

Walk-forward output tells you the truth:

```
  In-sample  metric    : +23.94  (optimized)
  Out-of-sample metric : +6.70   (honest)
  Verdict              : likely OVERFIT — be skeptical
```

A strategy that's brilliant in-sample but mediocre out-of-sample was tuned to
noise. **Trust the out-of-sample number, not the backtest.**

## How to read the metrics

| Metric | What it tells you |
|---|---|
| **Total return** vs **Buy & hold** | Did the strategy beat just holding? The only goal. |
| **CAGR** | Return annualized — lets you compare across timeframes. |
| **Max drawdown** | Worst peak-to-trough loss. A 60% drawdown is untradeable for most people. |
| **Sharpe** | Return per unit of total volatility. Higher = smoother. |
| **Sortino** | Like Sharpe but only penalizes *downside* volatility. |
| **Calmar** | CAGR ÷ max drawdown. Return earned per unit of pain. |
| **Profit factor** | Gross wins ÷ gross losses. >1 makes money; <1 loses. |
| **Exposure** | Share of time holding a position (idle cash isn't at risk). |
| **Win rate** | Share of round-trips that profited. High return + low win rate = a few big winners. |

## Writing your own strategy

```python
from tradebot.model import Candle, Signal
from tradebot.strategy import Strategy

class MyStrategy(Strategy):
    name = "mine"

    def warmup(self) -> int:
        return 20  # candles needed before signals are valid

    def evaluate(self, history: list[Candle]) -> Signal:
        # history[-1] is the latest closed candle. No look-ahead allowed.
        ...
        return Signal.HOLD
```

Register it in `tradebot/strategies/__init__.py` and it's available everywhere as
`--strategy mine`.

## Going live (read this twice)

Live trading is **disabled by design**. `tradebot live` refuses to run. To trade
real money you must, yourself:

1. Validate in `backtest`, `optimize`, `walkforward`, **and** `paper` first.
2. Implement signed order placement for your exchange in
   `tradebot/broker.py:LiveBroker` and supply API keys with trade permission.
3. Flip the guard in `LiveBroker._require_live_enabled`.
4. Start tiny (`--risk-per-trade 0.01`, `--max-drawdown 0.15`) with money you can
   afford to lose.

That friction is intentional. The fastest way to go broke is to skip steps 1–4.

## Project layout

```
tradebot/
  model.py        Candle, Signal, Trade
  indicators.py   SMA, EMA, RSI, MACD, Bollinger, ATR, Donchian (pure Python)
  strategy.py     Strategy base class
  strategies/     sma, rsi, macd, bollinger, breakout, ensemble + registry
  risk.py         RiskConfig + RiskManager (stops, sizing, circuit breaker)
  broker.py       PaperBroker (simulated) + LiveBroker (guarded stub)
  data.py         live fetch / CSV / synthetic data
  backtest.py     backtesting engine (with integrated risk)
  optimize.py     grid search + walk-forward validation
  metrics.py      return, drawdown, Sharpe, Sortino, Calmar, profit factor
  plot.py         ASCII equity charts (no matplotlib)
  engine.py       live/paper polling loop
  cli.py          command-line interface
tests/
  test_tradebot.py   32 offline tests
```

## Running the tests

```bash
python tests/test_tradebot.py     # no dependencies
# or, with pytest:  python -m pytest
```

## Disclaimer

For education and research only. **Not** financial advice. Trading
cryptocurrencies carries substantial risk of loss. You alone are responsible for
any trades you place with this software.
