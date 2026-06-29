# tradebot

A small, **dependency-free** crypto trading bot framework in pure Python.

The honest path to making money with an automated strategy isn't a magic
algorithm — it's a workflow:

1. **Backtest** a strategy on historical data. Does it actually beat just
   holding the asset, after fees and slippage?
2. **Paper-trade** it live (real prices, fake money). Does it still hold up on
   data it has never seen?
3. **Only then** consider trading real money — with size you can afford to lose.

`tradebot` is built around that workflow. Strategies are pluggable, the
backtester is honest (fees, slippage, no look-ahead), and going live is
deliberately gated behind your own code change so you can't fat-finger real
money on day one.

> ⚠️ **Reality check.** Most trading strategies lose money after costs. This is
> a tool for *finding out whether yours does* — not a guarantee of profit. Never
> trade money you can't afford to lose, and treat a good backtest as a reason to
> paper-trade, not a reason to go all-in.

## Requirements

Python 3.8+. **No third-party packages** — it uses only the standard library.
(`requirements.txt` lists optional dev tools.)

## Quickstart

```bash
cd trading-bot

# Backtest on offline synthetic data (no network needed):
python -m tradebot backtest --strategy sma --synthetic 500 -p fast=10 -p slow=30

# Backtest on real Binance history (needs internet, no API key):
python -m tradebot backtest --strategy sma --symbol BTCUSDT --interval 1h --limit 500

# Compare with the RSI mean-reversion strategy:
python -m tradebot backtest --strategy rsi --symbol ETHUSDT --interval 4h -p period=14

# Paper-trade live prices with fake money (Ctrl-C to stop):
python -m tradebot paper --strategy sma --symbol BTCUSDT --interval 1h --poll 60
```

Example backtest output:

```
Backtest: sma_crossover on 500 candles (BTCUSDT 1h)
  Start equity     : 10,000.00
  End equity       : 26,367.83
  Total return     : +163.68%
  Buy & hold return: +20.26%
  Edge vs hold     : +143.42%
  Max drawdown     : 8.14%
  Sharpe (annual)  : 28.55
  Trades (closed)  : 3
  Win rate         : 100.00%
```

(Those numbers are from synthetic demo data — real markets are much harder. The
point is the *comparison*: a strategy is only interesting if it beats
buy-and-hold after costs.)

## How to read the metrics

| Metric | What it tells you |
|---|---|
| **Total return** vs **Buy & hold** | The only number that matters: did the strategy beat just holding? |
| **Max drawdown** | Worst peak-to-trough loss. A 60% drawdown is untradeable for most people, however good the return. |
| **Sharpe** | Return per unit of risk. Higher is smoother. Negative means you're being paid to lose. |
| **Win rate** | Share of round-trips that profited. High return + low win rate = a few big winners (fragile). |

## Writing your own strategy

A strategy is one method. Drop a file in `tradebot/strategies/` and register it:

```python
from tradebot.model import Candle, Signal
from tradebot.strategy import Strategy

class MyStrategy(Strategy):
    name = "mine"

    def warmup(self) -> int:
        return 20  # candles needed before signals are valid

    def evaluate(self, history: list[Candle]) -> Signal:
        # history[-1] is the latest closed candle. No look-ahead.
        if some_condition(history):
            return Signal.BUY
        if other_condition(history):
            return Signal.SELL
        return Signal.HOLD
```

Then add it to `REGISTRY` in `tradebot/strategies/__init__.py` and it's
available as `--strategy mine`.

## Going live (read this twice)

Live trading is **disabled by design**. `tradebot live` refuses to run. To
trade real money you must, yourself:

1. Validate your strategy in `backtest` **and** `paper` mode first.
2. Open `tradebot/broker.py`, implement signed order placement for your
   exchange in `LiveBroker`, and provide API keys with trade permission.
3. Flip the guard in `LiveBroker._require_live_enabled`.
4. Start with tiny size (`--fraction 0.05`) and money you can afford to lose.

That friction is intentional. The fastest way to go broke is to skip steps 1–4.

## Project layout

```
tradebot/
  model.py        Candle, Signal, Trade
  indicators.py   SMA, EMA, RSI (pure Python)
  strategy.py     Strategy base class
  strategies/     SMA crossover, RSI reversion, registry
  broker.py       PaperBroker (simulated) + LiveBroker (guarded stub)
  data.py         live fetch / CSV / synthetic data
  backtest.py     backtesting engine
  metrics.py      return, drawdown, Sharpe, win rate
  engine.py       live/paper polling loop
  cli.py          command-line interface
tests/
  test_tradebot.py   14 offline tests
```

## Running the tests

```bash
python tests/test_tradebot.py     # no dependencies
# or, if you have pytest:
python -m pytest
```

## Disclaimer

This software is for education and research. It is **not** financial advice.
Trading cryptocurrencies carries substantial risk of loss. You are solely
responsible for any trades you place with it.
