# tradebot

[![CI](https://github.com/kyletrunkett2002/claude-code/actions/workflows/trading-bot-ci.yml/badge.svg)](https://github.com/kyletrunkett2002/claude-code/actions/workflows/trading-bot-ci.yml)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/runtime%20deps-0-brightgreen)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-70%20passing-brightgreen)](tests/test_tradebot.py)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A small but serious **crypto trading bot framework** in pure Python — zero
third-party dependencies, fully offline-capable, 70 tests.

Backtest long *and* short, across **multiple coins** at once, on data from
**Binance, Coinbase, or Kraken**, with real **risk management**, parameter
**optimization + walk-forward validation**, ASCII **charts and heatmaps**, and
reproducible **config-file experiments**.

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

## Install

Run it straight from the repo (no install needed):

```bash
cd trading-bot
python -m tradebot --help
```

Or install it so `tradebot` becomes a command anywhere:

```bash
cd trading-bot
pip install -e .
tradebot --help
```

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

# 5. Backtest LONG + SHORT (stop-and-reverse) on Coinbase data:
python -m tradebot backtest -s sma --source coinbase --symbol BTC-USD --allow-short

# 6. Backtest one strategy across a basket of coins (diversification):
python -m tradebot portfolio --symbols BTCUSDT,ETHUSDT,SOLUSDT --synthetic 800 -s sma

# 7. Replay a saved experiment from a config file:
python -m tradebot run examples/portfolio.config.json

# 8. Paper-trade live prices with fake money (Ctrl-C to stop):
python -m tradebot paper -s sma --symbol BTCUSDT --interval 1h --poll 60
```

## Data sources

Pass `--source binance|coinbase|kraken` (default `binance`); all are public, no
API key. Mind each exchange's symbol format: Binance `BTCUSDT`, Coinbase
`BTC-USD`, Kraken `XBTUSD`. Or work fully offline with `--synthetic N`, or load
your own `--csv file.csv`.

### Caching real data for reproducible, offline backtests

Download once, then backtest forever without hitting the network:

```bash
# Fetch and cache 1000 BTC candles as CSV:
python -m tradebot download --symbol BTCUSDT --interval 1h --limit 1000 --cache data_cache

# Now every command can read the cache (no network, identical data each run):
python -m tradebot backtest -s supertrend --symbol BTCUSDT --interval 1h --cache data_cache
python -m tradebot optimize -s supertrend --symbol BTCUSDT --interval 1h --cache data_cache
```

`--cache DIR` reads the cached CSV if present, otherwise fetches and saves it.
Reproducible results are the bedrock of trustworthy backtesting.

### Practice finding an edge (realistic market simulator)

`data.realistic_market()` generates synthetic prices with **fat tails and
volatility clustering** (a GARCH process — calm and turbulent regimes, like real
crypto) plus a **tunable, known edge** via return autocorrelation:

```python
from tradebot.data import realistic_market, save_csv
save_csv(realistic_market(n=2500, seed=1, momentum=0.0),  "noedge.csv")  # random walk
save_csv(realistic_market(n=2500, seed=1, momentum=0.40), "edge.csv")    # real momentum edge
```

Because you know the ground truth, it's the perfect practice range: run
`optimize → walkforward → montecarlo` on each and watch the tools correctly flag
the random walk as overfit while confirming the real edge survives out of sample.
That skill — telling a true edge from a lucky backtest — is the whole game.

## Strategies

| Name | Style | Idea |
|---|---|---|
| `sma` | trend | Fast/slow moving-average crossover |
| `macd` | trend | MACD line crossing its signal line |
| `breakout` | trend | Donchian channel breakout (the "turtle" entry) |
| `supertrend` | trend | ATR-based SuperTrend flip (a trader favourite) |
| `rsi` | mean-reversion | Buy oversold, sell overbought |
| `bollinger` | both | Band reversion *or* breakout (`-p mode=breakout`) |
| `vwap` | mean-reversion | Buy when price stretches below rolling VWAP |
| `stochastic` | mean-reversion | %K/%D crossover in oversold/overbought zones |
| `ensemble` | meta | Majority vote across several strategies |
| `regime` | meta | Reads trend strength (ADX) and switches between a trend and a reversion strategy |
| `mtf` | meta | Multi-timeframe filter: only takes trades aligned with the higher-timeframe trend |

The `regime` strategy deserves a callout: no single approach works in every
market, so it measures trend strength with ADX and **routes** each decision —
trend-following when the market trends, mean-reversion when it chops. Adapting to
the regime is one of the most robust ideas in systematic trading.

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

| `--trailing-stop 0.05` | A stop that ratchets in your favour — trails the best price by 5% and only ever tightens, locking in profit as a trade runs |

`--risk-per-trade` is volatility-aware position sizing — the single most
important habit separating traders who survive from those who don't.

### Short-selling

Add `--allow-short` to turn any strategy into a **stop-and-reverse** system: a
SELL with no long open becomes a *short* (profit when price falls), and a BUY
covers it before going long. Stops and targets automatically invert for shorts.
Useful for strategies that should profit in down-trends, not just sit in cash.

### Multi-coin portfolios

```bash
python -m tradebot portfolio --symbols BTCUSDT,ETHUSDT,SOLUSDT --synthetic 800 \
    -s sma --risk-per-trade 0.02 --stop-loss 0.05
```

Capital is split equally across symbols, each backtested independently, then
combined into one portfolio equity curve with a per-symbol breakdown. Spreading
the same edge over uncorrelated coins is the closest thing to a free lunch:
expect a *lower* portfolio drawdown than any single coin.

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

### Parameter heatmaps

For a 2-parameter search, add `--heatmap X,Y` to *see* the result:

```bash
python -m tradebot optimize -s sma --synthetic 800 --heatmap fast,slow
```

```
         5 10 15 20    [fast]
    30 @@ @@ %% %%
    50 ## ## ## **
    80 ++ == == --
   120 -- :: ..
  [slow]
```

A healthy strategy shows a **broad bright region** — many nearby settings work,
so the edge is robust. A single bright cell in a sea of dark is a cherry-picked
fluke that won't survive live trading.

### Monte Carlo robustness (how lucky was your backtest?)

A backtest is *one* sequence of trades. The future will deal the same edge in a
different order — and a strategy can look great purely because its winners
happened to land in a lucky run. Monte Carlo resamples your trades thousands of
times to show the *distribution* of what could happen:

```bash
python -m tradebot montecarlo -s sma --symbol BTCUSDT --interval 1h --sims 3000
```

```
  Final return  p5/p50/p95 : +12.0% / +48.0% / +95.0%
  Max drawdown  p50/p95    : 14.0% / 31.0%
  Probability of net loss  : 18.0%
  Probability of >30% drawdown : 6.0%
```

Now you can reason about risk honestly: *"1-in-5 chance this loses money, 1-in-16
chance of a 30%+ drawdown."* That beats a single hero number every time.
(On `--synthetic` data these come out unrealistically rosy — run it on real
exchange data for meaningful probabilities.)

### Shareable HTML reports

```bash
python -m tradebot backtest -s supertrend --symbol BTCUSDT --html report.html
```

Writes a single self-contained `.html` file — inline SVG equity curve (strategy
vs buy-and-hold), the full metrics table, and the trade log. No JavaScript, no
external assets; open it in any browser or email it to someone.

### Config-file experiments

Save a whole setup as JSON and replay it reproducibly:

```bash
python -m tradebot run examples/portfolio.config.json
```

The config picks the `mode` (backtest / portfolio / optimize / walkforward /
compare), data source, strategy, parameters, risk rules and plotting — so your
validated experiments live in version control, not shell history. See
`examples/portfolio.config.json`.

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
  indicators.py   SMA, EMA, RSI, MACD, Bollinger, ATR, Donchian, ADX, VWAP,
                  SuperTrend, Stochastic
  strategy.py     Strategy base class
  strategies/     sma, rsi, macd, bollinger, breakout, supertrend, vwap,
                  stochastic, ensemble, regime, mtf + registry
  risk.py         RiskConfig + RiskManager (stops, trailing, sizing, breaker)
  broker.py       PaperBroker (long + short) + LiveBroker (guarded stub)
  data.py         Binance/Coinbase/Kraken fetch, caching, resampling, CSV, synthetic
  backtest.py     backtesting engine (risk + shorting integrated)
  portfolio.py    multi-coin portfolio backtesting
  optimize.py     grid search + walk-forward validation
  monte_carlo.py  bootstrap robustness testing
  metrics.py      return, drawdown, Sharpe, Sortino, Calmar, profit factor
  plot.py         ASCII equity charts + parameter heatmaps (no matplotlib)
  report.py       self-contained HTML reports (inline SVG)
  config.py       JSON config-file runner
  engine.py       live/paper polling loop
  cli.py          command-line interface
examples/
  portfolio.config.json   sample experiment config
tests/
  test_tradebot.py   70 offline tests
pyproject.toml    pip-installable (`tradebot` command)
LICENSE           MIT
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
