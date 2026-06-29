# Contributing to tradebot

Thanks for your interest! tradebot is intentionally small and **dependency-free**
— that constraint is a feature, so please keep it.

## Ground rules

1. **No runtime dependencies.** The `tradebot` package must import and run using
   only the Python standard library. Dev-only tools (e.g. `pytest`) are fine and
   live under the `dev` optional-dependencies group.
2. **Everything stays testable offline.** Use the synthetic data generator
   (`tradebot.data.synthetic`) so tests never touch the network.
3. **No look-ahead in strategies.** A strategy's `evaluate(history)` may only use
   `history` up to and including the latest candle — never future data.

## Getting set up

```bash
cd trading-bot
pip install -e ".[dev]"
```

## Running the tests

```bash
python tests/test_tradebot.py     # stdlib-only runner, no pytest needed
python -m pytest -q                # or via pytest
```

All tests must pass on Python 3.8–3.12 (that's what CI checks).

## Adding a strategy

1. Create `tradebot/strategies/your_strategy.py` subclassing `Strategy` and
   implementing `evaluate()` (and `warmup()` if it needs history).
2. Register it in `tradebot/strategies/__init__.py` (`REGISTRY`).
3. Optionally add a parameter grid in `tradebot/optimize.py` (`DEFAULT_GRIDS`)
   so it works with `optimize` and `walkforward`.
4. Add a test that backtests it on synthetic data and asserts the equity curve
   never goes negative.

## Style

Match the surrounding code: clear names, docstrings that explain *why* (the
trading rationale), and comments only where the intent isn't obvious. Keep the
honest-about-risk tone — this project never promises profit.

## A note on scope

tradebot is for education and research. Please don't add anything that
encourages reckless real-money trading; the `LiveBroker` guard exists on purpose.
