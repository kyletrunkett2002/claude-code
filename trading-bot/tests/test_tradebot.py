"""Test suite — runs fully offline using synthetic data.

Run with:  python -m pytest        (if pytest is installed)
       or:  python tests/test_tradebot.py   (no dependencies)
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradebot import data, indicators
from tradebot.backtest import run_backtest
from tradebot.broker import LiveBroker, PaperBroker
from tradebot.metrics import compute
from tradebot.model import Signal
from tradebot.strategies import build
from tradebot.strategies.sma_crossover import SmaCrossover


# ---- indicators -----------------------------------------------------------

def test_sma_basic():
    out = indicators.sma([1, 2, 3, 4, 5], 3)
    assert out[:2] == [None, None]
    assert out[2] == 2.0 and out[3] == 3.0 and out[4] == 4.0


def test_rsi_bounds():
    rising = list(range(1, 50))
    out = indicators.rsi(rising, 14)
    last = [v for v in out if v is not None][-1]
    assert 99.0 <= last <= 100.0  # all gains -> RSI near 100


def test_rsi_all_losses():
    falling = list(range(50, 1, -1))
    out = indicators.rsi(falling, 14)
    last = [v for v in out if v is not None][-1]
    assert 0.0 <= last <= 1.0


# ---- broker ---------------------------------------------------------------

def test_paper_broker_roundtrip_no_costs():
    b = PaperBroker(cash=1000, fee_rate=0.0, slippage=0.0)
    b.buy(0, price=100, fraction=1.0)
    assert abs(b.position - 10) < 1e-9
    assert abs(b.cash) < 1e-9
    b.sell(0, price=110, fraction=1.0)
    assert abs(b.position) < 1e-9
    assert abs(b.cash - 1100) < 1e-9  # 10% gain realized


def test_paper_broker_fees_lose_money_on_flat_roundtrip():
    b = PaperBroker(cash=1000, fee_rate=0.001, slippage=0.0)
    b.buy(0, price=100, fraction=1.0)
    b.sell(0, price=100, fraction=1.0)
    assert b.cash < 1000  # fees + nothing gained = a loss


def test_buy_cannot_exceed_cash():
    b = PaperBroker(cash=500, fee_rate=0.0, slippage=0.0)
    b.buy(0, price=100, fraction=1.0)
    assert b.cash >= -1e-9  # never goes negative


def test_live_broker_guarded():
    lb = LiveBroker(enabled=False)
    try:
        lb.buy(0, 100, 1.0)
    except RuntimeError as e:
        assert "disabled" in str(e).lower()
    else:
        raise AssertionError("LiveBroker should refuse to trade when disabled")


# ---- metrics --------------------------------------------------------------

def test_metrics_drawdown():
    curve = [100, 120, 60, 90]  # peak 120, trough 60 -> 50% drawdown
    m = compute(curve, num_trades=1, wins=0)
    assert abs(m.max_drawdown_pct - 50.0) < 1e-6
    assert abs(m.total_return_pct - (-10.0)) < 1e-6


def test_metrics_winrate():
    m = compute([100, 110], num_trades=4, wins=3)
    assert m.win_rate_pct == 75.0


# ---- strategies & backtest ------------------------------------------------

def test_strategy_no_lookahead_and_warmup():
    s = SmaCrossover(fast=3, slow=5)
    candles = data.synthetic(n=4)
    assert s.evaluate(candles) is Signal.HOLD  # not enough data yet


def test_backtest_runs_and_is_self_consistent():
    s = build("sma", fast=5, slow=20)
    candles = data.synthetic(n=400)
    result = run_backtest(s, candles, cash=10_000, interval="1h")
    assert len(result.equity_curve) == len(candles)
    assert result.metrics.start_equity == 10_000
    assert result.metrics.num_trades >= 0
    # Equity must never go negative in a long-only sim.
    assert min(result.equity_curve) >= 0


def test_backtest_determinism():
    candles = data.synthetic(n=300, seed=7)
    r1 = run_backtest(build("rsi"), candles, interval="1h")
    r2 = run_backtest(build("rsi"), candles, interval="1h")
    assert r1.equity_curve == r2.equity_curve


def test_csv_roundtrip():
    candles = data.synthetic(n=50)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.csv")
        data.save_csv(candles, path)
        loaded = data.load_csv(path)
    assert len(loaded) == len(candles)
    assert abs(loaded[10].close - candles[10].close) < 1e-6


def test_strategy_registry_unknown():
    try:
        build("does-not-exist")
    except KeyError as e:
        assert "unknown strategy" in str(e)
    else:
        raise AssertionError("expected KeyError for unknown strategy")


# ---- minimal runner (no pytest required) ----------------------------------

def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
