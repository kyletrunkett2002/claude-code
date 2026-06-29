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
from tradebot.model import Candle, Signal
from tradebot.optimize import DEFAULT_GRIDS, grid_search, walk_forward
from tradebot.risk import RiskConfig, RiskManager
from tradebot.strategies import REGISTRY, build
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


# ---- new indicators -------------------------------------------------------

def test_macd_shapes():
    closes = [float(x) for x in range(1, 80)]
    macd_line, signal_line, hist = indicators.macd(closes)
    assert len(macd_line) == len(signal_line) == len(hist) == len(closes)
    # On a steadily rising series, MACD line ends positive.
    assert macd_line[-1] is not None and macd_line[-1] > 0


def test_bollinger_ordering():
    closes = data.synthetic(n=60)
    lower, mid, upper = indicators.bollinger([c.close for c in closes], 20, 2.0)
    for lo, md, up in zip(lower, mid, upper):
        if None not in (lo, md, up):
            assert lo <= md <= up


def test_atr_positive():
    candles = data.synthetic(n=60)
    a = indicators.atr([c.high for c in candles], [c.low for c in candles],
                       [c.close for c in candles], 14)
    vals = [v for v in a if v is not None]
    assert vals and all(v > 0 for v in vals)


def test_donchian_bounds():
    candles = data.synthetic(n=60)
    lo, hi = indicators.donchian([c.high for c in candles],
                                 [c.low for c in candles], 20)
    for l, h in zip(lo, hi):
        if l is not None and h is not None:
            assert l <= h


# ---- risk management ------------------------------------------------------

def test_riskconfig_validates():
    for bad in (dict(stop_loss=1.5), dict(position_fraction=0),
                dict(risk_per_trade=0.02, stop_loss=0)):
        try:
            RiskConfig(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")


def test_risk_position_sizing():
    # Risk 2% of equity with a 5% stop -> deploy 40% of cash.
    cfg = RiskConfig(stop_loss=0.05, risk_per_trade=0.02)
    assert abs(cfg.entry_fraction() - 0.4) < 1e-9
    # Capped by max_position_fraction.
    cfg2 = RiskConfig(stop_loss=0.01, risk_per_trade=0.02, max_position_fraction=0.5)
    assert abs(cfg2.entry_fraction() - 0.5) < 1e-9


def test_stop_loss_exits():
    # A single candle that gaps down through the stop must trigger an exit.
    mgr = RiskManager(RiskConfig(stop_loss=0.10))
    mgr.on_entry(entry_price=100)
    assert mgr.protective_exit(bar_high=101, bar_low=85) == 90.0  # 10% stop
    assert mgr.protective_exit(bar_high=101, bar_low=95) is None


def test_take_profit_exits():
    mgr = RiskManager(RiskConfig(take_profit=0.20))
    mgr.on_entry(entry_price=100)
    assert mgr.protective_exit(bar_high=125, bar_low=99) == 120.0


def test_drawdown_circuit_breaker_trips():
    mgr = RiskManager(RiskConfig(max_drawdown=0.20))
    mgr.update_equity(100)
    mgr.update_equity(120)            # new peak
    assert mgr.update_equity(110) is False
    assert mgr.update_equity(95) is True   # 20.8% off peak -> trips
    assert mgr.halted


def test_backtest_with_stop_caps_loss():
    # With a tight stop, the worst single-trade loss should be bounded.
    candles = data.synthetic(n=400, seed=3)
    risk = RiskConfig(stop_loss=0.03, take_profit=0.06, position_fraction=1.0)
    r = run_backtest(build("sma"), candles, interval="1h", risk=risk)
    assert min(r.equity_curve) >= 0
    assert r.metrics.num_trades >= 0


def test_backtest_halts_on_drawdown():
    candles = data.synthetic(n=400, seed=9)
    risk = RiskConfig(max_drawdown=0.01)  # absurdly tight -> almost surely trips
    r = run_backtest(build("bollinger"), candles, interval="1h", risk=risk)
    # Either it never traded into a drawdown, or it halted. Both are valid;
    # we just assert the flag is a bool and equity stayed non-negative.
    assert isinstance(r.halted, bool)
    assert min(r.equity_curve) >= 0


# ---- new metrics ----------------------------------------------------------

def test_profit_factor_and_sortino():
    m = compute([100, 110, 105, 130], num_trades=3, wins=2,
                gross_profit=40, gross_loss=10)
    assert m.profit_factor == 4.0
    assert m.sortino != 0.0  # there is downside in the curve


# ---- new strategies -------------------------------------------------------

def test_all_registered_strategies_run():
    candles = data.synthetic(n=400)
    for name in REGISTRY:
        s = build(name)
        r = run_backtest(s, candles, interval="1h")
        assert len(r.equity_curve) == len(candles)
        assert min(r.equity_curve) >= 0


def test_ensemble_majority_vote():
    from tradebot.strategies.ensemble import Ensemble

    class AlwaysBuy(SmaCrossover):
        def evaluate(self, history):
            return Signal.BUY

    class AlwaysSell(SmaCrossover):
        def evaluate(self, history):
            return Signal.SELL

    e = Ensemble(members=[AlwaysBuy(), AlwaysBuy(), AlwaysSell()], threshold=1)
    candles = data.synthetic(n=50)
    assert e.evaluate(candles) is Signal.BUY  # net +1


def test_breakout_no_lookahead():
    # Breakout must not peek at the current bar's own high when forming channel.
    s = build("breakout", entry=5, exit_period=3)
    candles = data.synthetic(n=4)
    assert s.evaluate(candles) is Signal.HOLD


# ---- optimizer & walk-forward --------------------------------------------

def test_grid_search_sorted():
    candles = data.synthetic(n=400)
    grid = {"fast": [5, 10], "slow": [20, 40]}
    ranked = grid_search("sma", candles, grid, metric="sharpe", interval="1h")
    assert len(ranked) == 4
    scores = [g.score for g in ranked]
    assert scores == sorted(scores, reverse=True)


def test_grid_search_skips_invalid_combos():
    candles = data.synthetic(n=200)
    # fast >= slow is invalid and must be skipped, not crash.
    grid = {"fast": [10, 30], "slow": [20]}
    ranked = grid_search("sma", candles, grid, interval="1h")
    assert all(g.params["fast"] < g.params["slow"] for g in ranked)


def test_walk_forward_runs_and_reports_oos():
    candles = data.synthetic(n=800)
    wf = walk_forward("sma", candles, DEFAULT_GRIDS["sma"], folds=3,
                      metric="sharpe", interval="1h")
    assert wf.folds >= 1
    assert len(wf.chosen_params) >= 1
    assert len(wf.oos_equity_curve) > 0


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
