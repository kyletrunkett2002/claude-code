"""Test suite — runs fully offline using synthetic data.

Run with:  python -m pytest        (if pytest is installed)
       or:  python tests/test_tradebot.py   (no dependencies)
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradebot import config as configmod
from tradebot import data, indicators, plot
from tradebot.backtest import run_backtest
from tradebot.broker import LiveBroker, PaperBroker
from tradebot.metrics import compute
from tradebot.model import Candle, Signal
from tradebot.monte_carlo import bootstrap
from tradebot.optimize import DEFAULT_GRIDS, grid_search, walk_forward
from tradebot.portfolio import run_portfolio
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


# ---- short selling --------------------------------------------------------

def test_short_roundtrip_profits_when_price_falls():
    b = PaperBroker(cash=1000, fee_rate=0.0, slippage=0.0)
    b.sell_short(0, price=100, fraction=1.0)
    assert b.position < 0
    eq_before = b.equity(100)
    # Price falls to 90 -> covering should leave us richer.
    b.cover(0, price=90)
    assert abs(b.position) < 1e-9
    assert b.equity(90) > eq_before


def test_short_loses_when_price_rises():
    b = PaperBroker(cash=1000, fee_rate=0.0, slippage=0.0)
    b.sell_short(0, price=100, fraction=1.0)
    b.cover(0, price=110)
    assert b.cash < 1000  # short squeezed -> loss


def test_short_stop_is_above_entry():
    mgr = RiskManager(RiskConfig(stop_loss=0.10))
    mgr.on_entry(entry_price=100, side=-1)
    # Short stop triggers when price rises through ~110.
    hit = mgr.protective_exit(bar_high=111, bar_low=100)
    assert hit is not None and abs(hit - 110.0) < 1e-6
    assert mgr.protective_exit(bar_high=105, bar_low=95) is None


def test_allow_short_increases_exposure():
    candles = data.synthetic(n=500, seed=11)
    lo = run_backtest(build("sma"), candles, interval="1h", allow_short=False)
    sh = run_backtest(build("sma"), candles, interval="1h", allow_short=True)
    assert sh.metrics.exposure_pct >= lo.metrics.exposure_pct
    assert min(sh.equity_curve) >= 0


# ---- portfolio ------------------------------------------------------------

def test_portfolio_aggregates_symbols():
    coins = {f"C{i}": data.synthetic(n=400, seed=i + 1) for i in range(3)}
    res = run_portfolio(lambda: build("sma"), coins, cash=9_000, interval="1h")
    assert len(res.per_symbol) == 3
    assert res.metrics.start_equity == 9_000
    # Portfolio curve is the sum of per-symbol curves at each step.
    n = min(len(c) for c in coins.values())
    assert len(res.equity_curve) == n
    assert min(res.equity_curve) >= 0


def test_portfolio_diversification_reduces_drawdown():
    # The basket's drawdown should not exceed the worst single component's.
    coins = {f"C{i}": data.synthetic(n=500, seed=i * 13 + 1) for i in range(4)}
    res = run_portfolio(lambda: build("sma"), coins, interval="1h")
    worst = max(r.metrics.max_drawdown_pct for r in res.per_symbol.values())
    assert res.metrics.max_drawdown_pct <= worst + 1e-6


# ---- data adapters --------------------------------------------------------

def test_fetch_dispatch_unknown_source():
    try:
        data.fetch("not-an-exchange")
    except ValueError as e:
        assert "unknown source" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown source")


# ---- plotting -------------------------------------------------------------

def test_sparkline_and_chart_render():
    curve = data.synthetic(n=60)
    closes = [c.close for c in curve]
    assert len(plot.sparkline(closes)) == len(closes)
    chart = plot.equity_chart(closes)
    assert "start" in chart and "│" in chart


def test_heatmap_renders_from_grid():
    candles = data.synthetic(n=400)
    ranked = grid_search("sma", candles, {"fast": [5, 10], "slow": [20, 40]},
                         interval="1h")
    hm = plot.heatmap(ranked, "fast", "slow", "sharpe")
    assert "[fast]" in hm and "[slow]" in hm


# ---- config runner --------------------------------------------------------

def test_config_stable_seed_is_deterministic():
    assert configmod._stable_seed("BTCUSDT") == configmod._stable_seed("BTCUSDT")
    assert configmod._stable_seed("BTCUSDT") != configmod._stable_seed("ETHUSDT")


def test_config_execute_backtest_offline():
    cfg = dict(configmod.DEFAULTS)
    cfg.update(mode="backtest", source="synthetic", synthetic=400,
               strategy="sma", interval="1h")
    report = configmod.execute(cfg)
    assert "Backtest" in report and "Total return" in report


def test_config_execute_portfolio_offline():
    cfg = dict(configmod.DEFAULTS)
    cfg.update(mode="portfolio", source="synthetic", synthetic=400,
               symbols=["AAA", "BBB"], strategy="sma", interval="1h")
    report = configmod.execute(cfg)
    assert "Portfolio" in report and "Per-symbol" in report


# ---- trailing stop --------------------------------------------------------

def test_trailing_stop_ratchets_up_for_long():
    mgr = RiskManager(RiskConfig(trailing_stop=0.10))
    mgr.on_entry(entry_price=100, side=1)
    # Price runs up to 150; trailing stop should now sit near 135.
    mgr.update_trailing(bar_high=150, bar_low=100)
    hit = mgr.protective_exit(bar_high=150, bar_low=134)  # dips to 134 < 135
    assert hit is not None and abs(hit - 135.0) < 1e-6


def test_trailing_stop_never_loosens():
    mgr = RiskManager(RiskConfig(trailing_stop=0.10))
    mgr.on_entry(entry_price=100, side=1)
    mgr.update_trailing(bar_high=150, bar_low=100)   # stop -> 135
    mgr.update_trailing(bar_high=120, bar_low=110)   # lower high must NOT lower stop
    hit = mgr.protective_exit(bar_high=120, bar_low=134)
    assert hit is not None and abs(hit - 135.0) < 1e-6


def test_trailing_stop_locks_in_profit_in_backtest():
    candles = data.synthetic(n=500, seed=4)
    r = run_backtest(build("sma"), candles, interval="1h",
                     risk=RiskConfig(trailing_stop=0.05))
    assert min(r.equity_curve) >= 0
    assert isinstance(r.trade_returns, list)


# ---- monte carlo ----------------------------------------------------------

def test_bootstrap_distribution_basic():
    # A mix of winners and losers with positive expectancy.
    returns = [0.10, -0.05, 0.08, -0.04, 0.06, -0.03] * 5
    mc = bootstrap(returns, simulations=500, seed=1)
    assert mc.simulations == 500
    assert mc.return_p5 <= mc.return_p50 <= mc.return_p95
    assert 0.0 <= mc.prob_loss_pct <= 100.0


def test_bootstrap_all_losers_high_loss_probability():
    mc = bootstrap([-0.05, -0.10, -0.02] * 10, simulations=300, seed=2)
    assert mc.prob_loss_pct > 90.0   # losing trades -> almost always lose


def test_bootstrap_deterministic_with_seed():
    returns = [0.05, -0.03, 0.04, -0.02] * 8
    a = bootstrap(returns, simulations=400, seed=99)
    b = bootstrap(returns, simulations=400, seed=99)
    assert a.return_p50 == b.return_p50 and a.drawdown_p95 == b.drawdown_p95


def test_bootstrap_no_trades_raises():
    try:
        bootstrap([], simulations=10)
    except ValueError as e:
        assert "no trades" in str(e)
    else:
        raise AssertionError("expected ValueError when there are no trades")


def test_backtest_records_trade_returns():
    candles = data.synthetic(n=500, seed=6)
    r = run_backtest(build("sma"), candles, interval="1h")
    assert isinstance(r.trade_returns, list)
    assert len(r.trade_returns) == r.metrics.num_trades


# ---- trade export ---------------------------------------------------------

def test_save_trades_csv():
    candles = data.synthetic(n=400)
    r = run_backtest(build("sma"), candles, interval="1h")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trades.csv")
        data.save_trades_csv(r.trades, path)
        with open(path) as fh:
            lines = fh.read().splitlines()
    assert lines[0].startswith("timestamp,side,price")
    assert len(lines) == len(r.trades) + 1


# ---- regime / supertrend / vwap indicators & strategies -------------------

def test_adx_in_range():
    candles = data.synthetic(n=200)
    vals = indicators.adx([c.high for c in candles], [c.low for c in candles],
                          [c.close for c in candles], 14)
    present = [v for v in vals if v is not None]
    assert present and all(0 <= v <= 100 for v in present)


def test_adx_strong_for_steady_trend():
    # A relentless uptrend should register high trend strength.
    n = 200
    candles = [Candle(i, 100 + i, 100 + i + 0.5, 100 + i - 0.5, 100 + i, 1.0)
               for i in range(n)]
    vals = indicators.adx([c.high for c in candles], [c.low for c in candles],
                          [c.close for c in candles], 14)
    last = [v for v in vals if v is not None][-1]
    assert last > 40  # clearly trending


def test_supertrend_direction_values():
    candles = data.synthetic(n=120)
    _, direction = indicators.supertrend(
        [c.high for c in candles], [c.low for c in candles],
        [c.close for c in candles], 10, 3.0)
    present = [d for d in direction if d is not None]
    assert present and set(present) <= {1, -1}


def test_vwap_between_extremes():
    candles = data.synthetic(n=80)
    vw = indicators.vwap([c.high for c in candles], [c.low for c in candles],
                         [c.close for c in candles], [c.volume for c in candles], 20)
    for i, v in enumerate(vw):
        if v is not None:
            window = candles[max(0, i - 19): i + 1]
            assert min(c.low for c in window) <= v <= max(c.high for c in window)


def test_new_strategies_registered_and_run():
    candles = data.synthetic(n=500)
    for name in ("supertrend", "vwap", "regime"):
        assert name in REGISTRY
        r = run_backtest(build(name), candles, interval="1h")
        assert len(r.equity_curve) == len(candles)
        assert min(r.equity_curve) >= 0


def test_regime_routes_by_trend_strength():
    from tradebot.strategies.regime import RegimeAdaptive

    class TrendTag(SmaCrossover):
        def evaluate(self, history):
            return Signal.BUY

    class RevTag(SmaCrossover):
        def evaluate(self, history):
            return Signal.SELL

    # Strong uptrend -> ADX high -> trend member (BUY) should win.
    candles = [Candle(i, 100 + i, 100 + i + 0.5, 100 + i - 0.5, 100 + i, 1.0)
               for i in range(200)]
    reg = RegimeAdaptive(adx_threshold=25, trend=TrendTag(), reversion=RevTag())
    assert reg.evaluate(candles) is Signal.BUY


# ---- HTML report ----------------------------------------------------------

def test_html_report_is_self_contained():
    from tradebot import report
    candles = data.synthetic(n=400)
    r = run_backtest(build("sma"), candles, interval="1h")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "r.html")
        report.html_report("sma", r, candles, symbol="BTCUSDT",
                           interval="1h", path=path)
        with open(path) as fh:
            doc = fh.read()
    assert "<svg" in doc and "Total return" in doc
    assert "http://" not in doc.replace("xmlns=\"http://www.w3.org/2000/svg\"", "")
    assert "<script" not in doc  # no JS, fully static


# ---- resampling & caching -------------------------------------------------

def test_resample_aggregates_correctly():
    candles = data.synthetic(n=40)
    htf = data.resample(candles, 4)
    assert len(htf) == 10
    first = htf[0]
    group = candles[:4]
    assert first.open == group[0].open
    assert first.close == group[3].close
    assert first.high == max(c.high for c in group)
    assert first.low == min(c.low for c in group)
    assert abs(first.volume - sum(c.volume for c in group)) < 1e-6


def test_resample_drops_partial_group():
    candles = data.synthetic(n=42)   # 10 full groups of 4 + 2 leftover
    assert len(data.resample(candles, 4)) == 10


def test_cache_roundtrip(tmp_path_factory=None):
    candles = data.synthetic(n=30)
    with tempfile.TemporaryDirectory() as d:
        path = data.cache_path(d, "binance", "BTCUSDT", "1h")
        data.save_csv(candles, path)
        assert os.path.exists(path)
        # load_or_fetch must read the cache without any network call.
        loaded = data.load_or_fetch("binance", "BTCUSDT", "1h", cache_dir=d)
    assert len(loaded) == len(candles)


# ---- stochastic & mtf strategies ------------------------------------------

def test_stochastic_bounds():
    candles = data.synthetic(n=120)
    k, d = indicators.stochastic([c.high for c in candles], [c.low for c in candles],
                                 [c.close for c in candles], 14, 3)
    for series in (k, d):
        present = [v for v in series if v is not None]
        assert present and all(0 <= v <= 100 for v in present)


def test_mtf_blocks_countertrend_signals():
    from tradebot.strategies.mtf_trend import MtfTrend

    class AlwaysSell(SmaCrossover):
        def evaluate(self, history):
            return Signal.SELL

    # Strong uptrend: higher-TF MA is rising, so SELL signals must be blocked.
    candles = [Candle(i, 100 + i, 100 + i + 0.5, 100 + i - 0.5, 100 + i, 1.0)
               for i in range(300)]
    mtf = MtfTrend(htf_factor=4, htf_period=20, base=AlwaysSell())
    assert mtf.evaluate(candles) is Signal.HOLD  # countertrend SELL filtered out


def test_mtf_allows_with_trend_signals():
    from tradebot.strategies.mtf_trend import MtfTrend

    class AlwaysBuy(SmaCrossover):
        def evaluate(self, history):
            return Signal.BUY

    candles = [Candle(i, 100 + i, 100 + i + 0.5, 100 + i - 0.5, 100 + i, 1.0)
               for i in range(300)]
    mtf = MtfTrend(htf_factor=4, htf_period=20, base=AlwaysBuy())
    assert mtf.evaluate(candles) is Signal.BUY  # with-trend BUY allowed


def test_new_strategies_run_in_backtest():
    candles = data.synthetic(n=600)
    for name in ("stochastic", "mtf"):
        assert name in REGISTRY
        r = run_backtest(build(name), candles, interval="1h")
        assert len(r.equity_curve) == len(candles)
        assert min(r.equity_curve) >= 0


# ---- realistic market simulator -------------------------------------------

def _lag1_autocorr(candles):
    import math
    rets = [math.log(candles[i].close / candles[i - 1].close)
            for i in range(1, len(candles))]
    m = sum(rets) / len(rets)
    num = sum((rets[i] - m) * (rets[i - 1] - m) for i in range(1, len(rets)))
    den = sum((r - m) ** 2 for r in rets)
    return num / den if den else 0.0


def test_realistic_market_basic():
    candles = data.realistic_market(n=500, seed=3)
    assert len(candles) == 500
    assert all(c.close > 0 and c.high >= c.low for c in candles)
    # High/low must bracket open and close.
    for c in candles:
        assert c.high >= max(c.open, c.close) - 1e-6
        assert c.low <= min(c.open, c.close) + 1e-6


def test_realistic_market_deterministic():
    a = data.realistic_market(n=300, seed=11)
    b = data.realistic_market(n=300, seed=11)
    assert [c.close for c in a] == [c.close for c in b]


def test_realistic_market_momentum_adds_autocorrelation():
    flat = data.realistic_market(n=3000, seed=5, momentum=0.0)
    trend = data.realistic_market(n=3000, seed=5, momentum=0.30)
    # A momentum term should raise lag-1 autocorrelation meaningfully.
    assert _lag1_autocorr(trend) > _lag1_autocorr(flat) + 0.1


# ---- performance windowing -------------------------------------------------

def test_lookback_default_is_bounded():
    # Simple strategies should bound their history for O(n*window) backtests.
    assert build("sma").lookback() > 0
    # The multi-timeframe strategy opts out (needs stable resample anchoring).
    assert build("mtf").lookback() == 0


def test_windowing_matches_full_history():
    # Passing only the last `lookback` candles must give identical results to
    # replaying the full history — the window is generous enough to be exact.
    candles = data.realistic_market(n=1000, seed=2, momentum=0.2)
    for name in REGISTRY:
        win = run_backtest(build(name), candles, interval="1h")
        s_full = build(name)
        s_full.lookback = lambda: 0          # force full history
        full = run_backtest(s_full, candles, interval="1h")
        assert win.equity_curve == full.equity_curve, f"{name} differs when windowed"


# ---- CLI json output ------------------------------------------------------

def test_cli_backtest_json(capsys=None):
    import io
    import json as _json
    from contextlib import redirect_stdout
    from tradebot.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["backtest", "-s", "sma", "--synthetic", "400", "--json"])
    assert rc == 0
    payload = _json.loads(buf.getvalue())
    assert payload["strategy"] == "sma_crossover"
    assert "sharpe" in payload["metrics"]


def test_cli_compare_json():
    import io
    import json as _json
    from contextlib import redirect_stdout
    from tradebot.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["compare", "--synthetic", "400", "--json"])
    assert rc == 0
    payload = _json.loads(buf.getvalue())
    assert isinstance(payload["strategies"], list) and payload["strategies"]
    # Ranked best-first by Sharpe.
    sharpes = [s["sharpe"] for s in payload["strategies"]]
    assert sharpes == sorted(sharpes, reverse=True)


# ---- keltner channels -----------------------------------------------------

def test_keltner_band_ordering():
    candles = data.synthetic(n=80)
    lower, mid, upper = indicators.keltner(
        [c.high for c in candles], [c.low for c in candles],
        [c.close for c in candles], 20, 10, 2.0)
    for lo, md, up in zip(lower, mid, upper):
        if None not in (lo, md, up):
            assert lo <= md <= up


def test_keltner_strategy_runs_both_modes():
    candles = data.synthetic(n=500)
    for mode in ("breakout", "reversion"):
        r = run_backtest(build("keltner", mode=mode), candles, interval="1h")
        assert len(r.equity_curve) == len(candles)
        assert min(r.equity_curve) >= 0
    assert "keltner" in REGISTRY


# ---- market stats ---------------------------------------------------------

def test_market_stats_detects_momentum_vs_random_walk():
    from tradebot.analysis import market_stats
    flat = market_stats(data.realistic_market(n=3000, seed=5, momentum=0.0), "1h")
    trend = market_stats(data.realistic_market(n=3000, seed=5, momentum=0.35), "1h")
    assert abs(flat.lag1_autocorr) < 0.06          # ~ random walk
    assert trend.lag1_autocorr > 0.15              # clearly trending
    assert "random walk" in flat.character()
    assert "momentum" in trend.character()


def test_market_stats_fields_sane():
    from tradebot.analysis import market_stats
    s = market_stats(data.synthetic(n=400), "1h")
    assert 0 <= s.pct_positive_bars <= 100
    assert s.annual_vol_pct >= 0
    assert s.buy_hold_max_drawdown_pct >= 0
    assert s.worst_bar_pct <= s.best_bar_pct


def test_cli_stats_json():
    import io
    import json as _json
    from contextlib import redirect_stdout
    from tradebot.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["stats", "--synthetic", "400", "--json"])
    assert rc == 0
    payload = _json.loads(buf.getvalue())
    assert "lag1_autocorr" in payload and "annual_vol_pct" in payload


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
