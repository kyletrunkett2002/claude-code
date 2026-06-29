"""Config-file driven runner.

Describe a whole experiment — data source, strategy, parameters, risk rules,
mode — in one JSON file and replay it with ``tradebot run config.json``. This
keeps your validated setups version-controlled and reproducible instead of
living in shell-history one-liners.

JSON is used (not YAML) so there are still zero dependencies.
"""

from __future__ import annotations

import json
from typing import Dict, List

from . import data as datamod
from .backtest import run_backtest
from .metrics import compute  # noqa: F401  (re-exported convenience)
from .model import Candle
from .optimize import DEFAULT_GRIDS, grid_search, walk_forward
from .portfolio import run_portfolio
from .risk import RiskConfig
from .strategies import REGISTRY, build

DEFAULTS = {
    "mode": "backtest",        # backtest | portfolio | optimize | walkforward | compare
    "source": "synthetic",     # synthetic | csv | binance | coinbase | kraken
    "symbol": "BTCUSDT",
    "symbols": None,           # list, for portfolio mode
    "interval": "1h",
    "limit": 500,
    "synthetic": 800,          # candle count when source == synthetic
    "csv": None,
    "strategy": "sma",
    "params": {},
    "cash": 10_000.0,
    "fee": 0.001,
    "slippage": 0.0005,
    "allow_short": False,
    "risk": {},
    "metric": "sharpe",
    "folds": 4,
    "plot": False,
}


def load(path: str) -> Dict:
    """Load a config file, filling in defaults for omitted keys."""
    with open(path) as fh:
        raw = json.load(fh)
    cfg = dict(DEFAULTS)
    cfg.update(raw)
    if cfg["mode"] not in ("backtest", "portfolio", "optimize", "walkforward", "compare"):
        raise ValueError(f"unknown mode {cfg['mode']!r}")
    return cfg


def _risk(cfg: Dict) -> RiskConfig:
    return RiskConfig(**cfg.get("risk", {})) if cfg.get("risk") else RiskConfig()


def _stable_seed(symbol: str) -> int:
    """Deterministic per-symbol seed (Python's hash() is randomized per run)."""
    h = 0
    for ch in symbol:
        h = (h * 131 + ord(ch)) & 0xFFFFFF
    return 42 + h % 1000


def _candles_for(cfg: Dict, symbol: str) -> List[Candle]:
    src = cfg["source"]
    if src == "synthetic":
        # Vary the seed by symbol so a portfolio gets distinct but reproducible series.
        return datamod.synthetic(n=cfg["synthetic"], seed=_stable_seed(symbol))
    if src == "csv":
        return datamod.load_csv(cfg["csv"])
    return datamod.fetch(src, symbol=symbol, interval=cfg["interval"],
                         limit=cfg["limit"])


def execute(cfg: Dict) -> str:
    """Run the experiment described by ``cfg`` and return a printable report."""
    from . import plot as plotmod

    mode = cfg["mode"]
    risk = _risk(cfg)

    if mode == "portfolio":
        symbols = cfg["symbols"] or [cfg["symbol"]]
        symbol_candles = {s: _candles_for(cfg, s) for s in symbols}
        result = run_portfolio(
            lambda: build(cfg["strategy"], **cfg["params"]),
            symbol_candles, cash=cfg["cash"], fee_rate=cfg["fee"],
            slippage=cfg["slippage"], interval=cfg["interval"],
            allow_short=cfg["allow_short"], risk=risk)
        out = [f"Portfolio [{cfg['strategy']}]:", result.summary()]
        if cfg["plot"]:
            out += ["", "Portfolio equity curve:", plotmod.equity_chart(result.equity_curve)]
        return "\n".join(out)

    candles = _candles_for(cfg, cfg["symbol"])

    if mode == "backtest":
        result = run_backtest(build(cfg["strategy"], **cfg["params"]), candles,
                              cash=cfg["cash"], fee_rate=cfg["fee"],
                              slippage=cfg["slippage"], interval=cfg["interval"],
                              allow_short=cfg["allow_short"], risk=risk)
        out = [f"Backtest [{cfg['strategy']}] on {len(candles)} candles:",
               result.summary()]
        if cfg["plot"]:
            out += ["", "Equity curve:", plotmod.equity_chart(result.equity_curve)]
        return "\n".join(out)

    if mode == "optimize":
        grid = DEFAULT_GRIDS.get(cfg["strategy"], {})
        ranked = grid_search(cfg["strategy"], candles, grid, metric=cfg["metric"],
                             interval=cfg["interval"], risk=risk, cash=cfg["cash"],
                             fee_rate=cfg["fee"], slippage=cfg["slippage"],
                             allow_short=cfg["allow_short"])
        out = [f"Optimize [{cfg['strategy']}] by {cfg['metric']}:"]
        for i, g in enumerate(ranked[:10], 1):
            out.append(f"  {i:<3}{g.score:>9.2f}  {g.params}")
        return "\n".join(out)

    if mode == "walkforward":
        grid = DEFAULT_GRIDS.get(cfg["strategy"], {})
        wf = walk_forward(cfg["strategy"], candles, grid, folds=cfg["folds"],
                          metric=cfg["metric"], interval=cfg["interval"], risk=risk,
                          cash=cfg["cash"], fee_rate=cfg["fee"],
                          slippage=cfg["slippage"], allow_short=cfg["allow_short"])
        return f"Walk-forward [{cfg['strategy']}]:\n{wf.summary()}"

    # compare
    out = [f"Compare all strategies ({cfg['symbol']} {cfg['interval']}):"]
    rows = []
    for name in sorted(REGISTRY):
        strat = build(name)
        if len(candles) <= strat.warmup():
            continue
        r = run_backtest(strat, candles, cash=cfg["cash"], fee_rate=cfg["fee"],
                         slippage=cfg["slippage"], interval=cfg["interval"],
                         allow_short=cfg["allow_short"], risk=risk)
        rows.append((name, r.metrics.total_return_pct, r.metrics.sharpe))
    rows.sort(key=lambda x: x[2], reverse=True)
    for name, ret, sharpe in rows:
        out.append(f"  {name:<12}{ret:>9.2f}%  Sharpe {sharpe:.2f}")
    return "\n".join(out)
