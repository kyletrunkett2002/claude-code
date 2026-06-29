"""Command-line interface.

Run ``python -m tradebot --help`` to see commands:

    backtest    — replay history and print performance metrics (+ optional chart)
    optimize    — grid-search a strategy's parameters for the best metric
    walkforward — honest out-of-sample validation (catches overfitting)
    compare     — backtest every strategy and rank them
    paper       — trade live prices with fake money (safe)
    live        — trade real money (disabled until you wire in keys)
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from . import config as configmod
from . import data as datamod
from . import plot as plotmod
from .backtest import run_backtest
from .broker import LiveBroker, PaperBroker
from .engine import LiveEngine
from .model import Candle
from .optimize import DEFAULT_GRIDS, grid_search, walk_forward
from .portfolio import run_portfolio
from .risk import RiskConfig
from .strategies import REGISTRY, build


def _load_candles(args) -> List[Candle]:
    if getattr(args, "csv", None):
        return datamod.load_csv(args.csv)
    if getattr(args, "synthetic", None):
        return datamod.synthetic(n=args.synthetic)
    source = getattr(args, "source", "binance")
    return datamod.fetch(source, symbol=args.symbol, interval=args.interval,
                         limit=args.limit)


def _add_strategy_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--strategy", "-s", default="sma", choices=sorted(REGISTRY),
                   help="strategy to use (default: sma)")
    p.add_argument("--param", "-p", action="append", default=[], metavar="KEY=VALUE",
                   help="strategy parameter, e.g. -p fast=5 -p slow=20 (repeatable)")


def _build_strategy(args):
    return build(args.strategy, **_parse_params(args.param))


def _parse_params(items) -> dict:
    params = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"bad --param {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        params[key] = _coerce(value)
    return params


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _build_risk(args) -> RiskConfig:
    return RiskConfig(
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        max_drawdown=args.max_drawdown,
        risk_per_trade=args.risk_per_trade,
        position_fraction=args.fraction,
        max_position_fraction=args.max_position,
    )


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_backtest(args) -> int:
    strategy = _build_strategy(args)
    candles = _load_candles(args)
    if len(candles) <= strategy.warmup():
        print("Not enough candles for this strategy's warmup period.", file=sys.stderr)
        return 1
    result = run_backtest(
        strategy, candles,
        cash=args.cash, fee_rate=args.fee, slippage=args.slippage,
        interval=args.interval, allow_short=args.allow_short, risk=_build_risk(args),
    )
    print(f"Backtest: {strategy.name} on {len(candles)} candles "
          f"({args.symbol} {args.interval}{', long+short' if args.allow_short else ''})")
    print(result.summary())
    if args.plot:
        print("\nEquity curve:")
        print(plotmod.equity_chart(result.equity_curve))
    verdict = "BEATS" if result.metrics.total_return_pct > result.buy_and_hold_return_pct else "trails"
    print(f"\nStrategy {verdict} buy-and-hold. "
          "Past performance never guarantees future results.")
    return 0


def cmd_optimize(args) -> int:
    candles = _load_candles(args)
    grid = DEFAULT_GRIDS.get(args.strategy)
    if grid is None:
        print(f"No default grid for {args.strategy!r}. Optimizable: "
              f"{', '.join(sorted(DEFAULT_GRIDS))}", file=sys.stderr)
        return 1
    ranked = grid_search(args.strategy, candles, grid, metric=args.metric,
                         interval=args.interval, risk=_build_risk(args),
                         cash=args.cash, fee_rate=args.fee, slippage=args.slippage,
                         allow_short=args.allow_short)
    if not ranked:
        print("No valid parameter combinations.", file=sys.stderr)
        return 1
    print(f"Optimizing {args.strategy} by {args.metric} over {len(ranked)} "
          f"combinations ({args.symbol} {args.interval}):\n")
    print(f"  {'rank':<5}{'score':>9}  {'return%':>9}  {'maxDD%':>8}  params")
    for i, g in enumerate(ranked[:args.top], 1):
        m = g.result.metrics
        print(f"  {i:<5}{g.score:>9.2f}  {m.total_return_pct:>9.2f}  "
              f"{m.max_drawdown_pct:>8.2f}  {g.params}")
    if args.heatmap:
        try:
            x, y = (s.strip() for s in args.heatmap.split(","))
        except ValueError:
            print("--heatmap expects two parameter names, e.g. --heatmap fast,slow",
                  file=sys.stderr)
            return 1
        print("\n" + plotmod.heatmap(ranked, x, y, args.metric))
    print("\n⚠️  The top in-sample result is often overfit. Confirm it with "
          "`walkforward` before trusting it.")
    return 0


def cmd_walkforward(args) -> int:
    candles = _load_candles(args)
    grid = DEFAULT_GRIDS.get(args.strategy)
    if grid is None:
        print(f"No default grid for {args.strategy!r}.", file=sys.stderr)
        return 1
    try:
        wf = walk_forward(args.strategy, candles, grid, folds=args.folds,
                          train_ratio=args.train, metric=args.metric,
                          interval=args.interval, risk=_build_risk(args),
                          cash=args.cash, fee_rate=args.fee, slippage=args.slippage,
                          allow_short=args.allow_short)
    except ValueError as e:
        print(f"Walk-forward failed: {e}", file=sys.stderr)
        return 1
    print(f"Walk-forward validation: {args.strategy} ({args.symbol} {args.interval})\n")
    print(wf.summary())
    if args.plot and wf.oos_equity_curve:
        print("\nStitched out-of-sample equity curve:")
        print(plotmod.equity_chart(wf.oos_equity_curve))
    return 0


def cmd_compare(args) -> int:
    candles = _load_candles(args)
    risk = _build_risk(args)
    rows = []
    for name in sorted(REGISTRY):
        try:
            strat = build(name)
            if len(candles) <= strat.warmup():
                continue
            r = run_backtest(strat, candles, cash=args.cash, fee_rate=args.fee,
                             slippage=args.slippage, interval=args.interval,
                             allow_short=args.allow_short, risk=risk)
            m = r.metrics
            rows.append((name, m.total_return_pct, m.max_drawdown_pct,
                         m.sharpe, m.calmar, m.num_trades))
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: skipped ({e})", file=sys.stderr)
    rows.sort(key=lambda x: x[3], reverse=True)  # by Sharpe
    bnh = 0.0
    if candles:
        bnh = (candles[-1].close / candles[0].close - 1) * 100
    print(f"Strategy comparison ({args.symbol} {args.interval}, "
          f"{len(candles)} candles). Buy & hold: {bnh:+.2f}%\n")
    print(f"  {'strategy':<12}{'return%':>10}{'maxDD%':>9}{'Sharpe':>9}"
          f"{'Calmar':>9}{'trades':>8}")
    for name, ret, dd, sharpe, calmar, n in rows:
        print(f"  {name:<12}{ret:>10.2f}{dd:>9.2f}{_clip(sharpe):>9}"
              f"{_clip(calmar):>9}{n:>8}")
    return 0


def _clip(value: float) -> str:
    """Format a ratio, clamping absurd magnitudes so columns stay aligned.

    Very large ratios are usually an artifact of short/smooth (e.g. synthetic)
    data rather than a real edge, so we cap the displayed magnitude.
    """
    if value != value:  # NaN
        return "nan"
    if value == float("inf"):
        return "inf"
    if abs(value) >= 1000:
        return f"{'>' if value > 0 else '<'}999"
    return f"{value:.2f}"


def cmd_portfolio(args) -> int:
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("Provide --symbols sym1,sym2,...", file=sys.stderr)
        return 1
    if args.synthetic:
        # Distinct synthetic series per symbol via different seeds.
        symbol_candles = {s: datamod.synthetic(n=args.synthetic, seed=42 + i * 7)
                          for i, s in enumerate(symbols)}
    elif args.csv:
        print("Portfolio mode needs one series per symbol; use --synthetic or a "
              "live source, not a single --csv.", file=sys.stderr)
        return 1
    else:
        symbol_candles = {s: datamod.fetch(args.source, symbol=s,
                                           interval=args.interval, limit=args.limit)
                          for s in symbols}
    strat_params = _parse_params(args.param)
    result = run_portfolio(
        lambda: build(args.strategy, **strat_params), symbol_candles,
        cash=args.cash, fee_rate=args.fee, slippage=args.slippage,
        interval=args.interval, allow_short=args.allow_short, risk=_build_risk(args))
    print(f"Portfolio backtest: {args.strategy} across {len(symbols)} symbols")
    print(result.summary())
    if args.plot:
        print("\nPortfolio equity curve:")
        print(plotmod.equity_chart(result.equity_curve))
    return 0


def cmd_run(args) -> int:
    try:
        cfg = configmod.load(args.config)
    except (OSError, ValueError) as e:
        print(f"Could not load config: {e}", file=sys.stderr)
        return 1
    print(configmod.execute(cfg))
    return 0


def cmd_paper(args) -> int:
    strategy = _build_strategy(args)
    broker = PaperBroker(cash=args.cash, fee_rate=args.fee, slippage=args.slippage)
    engine = LiveEngine(strategy, broker, symbol=args.symbol,
                        interval=args.interval, position_fraction=args.fraction)
    print(f"Paper trading {args.symbol} {args.interval} with {strategy.name}. "
          "Fake money, real prices. Ctrl-C to stop.")
    try:
        engine.run(poll_seconds=args.poll, max_steps=args.steps)
    except KeyboardInterrupt:
        print("\nStopped.")
    print(f"Final: cash {broker.cash:,.2f} + {broker.position:.6f} units held. "
          f"Started with {broker.start_cash:,.2f} cash.")
    return 0


def cmd_live(args) -> int:
    print("Live trading is disabled by design.", file=sys.stderr)
    print("Validate your strategy with `backtest`, `optimize`, `walkforward` and "
          "`paper` first, then wire exchange keys into tradebot/broker.py:"
          "LiveBroker and enable it yourself.", file=sys.stderr)
    LiveBroker(enabled=False)
    return 2


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tradebot",
        description="Backtest, optimize, validate, paper-trade and (eventually) "
                    "live-trade crypto strategies.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def data_args(p):
        p.add_argument("--symbol", default="BTCUSDT", help="trading pair")
        p.add_argument("--source", default="binance",
                       choices=["binance", "coinbase", "kraken"],
                       help="exchange to fetch live data from")
        p.add_argument("--interval", default="1h",
                       choices=["1m", "5m", "15m", "1h", "4h", "1d"])
        p.add_argument("--limit", type=int, default=500, help="candles to fetch")
        p.add_argument("--csv", help="load candles from CSV instead of fetching")
        p.add_argument("--synthetic", type=int, metavar="N",
                       help="use N synthetic candles (offline, no network)")

    def account_args(p):
        p.add_argument("--cash", type=float, default=10_000.0)
        p.add_argument("--fee", type=float, default=0.001, help="fee rate (0.001=0.1%%)")
        p.add_argument("--slippage", type=float, default=0.0005)

    def risk_args(p):
        p.add_argument("--fraction", type=float, default=1.0,
                       help="fraction of cash per BUY (default 1.0)")
        p.add_argument("--max-position", type=float, default=1.0,
                       help="hard cap on cash deployed per trade")
        p.add_argument("--stop-loss", type=float, default=0.0,
                       help="exit if down this fraction from entry (0.05=5%%)")
        p.add_argument("--take-profit", type=float, default=0.0,
                       help="exit if up this fraction from entry")
        p.add_argument("--max-drawdown", type=float, default=0.0,
                       help="halt trading if equity falls this far from peak")
        p.add_argument("--risk-per-trade", type=float, default=0.0,
                       help="size positions to risk this fraction per trade "
                            "(needs --stop-loss)")
        p.add_argument("--allow-short", action="store_true",
                       help="stop-and-reverse: open shorts on SELL signals")

    # backtest
    bt = sub.add_parser("backtest", help="replay historical data")
    data_args(bt); account_args(bt); risk_args(bt); _add_strategy_args(bt)
    bt.add_argument("--plot", action="store_true", help="draw an ASCII equity curve")
    bt.set_defaults(func=cmd_backtest)

    # optimize
    op = sub.add_parser("optimize", help="grid-search strategy parameters")
    data_args(op); account_args(op); risk_args(op)
    op.add_argument("--strategy", "-s", default="sma", choices=sorted(DEFAULT_GRIDS))
    op.add_argument("--metric", default="sharpe",
                    help="metric to maximize (sharpe, calmar, total_return_pct, ...)")
    op.add_argument("--top", type=int, default=10, help="show top N results")
    op.add_argument("--heatmap", metavar="X,Y",
                    help="draw a 2-param heatmap, e.g. --heatmap fast,slow")
    op.set_defaults(func=cmd_optimize)

    # walkforward
    wf = sub.add_parser("walkforward", help="out-of-sample validation (anti-overfit)")
    data_args(wf); account_args(wf); risk_args(wf)
    wf.add_argument("--strategy", "-s", default="sma", choices=sorted(DEFAULT_GRIDS))
    wf.add_argument("--metric", default="sharpe")
    wf.add_argument("--folds", type=int, default=4)
    wf.add_argument("--train", type=float, default=0.7, help="train fraction per fold")
    wf.add_argument("--plot", action="store_true")
    wf.set_defaults(func=cmd_walkforward)

    # compare
    cp = sub.add_parser("compare", help="backtest every strategy and rank them")
    data_args(cp); account_args(cp); risk_args(cp)
    cp.set_defaults(func=cmd_compare)

    # portfolio
    po = sub.add_parser("portfolio", help="backtest one strategy across many coins")
    po.add_argument("--symbols", required=True,
                    help="comma-separated symbols, e.g. BTCUSDT,ETHUSDT,SOLUSDT")
    po.add_argument("--source", default="binance",
                    choices=["binance", "coinbase", "kraken"])
    po.add_argument("--interval", default="1h",
                    choices=["1m", "5m", "15m", "1h", "4h", "1d"])
    po.add_argument("--limit", type=int, default=500)
    po.add_argument("--synthetic", type=int, metavar="N",
                    help="use N synthetic candles per symbol (offline)")
    po.add_argument("--csv", help=argparse.SUPPRESS)
    account_args(po); risk_args(po); _add_strategy_args(po)
    po.add_argument("--plot", action="store_true")
    po.set_defaults(func=cmd_portfolio)

    # run (config file)
    rn = sub.add_parser("run", help="run an experiment described by a JSON config")
    rn.add_argument("config", help="path to a JSON config file")
    rn.set_defaults(func=cmd_run)

    # paper
    pp = sub.add_parser("paper", help="trade live prices with fake money")
    data_args(pp); account_args(pp); risk_args(pp); _add_strategy_args(pp)
    pp.add_argument("--poll", type=float, help="seconds between polls")
    pp.add_argument("--steps", type=int, help="stop after N polls (default: forever)")
    pp.set_defaults(func=cmd_paper)

    # live
    lv = sub.add_parser("live", help="trade real money (disabled)")
    data_args(lv); account_args(lv); risk_args(lv); _add_strategy_args(lv)
    lv.set_defaults(func=cmd_live)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
