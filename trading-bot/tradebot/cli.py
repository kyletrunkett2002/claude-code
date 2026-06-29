"""Command-line interface.

Run ``python -m tradebot --help`` to see commands. The three you care about:

    backtest  — replay history and print performance metrics
    paper     — trade live prices with fake money (safe)
    live      — trade real money (disabled until you wire in keys)
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from . import data as datamod
from .backtest import run_backtest
from .broker import LiveBroker, PaperBroker
from .engine import LiveEngine
from .model import Candle
from .strategies import REGISTRY, build


def _load_candles(args) -> List[Candle]:
    if args.csv:
        return datamod.load_csv(args.csv)
    if args.synthetic:
        return datamod.synthetic(n=args.synthetic)
    return datamod.fetch_klines(args.symbol, args.interval, limit=args.limit)


def _add_strategy_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--strategy", "-s", default="sma", choices=sorted(REGISTRY),
                   help="strategy to use (default: sma)")
    p.add_argument("--param", "-p", action="append", default=[], metavar="KEY=VALUE",
                   help="strategy parameter, e.g. -p fast=5 -p slow=20 (repeatable)")


def _build_strategy(args):
    params = {}
    for item in args.param:
        if "=" not in item:
            raise SystemExit(f"bad --param {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        params[key] = _coerce(value)
    return build(args.strategy, **params)


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def cmd_backtest(args) -> int:
    strategy = _build_strategy(args)
    candles = _load_candles(args)
    if len(candles) <= strategy.warmup():
        print("Not enough candles for this strategy's warmup period.", file=sys.stderr)
        return 1
    result = run_backtest(
        strategy, candles,
        cash=args.cash, fee_rate=args.fee, slippage=args.slippage,
        position_fraction=args.fraction, interval=args.interval,
    )
    print(f"Backtest: {strategy.name} on {len(candles)} candles "
          f"({args.symbol} {args.interval})")
    print(result.summary())
    verdict = "BEATS" if result.metrics.total_return_pct > result.buy_and_hold_return_pct else "trails"
    print(f"\nStrategy {verdict} buy-and-hold. "
          "Remember: past performance never guarantees future results.")
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
    print("Validate your strategy with `backtest` and `paper` first, then wire "
          "exchange keys into tradebot/broker.py:LiveBroker and enable it "
          "yourself.", file=sys.stderr)
    LiveBroker(enabled=False)  # exists to make the guard discoverable
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tradebot",
        description="Backtest, paper-trade, and (eventually) live-trade crypto strategies.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--symbol", default="BTCUSDT", help="trading pair")
        p.add_argument("--interval", default="1h",
                       choices=["1m", "5m", "15m", "1h", "4h", "1d"])
        p.add_argument("--cash", type=float, default=10_000.0)
        p.add_argument("--fee", type=float, default=0.001, help="fee rate (0.001=0.1%%)")
        p.add_argument("--slippage", type=float, default=0.0005)
        p.add_argument("--fraction", type=float, default=1.0,
                       help="fraction of cash to deploy per BUY")
        _add_strategy_args(p)

    bt = sub.add_parser("backtest", help="replay historical data")
    common(bt)
    bt.add_argument("--limit", type=int, default=500, help="candles to fetch")
    bt.add_argument("--csv", help="load candles from CSV instead of fetching")
    bt.add_argument("--synthetic", type=int, metavar="N",
                    help="use N synthetic candles (offline, no network)")
    bt.set_defaults(func=cmd_backtest)

    pp = sub.add_parser("paper", help="trade live prices with fake money")
    common(pp)
    pp.add_argument("--poll", type=float, help="seconds between polls")
    pp.add_argument("--steps", type=int, help="stop after N polls (default: forever)")
    pp.set_defaults(func=cmd_paper)

    lv = sub.add_parser("live", help="trade real money (disabled)")
    common(lv)
    lv.set_defaults(func=cmd_live)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
