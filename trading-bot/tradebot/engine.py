"""Live loop for paper or real trading.

Polls the exchange for the latest candles on a fixed interval, asks the strategy
for a signal on each newly closed bar, and routes orders to the chosen broker.
In paper mode the broker is a :class:`PaperBroker`, so you watch the strategy
trade real-time prices with fake money — the essential step between backtest and
risking real capital.
"""

from __future__ import annotations

import time
from typing import Optional

from .broker import Broker, PaperBroker
from .data import fetch_klines
from .model import Signal
from .strategy import Strategy

# Seconds per bar interval, used to size the polling sleep.
INTERVAL_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400,
}


class LiveEngine:
    def __init__(self, strategy: Strategy, broker: Optional[Broker] = None,
                 symbol: str = "BTCUSDT", interval: str = "1h",
                 position_fraction: float = 1.0):
        self.strategy = strategy
        self.broker = broker or PaperBroker()
        self.symbol = symbol
        self.interval = interval
        self.position_fraction = position_fraction
        self._last_bar_ts: Optional[int] = None

    def step(self) -> Signal:
        """Fetch latest candles, evaluate once per newly closed bar, act.

        Returns the signal acted on (or HOLD). The most recent kline from the
        API is the still-forming bar, so we drop it and act only on the last
        *closed* candle to avoid trading on incomplete data.
        """
        candles = fetch_klines(self.symbol, self.interval, limit=self.strategy.warmup() + 5)
        if len(candles) < 2:
            return Signal.HOLD
        closed = candles[:-1]              # drop the in-progress bar
        latest = closed[-1]

        if self._last_bar_ts == latest.timestamp:
            return Signal.HOLD            # already handled this bar
        self._last_bar_ts = latest.timestamp

        signal = self.strategy.evaluate(closed)
        price = latest.close

        if signal is Signal.BUY:
            self.broker.buy(latest.timestamp, price, self.position_fraction)
        elif signal is Signal.SELL:
            self.broker.sell(latest.timestamp, price, 1.0)
        return signal

    def run(self, poll_seconds: Optional[float] = None, max_steps: Optional[int] = None):
        """Run the polling loop. ``Ctrl-C`` to stop.

        ``max_steps`` bounds the loop (handy for tests/demos); ``None`` runs
        forever.
        """
        sleep_for = poll_seconds or INTERVAL_SECONDS.get(self.interval, 3600)
        steps = 0
        while max_steps is None or steps < max_steps:
            signal = self.step()
            price = fetch_last_price(self.symbol)
            equity = self.broker.equity(price) if price else float("nan")
            print(f"[{_now()}] {self.symbol} {self.interval} "
                  f"signal={signal.value} equity={equity:,.2f}")
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
            time.sleep(sleep_for)


def fetch_last_price(symbol: str) -> Optional[float]:
    try:
        candles = fetch_klines(symbol, "1m", limit=1)
        return candles[-1].close if candles else None
    except Exception:
        return None


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
