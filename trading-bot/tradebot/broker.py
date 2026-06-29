"""Brokers: where orders actually get filled.

``PaperBroker`` simulates fills with realistic fees and slippage and tracks one
long position in a single market. It powers both the backtester and live paper
trading, so what you test is exactly what you trade.

``LiveBroker`` is a deliberately guarded stub: real-money trading requires you
to plug in your own exchange keys and remove the safety guard yourself. That
friction is on purpose.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from .model import Signal, Trade


class Broker(ABC):
    @abstractmethod
    def buy(self, timestamp: int, price: float, fraction: float) -> Optional[Trade]:
        ...

    @abstractmethod
    def sell(self, timestamp: int, price: float, fraction: float) -> Optional[Trade]:
        ...

    @abstractmethod
    def equity(self, price: float) -> float:
        ...


class PaperBroker(Broker):
    """Simulated broker for a single long-only market.

    Args:
        cash: starting quote-currency balance (e.g. USDT).
        fee_rate: per-trade fee as a fraction (0.001 = 0.1%, Binance taker).
        slippage: adverse price move applied to each fill, as a fraction.
    """

    def __init__(self, cash: float = 10_000.0, fee_rate: float = 0.001, slippage: float = 0.0005):
        if cash <= 0:
            raise ValueError("starting cash must be positive")
        self.start_cash = cash
        self.cash = cash
        self.position = 0.0        # units of base asset held
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.trades: List[Trade] = []

    def equity(self, price: float) -> float:
        return self.cash + self.position * price

    def buy(self, timestamp: int, price: float, fraction: float = 1.0) -> Optional[Trade]:
        """Spend ``fraction`` of available cash on the base asset."""
        fraction = _clamp(fraction)
        spend = self.cash * fraction
        if spend <= 0:
            return None
        fill = price * (1 + self.slippage)
        fee = spend * self.fee_rate
        qty = (spend - fee) / fill
        if qty <= 0:
            return None
        self.cash -= spend
        self.position += qty
        trade = Trade(timestamp, Signal.BUY, fill, qty, fee, self.equity(price))
        self.trades.append(trade)
        return trade

    def sell(self, timestamp: int, price: float, fraction: float = 1.0) -> Optional[Trade]:
        """Sell ``fraction`` of the current long position (close, never short)."""
        fraction = _clamp(fraction)
        qty = self.position * fraction
        if qty <= 0:
            return None
        fill = price * (1 - self.slippage)
        proceeds = qty * fill
        fee = proceeds * self.fee_rate
        self.cash += proceeds - fee
        self.position -= qty
        trade = Trade(timestamp, Signal.SELL, fill, qty, fee, self.equity(price))
        self.trades.append(trade)
        return trade

    def sell_short(self, timestamp: int, price: float, fraction: float = 1.0) -> Optional[Trade]:
        """Open a short worth ``fraction`` of current equity (only when flat).

        Shorting borrows the asset and sells it: you receive cash now and owe
        the units back later. Equity stays ``cash + position * price`` with a
        negative position, so a price *rise* correctly shows as a loss.
        """
        if self.position != 0:
            return None
        fraction = _clamp(fraction)
        notional = self.equity(price) * fraction
        if notional <= 0:
            return None
        fill = price * (1 - self.slippage)   # selling: adverse fill is lower
        qty = notional / fill
        fee = notional * self.fee_rate
        self.cash += notional - fee
        self.position -= qty
        trade = Trade(timestamp, Signal.SELL, fill, qty, fee, self.equity(price))
        self.trades.append(trade)
        return trade

    def cover(self, timestamp: int, price: float) -> Optional[Trade]:
        """Buy back the entire short position."""
        if self.position >= 0:
            return None
        qty = -self.position
        fill = price * (1 + self.slippage)   # buying: adverse fill is higher
        cost = qty * fill
        fee = cost * self.fee_rate
        self.cash -= cost + fee
        self.position = 0.0
        trade = Trade(timestamp, Signal.BUY, fill, qty, fee, self.equity(price))
        self.trades.append(trade)
        return trade


class LiveBroker(Broker):
    """Real-money trading against an exchange.

    This is intentionally inert. To trade for real you must:
      1. Provide exchange API keys with trade permission.
      2. Implement signed order placement for your exchange.
      3. Remove the guard in ``_require_live_enabled`` once you accept the risk.

    Losing real money is easy; this friction is the point.
    """

    def __init__(self, api_key: str = "", api_secret: str = "", enabled: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.enabled = enabled

    def _require_live_enabled(self):
        if not self.enabled:
            raise RuntimeError(
                "LiveBroker is disabled. Live trading risks real money. "
                "Set enabled=True and implement signed order placement only "
                "after you have validated a strategy in backtest and paper modes."
            )

    def buy(self, timestamp: int, price: float, fraction: float):  # pragma: no cover
        self._require_live_enabled()
        raise NotImplementedError("Implement signed order placement for your exchange.")

    def sell(self, timestamp: int, price: float, fraction: float):  # pragma: no cover
        self._require_live_enabled()
        raise NotImplementedError("Implement signed order placement for your exchange.")

    def equity(self, price: float):  # pragma: no cover
        self._require_live_enabled()
        raise NotImplementedError("Query your exchange balance here.")


def _clamp(fraction: float) -> float:
    return max(0.0, min(1.0, fraction))
