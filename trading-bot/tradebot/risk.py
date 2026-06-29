"""Risk management — the part that decides whether you survive.

Strategies decide *when* to trade; risk management decides *how much* and *when
to bail*. Most blown accounts come from skipping this: no stop-loss, betting the
whole stack on one idea, and refusing to stop after a losing streak. The defaults
here are conservative on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskConfig:
    """Risk rules applied on top of a strategy's signals.

    Attributes:
        stop_loss: exit if price falls this fraction below entry (0.05 = 5%).
            0 disables.
        take_profit: exit if price rises this fraction above entry. 0 disables.
        trailing_stop: a stop that ratchets in your favour. For a long it trails
            the highest price seen since entry by this fraction and only ever
            moves up; it locks in profit as the trade runs. 0 disables.
        max_drawdown: liquidate and halt new trades if account equity falls this
            fraction below its peak (the circuit breaker). 0 disables.
        risk_per_trade: if > 0, size each position so that being stopped out
            loses about this fraction of equity. Requires ``stop_loss`` > 0.
            This is volatility-aware position sizing: tighter stop -> bigger
            size, wider stop -> smaller size, constant dollar risk.
        position_fraction: fixed fraction of cash per entry when
            ``risk_per_trade`` is not used.
        max_position_fraction: hard cap on how much of cash a single entry may
            deploy, even after risk sizing.
    """

    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_stop: float = 0.0
    max_drawdown: float = 0.0
    risk_per_trade: float = 0.0
    position_fraction: float = 1.0
    max_position_fraction: float = 1.0

    def __post_init__(self):
        for name in ("stop_loss", "take_profit", "trailing_stop",
                     "max_drawdown", "risk_per_trade"):
            v = getattr(self, name)
            if not 0.0 <= v < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {v}")
        if not 0.0 < self.position_fraction <= 1.0:
            raise ValueError("position_fraction must be in (0, 1]")
        if not 0.0 < self.max_position_fraction <= 1.0:
            raise ValueError("max_position_fraction must be in (0, 1]")
        if self.risk_per_trade > 0 and self.stop_loss <= 0:
            raise ValueError("risk_per_trade requires a stop_loss to size against")

    def entry_fraction(self) -> float:
        """Fraction of cash to deploy on a new entry."""
        if self.risk_per_trade > 0 and self.stop_loss > 0:
            sized = self.risk_per_trade / self.stop_loss
            return min(sized, self.max_position_fraction)
        return min(self.position_fraction, self.max_position_fraction)


@dataclass
class _OpenPosition:
    entry_price: float
    side: int                       # +1 long, -1 short
    stop_price: Optional[float]
    target_price: Optional[float]
    extreme: float                  # best price seen since entry (high if long)


class RiskManager:
    """Stateful helper the backtester/engine consult each bar.

    Tracks the active position's stop/target levels and the account's peak
    equity for the drawdown circuit breaker.
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self._pos: Optional[_OpenPosition] = None
        self._peak_equity = 0.0
        self.halted = False

    # -- position lifecycle -------------------------------------------------

    def on_entry(self, entry_price: float, side: int = 1) -> None:
        """Register a new position. ``side`` is +1 for long, -1 for short.

        Stop and target levels invert for shorts: a short is hurt by price
        *rising*, so its stop sits above entry and its target below.
        """
        cfg = self.config
        stop = target = None
        if side >= 0:  # long
            if cfg.stop_loss > 0:
                stop = entry_price * (1 - cfg.stop_loss)
            if cfg.take_profit > 0:
                target = entry_price * (1 + cfg.take_profit)
        else:          # short
            if cfg.stop_loss > 0:
                stop = entry_price * (1 + cfg.stop_loss)
            if cfg.take_profit > 0:
                target = entry_price * (1 - cfg.take_profit)
        self._pos = _OpenPosition(entry_price, 1 if side >= 0 else -1, stop,
                                  target, extreme=entry_price)

    def on_exit(self) -> None:
        self._pos = None

    def update_trailing(self, bar_high: float, bar_low: float) -> None:
        """Ratchet the trailing stop in the position's favour for this bar.

        Called once per bar before :meth:`protective_exit`. The stop only ever
        tightens (up for a long, down for a short); it never loosens.
        """
        if self._pos is None or self.config.trailing_stop <= 0:
            return
        trail = self.config.trailing_stop
        if self._pos.side > 0:
            self._pos.extreme = max(self._pos.extreme, bar_high)
            new_stop = self._pos.extreme * (1 - trail)
            if self._pos.stop_price is None or new_stop > self._pos.stop_price:
                self._pos.stop_price = new_stop
        else:
            self._pos.extreme = min(self._pos.extreme, bar_low)
            new_stop = self._pos.extreme * (1 + trail)
            if self._pos.stop_price is None or new_stop < self._pos.stop_price:
                self._pos.stop_price = new_stop

    def protective_exit(self, bar_high: float, bar_low: float):
        """Check stop-loss / take-profit against a bar's range.

        Returns the fill price if a protective level was hit this bar, else
        None. If both could trigger in the same bar we assume the stop hit first
        (the conservative, pessimistic assumption).
        """
        if self._pos is None:
            return None
        stop = self._pos.stop_price
        target = self._pos.target_price
        if self._pos.side > 0:   # long: stop below, target above
            if stop is not None and bar_low <= stop:
                return stop
            if target is not None and bar_high >= target:
                return target
        else:                    # short: stop above, target below
            if stop is not None and bar_high >= stop:
                return stop
            if target is not None and bar_low <= target:
                return target
        return None

    # -- account-level circuit breaker -------------------------------------

    def update_equity(self, equity: float) -> bool:
        """Record equity; return True if the drawdown breaker just tripped."""
        self._peak_equity = max(self._peak_equity, equity)
        if self.config.max_drawdown <= 0 or self._peak_equity <= 0:
            return False
        drawdown = (self._peak_equity - equity) / self._peak_equity
        if drawdown >= self.config.max_drawdown and not self.halted:
            self.halted = True
            return True
        return False
