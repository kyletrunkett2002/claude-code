"""Pure-Python technical indicators.

These operate on plain lists of floats so the framework needs no numpy/pandas.
Each function returns a list the same length as the input; positions that do not
yet have enough data to compute are ``None``.
"""

from __future__ import annotations

import math
from typing import List, Optional


def sma(values: List[float], period: int) -> List[Optional[float]]:
    """Simple moving average."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: List[Optional[float]] = [None] * len(values)
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= period:
            running -= values[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


def ema(values: List[float], period: int) -> List[Optional[float]]:
    """Exponential moving average, seeded with the first SMA."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: List[Optional[float]] = [None] * len(values)
    if len(values) < period:
        return out
    k = 2 / (period + 1)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def rsi(values: List[float], period: int = 14) -> List[Optional[float]]:
    """Wilder's Relative Strength Index."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: List[Optional[float]] = [None] * len(values)
    if len(values) <= period:
        return out

    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        change = values[i] - values[i - 1]
        gains += max(change, 0.0)
        losses += max(-change, 0.0)
    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = _rsi_from(avg_gain, avg_loss)

    for i in range(period + 1, len(values)):
        change = values[i] - values[i - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = _rsi_from(avg_gain, avg_loss)
    return out


def _rsi_from(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def stddev(values: List[float], period: int) -> List[Optional[float]]:
    """Rolling sample standard deviation."""
    if period <= 1:
        raise ValueError("period must be > 1")
    out: List[Optional[float]] = [None] * len(values)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1 : i + 1]
        mean = sum(window) / period
        var = sum((x - mean) ** 2 for x in window) / (period - 1)
        out[i] = math.sqrt(var)
    return out


def bollinger(values: List[float], period: int = 20, num_std: float = 2.0):
    """Bollinger Bands: returns (lower, middle, upper) lists.

    The middle band is an SMA; the outer bands sit ``num_std`` standard
    deviations away. Price riding the bands signals stretch.
    """
    mid = sma(values, period)
    sd = stddev(values, period)
    lower: List[Optional[float]] = [None] * len(values)
    upper: List[Optional[float]] = [None] * len(values)
    for i in range(len(values)):
        if mid[i] is not None and sd[i] is not None:
            lower[i] = mid[i] - num_std * sd[i]
            upper[i] = mid[i] + num_std * sd[i]
    return lower, mid, upper


def macd(values: List[float], fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD: returns (macd_line, signal_line, histogram).

    ``macd_line = EMA(fast) - EMA(slow)``; the signal line is an EMA of that.
    The histogram (macd - signal) crossing zero is the classic trade trigger.
    """
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    macd_line: List[Optional[float]] = [None] * len(values)
    for i in range(len(values)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line[i] = ema_fast[i] - ema_slow[i]

    # Signal line = EMA of the macd_line over its non-None tail.
    start = next((i for i, v in enumerate(macd_line) if v is not None), None)
    signal_line: List[Optional[float]] = [None] * len(values)
    if start is not None:
        dense = [v for v in macd_line[start:]]
        sig = ema(dense, signal)
        for offset, v in enumerate(sig):
            signal_line[start + offset] = v

    hist: List[Optional[float]] = [None] * len(values)
    for i in range(len(values)):
        if macd_line[i] is not None and signal_line[i] is not None:
            hist[i] = macd_line[i] - signal_line[i]
    return macd_line, signal_line, hist


def atr(highs: List[float], lows: List[float], closes: List[float],
        period: int = 14) -> List[Optional[float]]:
    """Average True Range — Wilder's volatility measure.

    Used for volatility-based position sizing and stop placement.
    """
    n = len(closes)
    if not (len(highs) == len(lows) == n):
        raise ValueError("highs, lows, closes must be equal length")
    tr: List[float] = [0.0] * n
    for i in range(n):
        if i == 0:
            tr[i] = highs[i] - lows[i]
        else:
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
    out: List[Optional[float]] = [None] * n
    if n < period:
        return out
    prev = sum(tr[:period]) / period
    out[period - 1] = prev
    for i in range(period, n):
        prev = (prev * (period - 1) + tr[i]) / period
        out[i] = prev
    return out


def donchian(highs: List[float], lows: List[float], period: int = 20):
    """Donchian channel: rolling (lowest_low, highest_high) over ``period``.

    A breakout above the prior highest-high is the classic momentum entry.
    """
    n = len(highs)
    hi: List[Optional[float]] = [None] * n
    lo: List[Optional[float]] = [None] * n
    for i in range(period - 1, n):
        hi[i] = max(highs[i - period + 1 : i + 1])
        lo[i] = min(lows[i - period + 1 : i + 1])
    return lo, hi
