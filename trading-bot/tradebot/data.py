"""Market data: fetch live candles, load from CSV, or generate synthetic data.

Live fetching uses the public Binance REST API (no API key required) via the
standard library, so there are no third-party dependencies. The synthetic
generator lets the test suite and backtests run fully offline.
"""

from __future__ import annotations

import csv
import json
import math
import urllib.parse
import urllib.request
from typing import List

from .model import Candle

BINANCE_BASE = "https://api.binance.com"
VALID_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "1d"}


def fetch_klines(symbol: str = "BTCUSDT", interval: str = "1h",
                 limit: int = 500, timeout: float = 15.0) -> List[Candle]:
    """Fetch recent candles from Binance's public API.

    Args:
        symbol: trading pair, e.g. ``BTCUSDT``.
        interval: one of ``VALID_INTERVALS``.
        limit: number of candles (max 1000 per Binance).
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(f"interval must be one of {sorted(VALID_INTERVALS)}")
    query = urllib.parse.urlencode(
        {"symbol": symbol.upper(), "interval": interval, "limit": min(limit, 1000)}
    )
    url = f"{BINANCE_BASE}/api/v3/klines?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "tradebot/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        rows = json.loads(resp.read().decode())
    return [Candle.from_binance(r) for r in rows]


def load_csv(path: str) -> List[Candle]:
    """Load candles from a CSV with a header row.

    Expected columns (case-insensitive): timestamp, open, high, low, close,
    volume. Extra columns are ignored.
    """
    candles: List[Candle] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in cols]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        for row in reader:
            candles.append(
                Candle(
                    timestamp=int(float(row[cols["timestamp"]])),
                    open=float(row[cols["open"]]),
                    high=float(row[cols["high"]]),
                    low=float(row[cols["low"]]),
                    close=float(row[cols["close"]]),
                    volume=float(row[cols["volume"]]),
                )
            )
    return candles


def save_csv(candles: List[Candle], path: str) -> None:
    """Write candles to a CSV file (useful for caching fetched data)."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for c in candles:
            writer.writerow([c.timestamp, c.open, c.high, c.low, c.close, c.volume])


def synthetic(n: int = 500, start_price: float = 100.0, seed: int = 42,
              interval_ms: int = 3_600_000) -> List[Candle]:
    """Generate deterministic pseudo-random candles for offline testing.

    Uses a self-contained linear-congruential generator so results are
    reproducible without seeding the global ``random`` module. The series
    combines a gentle sine wave (so mean-reversion has something to revert to)
    with noise and mild drift.
    """
    state = seed & 0xFFFFFFFF

    def rand() -> float:  # uniform in [0, 1)
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF

    candles: List[Candle] = []
    price = start_price
    ts = 0
    for i in range(n):
        cycle = math.sin(i / 20.0) * 0.01          # slow oscillation
        drift = 0.0003                              # mild upward bias
        shock = (rand() - 0.5) * 0.03               # noise
        ret = cycle + drift + shock
        open_p = price
        close_p = max(0.01, price * (1 + ret))
        high_p = max(open_p, close_p) * (1 + rand() * 0.005)
        low_p = min(open_p, close_p) * (1 - rand() * 0.005)
        vol = 10 + rand() * 5
        candles.append(Candle(ts, open_p, high_p, low_p, close_p, vol))
        price = close_p
        ts += interval_ms
    return candles
