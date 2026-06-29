"""Market data: fetch live candles, load from CSV, or generate synthetic data.

Live fetching uses the public Binance REST API (no API key required) via the
standard library, so there are no third-party dependencies. The synthetic
generator lets the test suite and backtests run fully offline.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import urllib.parse
import urllib.request
from typing import List, Optional

from .model import Candle

BINANCE_BASE = "https://api.binance.com"
COINBASE_BASE = "https://api.exchange.coinbase.com"
KRAKEN_BASE = "https://api.kraken.com"
VALID_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "1d"}

# Interval -> granularity in seconds, shared by Coinbase and Kraken.
_INTERVAL_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600,
                     "4h": 14400, "1d": 86400}


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


def _get_json(url: str, timeout: float):
    req = urllib.request.Request(url, headers={"User-Agent": "tradebot/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_coinbase(symbol: str = "BTC-USD", interval: str = "1h",
                   limit: int = 300, timeout: float = 15.0) -> List[Candle]:
    """Fetch candles from Coinbase's public API (no key required).

    Coinbase symbols use a dash, e.g. ``BTC-USD``. It returns at most 300
    candles per request, newest-first; we normalize to oldest-first.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(f"interval must be one of {sorted(VALID_INTERVALS)}")
    granularity = _INTERVAL_SECONDS[interval]
    url = f"{COINBASE_BASE}/products/{symbol.upper()}/candles?granularity={granularity}"
    rows = _get_json(url, timeout)
    # Coinbase row: [time, low, high, open, close, volume]
    candles = [
        Candle(int(r[0]) * 1000, float(r[3]), float(r[2]), float(r[1]),
               float(r[4]), float(r[5]))
        for r in rows
    ]
    candles.sort(key=lambda c: c.timestamp)
    return candles[-min(limit, 300):]


def fetch_kraken(symbol: str = "XBTUSD", interval: str = "1h",
                 limit: int = 500, timeout: float = 15.0) -> List[Candle]:
    """Fetch candles from Kraken's public API (no key required).

    Kraken uses pairs like ``XBTUSD`` (note Bitcoin is ``XBT``). Returns up to
    720 candles; we keep the most recent ``limit``.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(f"interval must be one of {sorted(VALID_INTERVALS)}")
    minutes = _INTERVAL_SECONDS[interval] // 60
    url = f"{KRAKEN_BASE}/0/public/OHLC?pair={symbol.upper()}&interval={minutes}"
    payload = _get_json(url, timeout)
    if payload.get("error"):
        raise RuntimeError(f"Kraken error: {payload['error']}")
    result = payload["result"]
    key = next(k for k in result if k != "last")
    rows = result[key]
    # Kraken row: [time, open, high, low, close, vwap, volume, count]
    candles = [
        Candle(int(r[0]) * 1000, float(r[1]), float(r[2]), float(r[3]),
               float(r[4]), float(r[6]))
        for r in rows
    ]
    return candles[-limit:]


_SOURCES = {
    "binance": fetch_klines,
    "coinbase": fetch_coinbase,
    "kraken": fetch_kraken,
}


def fetch(source: str = "binance", **kwargs) -> List[Candle]:
    """Fetch candles from a named exchange: binance, coinbase, or kraken."""
    try:
        fn = _SOURCES[source.lower()]
    except KeyError:
        raise ValueError(f"unknown source {source!r}; "
                         f"choose from {sorted(_SOURCES)}")
    return fn(**kwargs)


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


def resample(candles: List[Candle], factor: int) -> List[Candle]:
    """Aggregate every ``factor`` candles into one higher-timeframe candle.

    E.g. ``resample(hourly, 4)`` builds 4-hour candles: open = first open,
    close = last close, high = max high, low = min low, volume = summed. A
    trailing partial group is dropped so every output bar is complete. This is
    how multi-timeframe analysis sees the "bigger picture" trend.
    """
    if factor <= 1:
        return list(candles)
    out: List[Candle] = []
    for i in range(0, len(candles) - factor + 1, factor):
        group = candles[i:i + factor]
        out.append(Candle(
            timestamp=group[0].timestamp,
            open=group[0].open,
            high=max(c.high for c in group),
            low=min(c.low for c in group),
            close=group[-1].close,
            volume=sum(c.volume for c in group),
        ))
    return out


def cache_path(cache_dir: str, source: str, symbol: str, interval: str) -> str:
    safe = symbol.replace("/", "-").upper()
    return os.path.join(cache_dir, f"{source}_{safe}_{interval}.csv")


def load_or_fetch(source: str = "binance", symbol: str = "BTCUSDT",
                  interval: str = "1h", limit: int = 500,
                  cache_dir: Optional[str] = None) -> List[Candle]:
    """Return cached candles if present, otherwise fetch and (if caching) save.

    With ``cache_dir`` set, the first run downloads and writes a CSV; later runs
    read that file, so backtests become reproducible and work offline.
    """
    if cache_dir:
        path = cache_path(cache_dir, source, symbol, interval)
        if os.path.exists(path):
            return load_csv(path)
        candles = fetch(source, symbol=symbol, interval=interval, limit=limit)
        os.makedirs(cache_dir, exist_ok=True)
        save_csv(candles, path)
        return candles
    return fetch(source, symbol=symbol, interval=interval, limit=limit)


def save_trades_csv(trades, path: str) -> None:
    """Write a list of executed Trades to CSV for inspection in a spreadsheet."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "side", "price", "quantity", "fee", "equity_after"])
        for t in trades:
            writer.writerow([t.timestamp, t.side.value, t.price, t.quantity,
                             t.fee, t.equity_after])


def realistic_market(n: int = 1500, start_price: float = 30_000.0, seed: int = 7,
                     annual_vol: float = 0.80, momentum: float = 0.0,
                     drift: float = 0.0, interval_ms: int = 3_600_000,
                     bars_per_year: int = 8_760) -> List[Candle]:
    """Generate a realistic synthetic market with fat tails and vol clustering.

    Unlike :func:`synthetic` (a smooth sine + noise, which is unrealistically
    easy to trade), this models log-returns with a GARCH(1,1) volatility process
    — calm and turbulent regimes cluster, just like real crypto — and optional
    return autocorrelation that acts as a *tunable, known edge*:

    * ``momentum = 0``   -> an efficient random walk with **no exploitable edge**.
      Any strategy that "beats" it in a backtest is being fooled by luck, and
      walk-forward / Monte Carlo should expose that.
    * ``momentum > 0``   -> returns persist (trends continue), a **real edge** a
      trend strategy can capture — and walk-forward should confirm it survives
      out of sample.

    This makes it a teaching instrument: you know the ground truth, so you can
    check whether the validation tools correctly tell edge from illusion.
    """
    import math
    rng = random.Random(seed)

    base_sigma = annual_vol / math.sqrt(bars_per_year)   # per-bar volatility
    alpha, beta = 0.10, 0.85                              # GARCH persistence ~0.95
    omega = base_sigma ** 2 * (1 - alpha - beta)
    sigma2 = base_sigma ** 2
    prev_r = 0.0

    candles: List[Candle] = []
    price = start_price
    ts = 0
    for _ in range(n):
        sigma2 = omega + alpha * (prev_r ** 2) + beta * sigma2
        sigma = math.sqrt(sigma2)
        z = rng.gauss(0.0, 1.0)
        r = drift + momentum * prev_r + sigma * z           # mean + shock
        prev_r = r

        open_p = price
        close_p = max(0.01, price * math.exp(r))
        # Intrabar range scaled by this bar's volatility.
        wick = sigma * (0.5 + rng.random())
        high_p = max(open_p, close_p) * math.exp(wick)
        low_p = min(open_p, close_p) * math.exp(-wick)
        vol = 100 * (0.5 + rng.random()) * (1 + 5 * abs(z))  # volume spikes on big moves
        candles.append(Candle(ts, open_p, high_p, low_p, close_p, vol))
        price = close_p
        ts += interval_ms
    return candles


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
