"""Market analysis — understand a series *before* you trade it.

Knowing a market's character saves you from bringing the wrong tool: positive
return autocorrelation favours trend-following, negative favours mean-reversion,
and fat tails (high kurtosis) warn that crash risk is bigger than a normal
model assumes. These are descriptive statistics of the data itself, independent
of any strategy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List

from .metrics import BARS_PER_YEAR
from .model import Candle


@dataclass
class MarketStats:
    candles: int
    buy_hold_return_pct: float
    cagr_pct: float
    annual_vol_pct: float
    mean_bar_return_pct: float
    bar_volatility_pct: float
    skew: float
    excess_kurtosis: float
    lag1_autocorr: float
    pct_positive_bars: float
    best_bar_pct: float
    worst_bar_pct: float
    buy_hold_max_drawdown_pct: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def character(self) -> str:
        """A one-line plain-English read of the market's personality."""
        if self.lag1_autocorr > 0.05:
            bias = "trending (momentum-friendly)"
        elif self.lag1_autocorr < -0.05:
            bias = "choppy (mean-reversion-friendly)"
        else:
            bias = "close to a random walk (hard to beat)"
        tails = "fat-tailed — crash/squeeze risk" if self.excess_kurtosis > 1 \
            else "roughly normal tails"
        return f"{bias}; {tails}"

    def summary(self) -> str:
        return "\n".join([
            f"  Candles               : {self.candles}",
            f"  Buy & hold return     : {self.buy_hold_return_pct:+.2f}%",
            f"  CAGR                  : {self.cagr_pct:+.2f}%",
            f"  Annualized volatility : {self.annual_vol_pct:.1f}%",
            f"  Mean / vol per bar    : {self.mean_bar_return_pct:+.3f}% / "
            f"{self.bar_volatility_pct:.3f}%",
            f"  Skew                  : {self.skew:+.2f}",
            f"  Excess kurtosis       : {self.excess_kurtosis:+.2f}  "
            f"(0 = normal; >1 = fat tails)",
            f"  Lag-1 autocorrelation : {self.lag1_autocorr:+.3f}  "
            f"(>0 trend, <0 reversion)",
            f"  Positive bars         : {self.pct_positive_bars:.1f}%",
            f"  Best / worst bar      : {self.best_bar_pct:+.2f}% / {self.worst_bar_pct:+.2f}%",
            f"  Buy & hold max DD     : {self.buy_hold_max_drawdown_pct:.2f}%",
            f"  Character             : {self.character()}",
        ])


def market_stats(candles: List[Candle], interval: str = "1h") -> MarketStats:
    """Compute descriptive statistics for a candle series."""
    if len(candles) < 3:
        raise ValueError("need at least 3 candles for statistics")

    closes = [c.close for c in candles]
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    n = len(rets)

    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / n
    std = math.sqrt(var)

    skew = kurt = 0.0
    if std > 0:
        skew = sum((r - mean) ** 3 for r in rets) / n / std ** 3
        kurt = sum((r - mean) ** 4 for r in rets) / n / std ** 4 - 3  # excess

    # Lag-1 autocorrelation.
    autocorr = 0.0
    denom = sum((r - mean) ** 2 for r in rets)
    if denom > 0:
        autocorr = sum((rets[i] - mean) * (rets[i - 1] - mean)
                       for i in range(1, n)) / denom

    periods = BARS_PER_YEAR.get(interval, 365)
    annual_vol = std * math.sqrt(periods)
    bnh = (closes[-1] / closes[0] - 1) * 100
    years = n / periods
    cagr = ((closes[-1] / closes[0]) ** (1 / years) - 1) * 100 if years > 0 and closes[0] > 0 else 0.0

    # Buy & hold max drawdown.
    peak = closes[0]
    worst_dd = 0.0
    for c in closes:
        peak = max(peak, c)
        if peak > 0:
            worst_dd = min(worst_dd, c / peak - 1)

    pos = sum(1 for r in rets if r > 0) / n * 100

    return MarketStats(
        candles=len(candles),
        buy_hold_return_pct=round(bnh, 2),
        cagr_pct=round(cagr, 2),
        annual_vol_pct=round(annual_vol * 100, 1),
        mean_bar_return_pct=round(mean * 100, 3),
        bar_volatility_pct=round(std * 100, 3),
        skew=round(skew, 2),
        excess_kurtosis=round(kurt, 2),
        lag1_autocorr=round(autocorr, 3),
        pct_positive_bars=round(pos, 1),
        best_bar_pct=round(max(rets) * 100, 2),
        worst_bar_pct=round(min(rets) * 100, 2),
        buy_hold_max_drawdown_pct=round(abs(worst_dd) * 100, 2),
    )
