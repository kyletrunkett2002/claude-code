"""Self-contained HTML reports.

Turns a backtest into a single ``.html`` file you can open in any browser or
email to someone — with an inline SVG equity curve (strategy vs buy-and-hold),
the full metrics table, and the trade log. No JavaScript, no external assets,
no dependencies; everything is embedded in the file.
"""

from __future__ import annotations

import html
from typing import List

from .backtest import BacktestResult
from .model import Candle


def _svg_equity(strategy_curve: List[float], hold_curve: List[float],
                width: int = 820, height: int = 320, pad: int = 40) -> str:
    """Render two equity curves as an inline SVG line chart."""
    all_vals = strategy_curve + hold_curve
    lo, hi = min(all_vals), max(all_vals)
    span = (hi - lo) or 1.0
    n = max(len(strategy_curve), 2)

    def points(curve: List[float]) -> str:
        m = len(curve)
        pts = []
        for i, v in enumerate(curve):
            x = pad + (width - 2 * pad) * (i / (m - 1 if m > 1 else 1))
            y = height - pad - (height - 2 * pad) * ((v - lo) / span)
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    # Horizontal gridlines with value labels.
    grid = []
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = height - pad - (height - 2 * pad) * f
        val = lo + span * f
        grid.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{width - pad}" '
                    f'y2="{y:.1f}" stroke="#eee"/>')
        grid.append(f'<text x="6" y="{y + 4:.1f}" font-size="11" '
                    f'fill="#888">{val:,.0f}</text>')

    return f'''<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg">
  {''.join(grid)}
  <polyline fill="none" stroke="#bbb" stroke-width="1.5"
            stroke-dasharray="4 3" points="{points(hold_curve)}"/>
  <polyline fill="none" stroke="#2563eb" stroke-width="2"
            points="{points(strategy_curve)}"/>
</svg>'''


def _metric_rows(result: BacktestResult) -> str:
    m = result.metrics
    rows = [
        ("Total return", f"{m.total_return_pct:+.2f}%"),
        ("Buy &amp; hold", f"{result.buy_and_hold_return_pct:+.2f}%"),
        ("CAGR", f"{m.cagr_pct:+.2f}%"),
        ("Max drawdown", f"{m.max_drawdown_pct:.2f}%"),
        ("Sharpe", f"{m.sharpe:.2f}"),
        ("Sortino", f"{m.sortino:.2f}"),
        ("Calmar", f"{m.calmar:.2f}"),
        ("Profit factor", f"{m.profit_factor:.2f}"),
        ("Exposure", f"{m.exposure_pct:.1f}%"),
        ("Trades", f"{m.num_trades}"),
        ("Win rate", f"{m.win_rate_pct:.2f}%"),
    ]
    return "".join(
        f"<tr><td>{k}</td><td class='num'>{v}</td></tr>" for k, v in rows
    )


def _trade_rows(result: BacktestResult, limit: int = 200) -> str:
    out = []
    for t in result.trades[:limit]:
        out.append(
            f"<tr><td>{t.timestamp}</td><td>{t.side.value}</td>"
            f"<td class='num'>{t.price:,.2f}</td>"
            f"<td class='num'>{t.quantity:.6f}</td>"
            f"<td class='num'>{t.fee:,.2f}</td>"
            f"<td class='num'>{t.equity_after:,.2f}</td></tr>"
        )
    if len(result.trades) > limit:
        out.append(f"<tr><td colspan='6'>… {len(result.trades) - limit} "
                   f"more orders omitted</td></tr>")
    return "".join(out)


def html_report(strategy_name: str, result: BacktestResult, candles: List[Candle],
                symbol: str = "", interval: str = "", path: str = "report.html") -> str:
    """Write an HTML report for a backtest and return the file path."""
    start_equity = result.equity_curve[0] if result.equity_curve else 0.0
    first_close = candles[0].close if candles else 1.0
    hold_curve = [start_equity * (c.close / first_close) for c in candles]

    title = html.escape(f"{strategy_name} — {symbol} {interval}".strip())
    beats = result.metrics.total_return_pct > result.buy_and_hold_return_pct
    verdict = ("beat" if beats else "trailed")
    chart = _svg_equity(result.equity_curve, hold_curve)

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>tradebot report — {title}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2rem auto;
         max-width: 880px; color: #1a1a1a; padding: 0 1rem; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0; }}
  .sub {{ color: #666; margin-top: .2rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .9rem; }}
  td, th {{ padding: .35rem .6rem; border-bottom: 1px solid #eee; text-align: left; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .legend span {{ display: inline-block; margin-right: 1.2rem; font-size: .85rem; }}
  .blue {{ color: #2563eb; }} .grey {{ color: #999; }}
  .note {{ color: #666; font-size: .8rem; margin-top: 2rem; border-top: 1px solid #eee;
          padding-top: 1rem; }}
  details summary {{ cursor: pointer; font-weight: 600; margin: 1rem 0 .5rem; }}
</style></head><body>
<h1>{title}</h1>
<p class="sub">Strategy <b>{verdict}</b> buy-and-hold over {len(candles)} candles.</p>
<div class="legend">
  <span class="blue">■ strategy equity</span>
  <span class="grey">▱ buy &amp; hold</span>
</div>
{chart}
<h2 style="font-size:1.1rem">Performance</h2>
<table><tbody>{_metric_rows(result)}</tbody></table>
<details><summary>Trade log ({result.metrics.num_trades} round-trips, {len(result.trades)} orders)</summary>
<table>
<thead><tr><th>timestamp</th><th>side</th><th class="num">price</th>
<th class="num">qty</th><th class="num">fee</th><th class="num">equity</th></tr></thead>
<tbody>{_trade_rows(result)}</tbody></table></details>
<p class="note">Generated by tradebot. For education and research only — not
financial advice. Past performance never guarantees future results.</p>
</body></html>"""

    with open(path, "w") as fh:
        fh.write(doc)
    return path
