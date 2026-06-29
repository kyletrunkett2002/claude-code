"""Tiny ASCII charts so you can *see* an equity curve in the terminal.

No matplotlib, no dependencies — just enough to eyeball whether a curve grinds
smoothly up or rockets and crashes.
"""

from __future__ import annotations

from typing import List

_BLOCKS = " ▁▂▃▄▅▆▇█"


def sparkline(values: List[float]) -> str:
    """One-line sparkline of a series using Unicode block characters."""
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return _BLOCKS[0] * len(values)
    out = []
    for v in values:
        idx = round((v - lo) / (hi - lo) * (len(_BLOCKS) - 1))
        out.append(_BLOCKS[idx])
    return "".join(out)


def equity_chart(curve: List[float], width: int = 70, height: int = 14) -> str:
    """A multi-line ASCII chart of an equity curve, downsampled to ``width``."""
    if len(curve) < 2:
        return "(not enough data to chart)"

    # Downsample to at most `width` columns by averaging buckets.
    step = max(1, len(curve) // width)
    cols = [sum(curve[i:i + step]) / len(curve[i:i + step])
            for i in range(0, len(curve), step)][:width]

    lo = min(cols)
    hi = max(cols)
    span = hi - lo or 1.0
    rows = []
    for r in range(height, 0, -1):
        threshold = lo + span * (r - 0.5) / height
        line = "".join("█" if c >= threshold else " " for c in cols)
        if r == height:
            label = f"{hi:>10,.0f} "
        elif r == 1:
            label = f"{lo:>10,.0f} "
        else:
            label = " " * 11
        rows.append(label + "│" + line)
    axis = " " * 11 + "└" + "─" * len(cols)
    pct = (curve[-1] / curve[0] - 1) * 100 if curve[0] else 0.0
    footer = " " * 12 + f"start {curve[0]:,.0f}  →  end {curve[-1]:,.0f}  ({pct:+.1f}%)"
    return "\n".join(rows + [axis, footer])
