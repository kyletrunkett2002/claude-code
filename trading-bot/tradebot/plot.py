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


_SHADES = " .:-=+*#%@"


def heatmap(grid_results, x_param: str, y_param: str, metric: str) -> str:
    """Render a 2-D parameter heatmap from grid-search results.

    Each cell shows how a (x_param, y_param) combination scored on ``metric``,
    shaded from light (worst) to dark (best). This makes overfitting visible: a
    healthy strategy shows a broad bright *region* (many nearby settings work);
    a single bright cell surrounded by darkness is a fragile, cherry-picked
    fluke that probably won't survive live.
    """
    cells = {}
    xs, ys = set(), set()
    for g in grid_results:
        if x_param in g.params and y_param in g.params:
            x, y = g.params[x_param], g.params[y_param]
            xs.add(x); ys.add(y)
            # Keep the best score if combos repeat across other params.
            cells[(x, y)] = max(cells.get((x, y), float("-inf")), g.score)
    if not cells:
        return "(need a 2-parameter grid to draw a heatmap)"

    xs = sorted(xs)
    ys = sorted(ys)
    scores = [v for v in cells.values() if v != float("-inf")]
    lo, hi = min(scores), max(scores)
    span = hi - lo or 1.0

    def shade(val):
        idx = round((val - lo) / span * (len(_SHADES) - 1))
        return _SHADES[idx]

    xw = max(len(str(x)) for x in xs)
    lines = [f"  metric = {metric}   (shade: '{_SHADES[1]}'=worst {lo:.2f} "
             f"… '{_SHADES[-1]}'=best {hi:.2f})", ""]
    header = " " * (len(y_param) + 2) + " ".join(f"{x:>{xw}}" for x in xs)
    lines.append(f"  {header}    [{x_param}]")
    for y in ys:
        row_cells = []
        for x in xs:
            v = cells.get((x, y))
            row_cells.append(shade(v) * xw if v is not None and v != float("-inf")
                             else "?" * xw)
        lines.append(f"  {y:>{len(y_param)}} " + " ".join(row_cells))
    lines.append(f"  [{y_param}]")
    return "\n".join(lines)
