# Changelog

All notable changes to tradebot are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Data workflow**: `download` command and `--cache DIR` for reproducible,
  offline backtests; candle resampling to higher timeframes.
- **Realistic market simulator** (`data.realistic_market`): GARCH volatility
  clustering, fat tails, and a tunable known edge for practising edge detection.
- **Strategies** (11 total): SMA crossover, RSI reversion, MACD, Bollinger
  (reversion/breakout), Donchian breakout, SuperTrend, VWAP reversion,
  Stochastic, plus three meta-strategies — Ensemble (vote), RegimeAdaptive
  (ADX-routed), and MtfTrend (multi-timeframe filter).
- **Risk management**: stop-loss, take-profit, trailing stop, volatility-based
  position sizing, and a max-drawdown circuit breaker.
- **Short-selling**: stop-and-reverse backtesting (`--allow-short`).
- **Multi-coin portfolios**: equal-weight allocation with a combined equity curve.
- **Validation**: grid-search `optimize`, anchored `walkforward`, and Monte
  Carlo `montecarlo` robustness testing.
- **Analytics**: return, CAGR, max drawdown, Sharpe, Sortino, Calmar, profit
  factor, exposure, win rate.
- **Visualization**: ASCII equity charts, parameter heatmaps, and self-contained
  HTML reports with inline SVG.
- **Indicators**: SMA, EMA, RSI, MACD, Bollinger, ATR, Donchian, ADX, VWAP,
  SuperTrend, Stochastic.
- **Tooling**: pip-installable `tradebot` command, JSON config-file runner,
  trade-log CSV export, GitHub Actions CI on Python 3.8–3.12, 67 offline tests.

### Tooling (later additions)
- `--json` output on `backtest` and `compare` for piping into other tools.
- `stats` command + `analysis.market_stats()`: volatility, skew, excess
  kurtosis, lag-1 autocorrelation and a plain-English market character read.
- `keltner` strategy (ATR-band breakout/reversion) — 12 strategies total.

### Performance
- Backtester now passes each strategy only its required lookback window instead
  of the full history, turning the replay from O(n²) into O(n·window). A 5000-bar
  all-strategy comparison drops from a >120s timeout to ~30s; single-strategy
  backtests on 5000 bars are sub-second. Results are byte-for-byte identical to
  the full-history replay (verified by a regression test).

### Safety
- Live trading is disabled by design; `LiveBroker` is a guarded stub.
