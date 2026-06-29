# Changelog

All notable changes to tradebot are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Data workflow**: `download` command and `--cache DIR` for reproducible,
  offline backtests; candle resampling to higher timeframes.
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

### Safety
- Live trading is disabled by design; `LiveBroker` is a guarded stub.
