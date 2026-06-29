#!/usr/bin/env bash
# tradebot demo — runs the whole workflow on offline synthetic data (no network,
# no API keys). From the trading-bot/ directory, run:  bash examples/demo.sh
set -e
cd "$(dirname "$0")/.."

echo "============================================================"
echo " 1. Compare every strategy"
echo "============================================================"
python -m tradebot compare --synthetic 800

echo
echo "============================================================"
echo " 2. Backtest the best one with risk management + a chart"
echo "============================================================"
python -m tradebot backtest -s supertrend --synthetic 800 \
    --stop-loss 0.05 --trailing-stop 0.05 --max-drawdown 0.25 --plot

echo
echo "============================================================"
echo " 3. Optimize parameters (with a heatmap)"
echo "============================================================"
python -m tradebot optimize -s sma --synthetic 800 --heatmap fast,slow --top 5

echo
echo "============================================================"
echo " 4. Walk-forward validation (catches overfitting)"
echo "============================================================"
python -m tradebot walkforward -s sma --synthetic 800 --folds 4

echo
echo "============================================================"
echo " 5. Monte Carlo robustness (how lucky was the backtest?)"
echo "============================================================"
python -m tradebot montecarlo -s sma --synthetic 1500 --sims 3000

echo
echo "============================================================"
echo " 6. Multi-coin portfolio"
echo "============================================================"
python -m tradebot portfolio --symbols BTC,ETH,SOL --synthetic 800 -s sma

echo
echo "Done. Swap --synthetic for --symbol BTCUSDT (live data) on your own machine."
