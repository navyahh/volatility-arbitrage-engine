A research framework for analyzing volatility mispricing, forecasting realized volatility, and backtesting delta-hedged option strategies on synthetic and real market data (SPY, VIX, VXST).

This project combines synthetic simulation, quantitative finance, and real-world volatility index data to study the variance risk premium (VRP) and build systematic volatility-trading signals.
Features
📈 1. Synthetic Volatility Engine

Simulates price paths

Generates realized volatility (RV)

Generates implied volatility (IV)

Computes RV–IV edge

Backtests delta-hedged straddles with transaction costs

🧪 2. Real-Market Data Modules

SPY + VIX

SPY + VXST (9-day IV)

Cleaned, aligned daily dataset

Realized volatility calculation

IV extraction from index levels

🧠 3. Forecasting & Research Tools

Rolling-mean RV forecast

Parameter sweeps (MAE window, thresholds, risk %)

Heatmaps / visualizations

Vol anomaly detection

PnL distributions for long-vol / short-vol trades

📊 4. Analytics

Equity curves

Sharpe ratio

Max drawdown

PnL per signal type

Edge vs PnL scatter

🧹 5. Clean Codebase

Organized modularly:

backtest/
    run_backtest.py
    run_backtest_spy_vix.py
    run_backtest_spy_vxst.py
    param_sweep.py
    param_viz.py
    anomaly_view.py
    loader.py
    equity_plot.py
    bs.py

Installation
git clone https://github.com/navyahh/volatility-arbitrage-engine
cd volatility-arbitrage-engine

python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt

Usage
Run synthetic backtest
python backtest/run_backtest.py

Run SPY + VIX backtest
python backtest/run_backtest_spy_vix.py

Run parameter sweep
python backtest/param_sweep.py

Plot parameter heatmaps
python backtest/param_viz.py

Analyze anomalies
python backtest/anomaly_view.py
