# backtest/run_backtest_spy_vix.py

from loader import make_spy_vix_daily
from run_backtest import run_backtest


def main():
    # Build real SPY + VIX dataset (30-day IV, RV from SPY)
    df = make_spy_vix_daily(start="2010-01-01", rv_window=21)

    print("Real SPY+VIX dataset:")
    print(df.head())
    print("rows:", len(df))

    # --- smoothing to avoid insane noise ---
    # smooth realized vol
    df["rv_smooth"] = df["rv_next"].rolling(21).mean().bfill()
    df["forecast"] = df["rv_smooth"]

    # smooth implied vol a bit
    df["iv_atm"] = df["iv_atm"].rolling(5).mean().bfill()

    # Run backtest with conservative parameters
    stats = run_backtest(
        equity0=100_000.0,
        premium_risk_pct=0.0015,        # 0.15% of equity per trade
        mae_window=120,                 # longer MAE → wider band
        edge_mae_mult=1.75,             # need bigger edge to trade
        t=30 / 252,                     # ~30-day horizon to match VIX
        save_csv=True,
        csv_path="backtest_results_spy_vix.csv",
        do_plot=True,
        data_df=df,
        min_trade_gap=5,                # at least 5 days between trades
    )

    print("\nPerformance (SPY+VIX real-data demo):")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
