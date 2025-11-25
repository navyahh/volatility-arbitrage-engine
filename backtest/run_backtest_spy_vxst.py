# backtest/run_backtest_spy_vxst.py

from loader import make_spy_vxst_daily
from run_backtest import run_backtest


def main():
    df = make_spy_vxst_daily(start="2011-01-01", rv_window=21)

    print("Real SPY+VXST dataset (head):")
    print(df.head())
    print("rows:", len(df))

    stats = run_backtest(
        # REAL-DATA SETTINGS (more conservative)
        equity0=100_000.0,
        premium_risk_pct=0.002,   # 0.2% of equity per trade (MUCH smaller)
        mae_window=100,           # smoother band
        edge_mae_mult=1.5,        # require bigger edge to trade
        t=9 / 252,                # ~9-day horizon to match VXST
        save_csv=True,
        csv_path="backtest_results_spy_vxst.csv",
        do_plot=True,
        data_df=df,
    )

    print("\nPerformance (SPY+VXST real-data demo):")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
