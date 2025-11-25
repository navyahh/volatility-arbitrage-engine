# backtest/anomaly_view.py

from pathlib import Path
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MAE_WINDOW = 60
EDGE_MUL = 1.25


def main():
    print("=== anomaly_view.py STARTED ===")

    root = Path(__file__).resolve().parents[1]
    csv_path = root / "backtest_results.csv"
    print(f"Looking for CSV at: {csv_path}")

    if not csv_path.exists():
        print("ERROR: backtest_results.csv not found.")
        return

    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    except Exception as e:
        print("ERROR: failed to read CSV:", e)
        traceback.print_exc()
        return

    print("CSV loaded. Rows:", len(df))
    print("Columns:", list(df.columns))

    # Rebuild MAE band so we know when edge is 'large'
    try:
        err = (df["rv_next"] - df["forecast"]).abs()
        mae = err.rolling(MAE_WINDOW).mean()
        band = EDGE_MUL * mae

        df["abs_edge"] = df["edge"].abs()
        df["band"] = band
        df["is_extreme"] = df["abs_edge"] >= df["band"]
    except Exception as e:
        print("ERROR: failed computing edge stats:", e)
        traceback.print_exc()
        return

    print("\n=== Basic counts ===")
    print("rows:", len(df))
    print("long-vol days (signal=+1):", int((df["signal"] == 1).sum()))
    print("short-vol days (signal=-1):", int((df["signal"] == -1).sum()))
    print("flat days (signal=0):", int((df["signal"] == 0).sum()))
    print("extreme edge days (|edge|>=band):", int(df["is_extreme"].sum()))

    print("\n=== Top 10 sell-vol days (edge most positive) ===")
    try:
        top_short = df.sort_values("edge", ascending=False).head(10)
        print(top_short[["date", "edge", "signal", "pnl", "equity"]].to_string(index=False))
    except Exception as e:
        print("ERROR: failed printing top_short:", e)

    print("\n=== Top 10 buy-vol days (edge most negative) ===")
    try:
        top_long = df.sort_values("edge", ascending=True).head(10)
        print(top_long[["date", "edge", "signal", "pnl", "equity"]].to_string(index=False))
    except Exception as e:
        print("ERROR: failed printing top_long:", e)

    print("\n=== PnL by side (mean per trade day) ===")
    for side, name in [(-1, "short vol"), (1, "long vol")]:
        m = df["signal"] == side
        if m.any():
            print(
                f"{name:10s} | n={int(m.sum()):4d} | "
                f"mean pnl={df.loc[m, 'pnl'].mean():8.2f} | "
                f"median pnl={df.loc[m, 'pnl'].median():8.2f}"
            )

    m_trade = df["signal"] != 0
    if m_trade.any():
        print("\nPlotting edge vs PnL scatter...")
        plt.figure(figsize=(8, 4))
        plt.scatter(df.loc[m_trade, "edge"], df.loc[m_trade, "pnl"], s=10, alpha=0.6)
        plt.axvline(0.0, linewidth=1)
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("IV - forecast RV (edge)")
        plt.ylabel("Trade PnL")
        plt.title("Edge vs PnL (trade days only)")
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo trade days found (signal == 0 everywhere).")


if __name__ == "__main__":
    main()
