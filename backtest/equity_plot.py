# backtest/equity_plot.py

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity(df: pd.DataFrame):
    if "equity" not in df.columns:
        raise KeyError(f"'equity' column not found. Got: {list(df.columns)}")

    plt.figure(figsize=(10, 4))
    plt.plot(df["equity"], linewidth=1.5)
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.show()
