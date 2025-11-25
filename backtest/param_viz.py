# backtest/param_viz.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_results(path: str = "param_results.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {
        "mae_window",
        "edge_mae_mult",
        "premium_risk_pct",
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "n_trades",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("=== BASIC SUMMARY ===")
    print("rows:", len(df))
    print("\nTop 10 configs by Sharpe:")
    top = df.sort_values("sharpe", ascending=False).head(10)
    print(
        top[
            [
                "mae_window",
                "edge_mae_mult",
                "premium_risk_pct",
                "sharpe",
                "ann_return",
                "max_drawdown",
                "n_trades",
            ]
        ].to_string(index=False)
    )

    print("\nSharpe stats:")
    print(df["sharpe"].describe())


def heatmap_for_premium(df: pd.DataFrame, prem: float) -> None:
    sub = df[df["premium_risk_pct"].round(6) == round(prem, 6)]
    if sub.empty:
        print(f"[heatmap] no rows for premium={prem}")
        return

    pivot = sub.pivot(
        index="mae_window",
        columns="edge_mae_mult",
        values="sharpe",
    )

    mae_vals = pivot.index.values
    edge_vals = pivot.columns.values

    plt.figure()
    plt.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
    )
    plt.xticks(
        ticks=np.arange(len(edge_vals)),
        labels=[str(e) for e in edge_vals],
    )
    plt.yticks(
        ticks=np.arange(len(mae_vals)),
        labels=[str(m) for m in mae_vals],
    )
    plt.colorbar(label="Sharpe")
    plt.title(f"Sharpe heatmap (premium_risk_pct={prem})")
    plt.xlabel("edge_mae_mult")
    plt.ylabel("mae_window")
    plt.tight_layout()
    plt.show()


def scatter_sharpe_vs_drawdown(df: pd.DataFrame) -> None:
    plt.figure()
    plt.scatter(df["max_drawdown"], df["sharpe"], s=25, alpha=0.7)
    plt.xlabel("max_drawdown")
    plt.ylabel("sharpe")
    plt.title("Sharpe vs Max Drawdown")
    plt.tight_layout()
    plt.show()


def scatter_sharpe_vs_trades(df: pd.DataFrame) -> None:
    plt.figure()
    plt.scatter(df["n_trades"], df["sharpe"], s=25, alpha=0.7)
    plt.xlabel("n_trades")
    plt.ylabel("sharpe")
    plt.title("Sharpe vs Number of Trades")
    plt.tight_layout()
    plt.show()


def lines_sharpe_vs_premium(df: pd.DataFrame) -> None:
    """
    For each (mae_window, edge_mae_mult), plot Sharpe vs premium_risk_pct.
    This shows how leverage affects performance shape.
    """
    plt.figure()

    # only show combos with at least 3 points (all 3 premium levels)
    grouped = df.groupby(["mae_window", "edge_mae_mult"])
    for (mae, edge), grp in grouped:
        if grp["premium_risk_pct"].nunique() < 3:
            continue
        grp_sorted = grp.sort_values("premium_risk_pct")
        plt.plot(
            grp_sorted["premium_risk_pct"],
            grp_sorted["sharpe"],
            marker="o",
            linewidth=1,
        )

    plt.xlabel("premium_risk_pct")
    plt.ylabel("sharpe")
    plt.title("Sharpe vs premium_risk_pct (per mae/edge combo)")
    plt.tight_layout()
    plt.show()


def main():
    df = load_results("param_results.csv")
    print_summary(df)

    # Unique premium levels (sorted)
    prem_levels = sorted(df["premium_risk_pct"].unique())
    print("\nPremium levels found:", prem_levels)

    # 1) Heatmaps per premium level
    for prem in prem_levels:
        print(f"\n=== Heatmap for premium_risk_pct = {prem} ===")
        heatmap_for_premium(df, prem)

    # 2) Scatter: Sharpe vs drawdown
    scatter_sharpe_vs_drawdown(df)

    # 3) Scatter: Sharpe vs number of trades
    scatter_sharpe_vs_trades(df)

    # 4) Lines: Sharpe vs premium per mae/edge combo
    lines_sharpe_vs_premium(df)


if __name__ == "__main__":
    main()
