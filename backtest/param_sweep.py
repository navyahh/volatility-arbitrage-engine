# backtest/param_sweep.py

import itertools

import pandas as pd

from run_backtest import run_backtest


def main():
    mae_windows = [20, 40, 60, 100]
    edge_mults = [1.0, 1.25, 1.5, 2.0]
    premium_pcts = [0.005, 0.0075, 0.01]

    rows: list[dict] = []

    for mae, edge_mul, prem in itertools.product(mae_windows, edge_mults, premium_pcts):
        stats = run_backtest(
            mae_window=mae,
            edge_mae_mult=edge_mul,
            premium_risk_pct=prem,
            save_csv=False,
            do_plot=False,
        )

        row = {
            "mae_window": mae,
            "edge_mae_mult": edge_mul,
            "premium_risk_pct": prem,
            "ann_return": stats["ann_return"],
            "ann_vol": stats["ann_vol"],
            "sharpe": stats["sharpe"],
            "max_drawdown": stats["max_drawdown"],
            "n_trades": stats["n_trades"],
        }
        rows.append(row)

        print(
            f"mae={mae:3d}, edge={edge_mul:4.2f}, prem={prem:7.4f} "
            f"-> sharpe={stats['sharpe']:.3f}, n_trades={stats['n_trades']}"
        )

    df = pd.DataFrame(rows)
    df.to_csv("param_results.csv", index=False)
    print(f"\nSaved param_results.csv with {len(df)} rows")


if __name__ == "__main__":
    main()
