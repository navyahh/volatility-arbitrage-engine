# backtest/run_backtest.py

from equity_plot import plot_equity
from loader import make_synthetic_daily

import numpy as np
import pandas as pd
from bs import call_price, put_price, call_delta, put_delta


def perf_stats(equity: pd.Series, freq: int = 252) -> dict:
    """
    Compute annualized return, vol, Sharpe, and max drawdown from an equity curve.
    """
    ret = equity.pct_change().dropna()
    if len(ret) == 0 or equity.iloc[0] <= 0:
        return dict(
            ann_return=float("nan"),
            ann_vol=0.0,
            sharpe=float("nan"),
            max_drawdown=float("nan"),
        )

    final_ratio = equity.iloc[-1] / equity.iloc[0]
    if final_ratio <= 0:
        ann_ret = -1.0
    else:
        ann_ret = final_ratio ** (freq / len(ret)) - 1

    ann_vol = ret.std() * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-9 else float("nan")
    max_dd = (equity / equity.cummax() - 1).min()

    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def naive_forecast(rv_series: pd.Series) -> pd.Series:
    """
    Five-day rolling mean of past realized volatility (dumb forecast).
    Uses 1-day lag to avoid look-ahead.
    """
    return rv_series.shift(1).rolling(5).mean().bfill()


def run_backtest(
    *,
    seed: int = 42,
    days: int = 600,
    r: float = 0.03,
    q: float = 0.015,
    t: float = 2 / 252,
    equity0: float = 100_000.0,
    premium_risk_pct: float = 0.0075,
    mae_window: int = 60,
    edge_mae_mult: float = 1.25,
    contract_mult: int = 100,
    opt_half_spread: float = 0.02,
    opt_slippage: float = 0.01,
    opt_commission: float = 0.65,
    stock_half_spread: float = 0.005,
    save_csv: bool = True,
    csv_path: str = "backtest_results.csv",
    do_plot: bool = True,
    data_df: pd.DataFrame | None = None,
    min_trade_gap: int = 0,
) -> dict:
    """
    Core vol-arb backtest.

    If data_df is None:
        uses synthetic data from make_synthetic_daily().
    Else:
        expects columns: date, close, rv_next, iv_atm.
        If 'forecast' exists, it is used as RV forecast.
        Otherwise a naive rolling forecast is built.

    Returns dict with performance stats + n_trades + params.
    """

    # ---------------- DATA ----------------
    if data_df is None:
        df = make_synthetic_daily(seed=seed, days=days)
    else:
        df = data_df.copy().reset_index(drop=True)

    # ensure required columns exist
    for col in ["date", "close", "rv_next", "iv_atm"]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from data_df")

    # If no forecast provided, use naive forecast
    if "forecast" not in df.columns:
        df["forecast"] = naive_forecast(df["rv_next"])

    # Edge (variance risk premium): IV - forecast RV
    err = (df["rv_next"] - df["forecast"]).abs()
    mae = err.rolling(mae_window).mean()
    band = edge_mae_mult * mae
    edge = df["iv_atm"] - df["forecast"]

    signal = pd.Series(0, index=df.index)
    signal[edge >= band] = -1   # short vol
    signal[edge <= -band] = +1  # long vol

    df["edge"] = edge
    df["signal"] = signal

    # ---------------- BACKTEST LOOP ----------------
    equity = [equity0]
    pnl = [0.0]
    used_signals = [0]  # effective signals (after min_trade_gap)
    last_trade_i = -10**9

    for i in range(len(df) - 1):
        raw_sgn = signal.iloc[i]

        # enforce minimum gap between trades
        if raw_sgn != 0 and (i - last_trade_i) < min_trade_gap:
            sgn = 0
        else:
            sgn = raw_sgn

        if sgn == 0:
            equity.append(equity[-1])
            pnl.append(0.0)
            used_signals.append(0)
            continue

        S0 = df.loc[i, "close"]
        S1 = df.loc[i + 1, "close"]
        K = round(S0)
        iv0 = df.loc[i, "iv_atm"]
        iv1 = df.loc[i + 1, "iv_atm"]

        # prices and deltas at entry
        c0 = call_price(S0, K, r, q, iv0, t)
        p0 = put_price(S0, K, r, q, iv0, t)
        dc0 = call_delta(S0, K, r, q, iv0, t)
        dp0 = put_delta(S0, K, r, q, iv0, t)

        straddle_price = c0 + p0

        # position sizing: premium risk = premium_risk_pct * equity
        max_premium = equity[-1] * premium_risk_pct
        contracts = max(1, int(max_premium / (straddle_price * contract_mult)))

        # option entry cash
        entry_options = -sgn * straddle_price * contracts * contract_mult

        # delta hedge at entry (delta-neutral)
        net_delta = (dc0 + dp0) * contracts * contract_mult
        hedge_shares = int(round(-net_delta))
        hedge_entry = -hedge_shares * S0

        # mark to next day
        c1 = call_price(S1, K, r, q, iv1, t)
        p1 = put_price(S1, K, r, q, iv1, t)
        exit_options = sgn * (c1 + p1) * contracts * contract_mult
        hedge_exit = hedge_shares * S1

        # transaction costs (options + stock, in & out)
        opt_cost = (opt_half_spread + opt_slippage + opt_commission) * (2 * contracts) * 2
        stk_cost = stock_half_spread * abs(hedge_shares) * 2

        trade_pnl = entry_options + hedge_entry + exit_options + hedge_exit - opt_cost - stk_cost
        pnl.append(trade_pnl)
        equity.append(equity[-1] + trade_pnl)
        used_signals.append(sgn)
        last_trade_i = i

    equity_series = pd.Series(equity, index=df.index)
    stats = perf_stats(equity_series)

    signal_used_series = pd.Series(used_signals, index=df.index)
    n_trades = int((signal_used_series != 0).sum())

    out = pd.DataFrame(
        {
            "date": df["date"],
            "close": df["close"],
            "rv_next": df["rv_next"],
            "forecast": df["forecast"],
            "iv_atm": df["iv_atm"],
            "edge": df["edge"],
            "signal_raw": df["signal"].astype(int),
            "signal_used": signal_used_series.astype(int),
            "equity": equity_series,
            "pnl": pnl,
        }
    )

    if save_csv:
        out.to_csv(csv_path, index=False)

    if do_plot:
        try:
            plot_equity(out)
        except Exception:
            pass

    result = dict(stats)
    result.update(
        dict(
            n_trades=n_trades,
            mae_window=mae_window,
            edge_mae_mult=edge_mae_mult,
            premium_risk_pct=premium_risk_pct,
            min_trade_gap=min_trade_gap,
        )
    )
    return result


def main():
    stats = run_backtest()
    print("Performance (synthetic demo):")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print("Saved backtest_results.csv")


if __name__ == "__main__":
    main()
