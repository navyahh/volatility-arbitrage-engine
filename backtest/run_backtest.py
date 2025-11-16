# backtest.py
import numpy as np
import pandas as pd

from loader import make_synthetic_daily
from bs import call_price, put_price, call_delta, put_delta


def perf_stats(equity, freq=252):
    """
    equity: pandas Series of account value.
    freq: trading days per year.
    Sharpe ratio (return per unit of volatility),
    max drawdown (worst peak-to-trough loss).
    """
    ret = equity.pct_change().dropna()
    if len(ret) == 0:
        return dict(ann_return=0.0, ann_vol=0.0, sharpe=0.0, max_drawdown=0.0)

    ann_ret = (equity.iloc[-1] / equity.iloc[0]) ** (freq / len(ret)) - 1
    ann_vol = ret.std() * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-9 else 0.0
    dd = (equity / equity.cummax() - 1).min()
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": dd,
    }


def naive_forecast(rv_series):
    """
    Five-day rolling mean of past realized volatility (dumb forecast).
    """
    return rv_series.shift(1).rolling(5).mean().bfill()


def main():
    # ----- CONFIG -----
    seed = 42
    days = 600
    r = 0.03          # risk-free rate (annual)
    q = 0.015         # dividend yield (annual)
    t = 2 / 252       # ~2 trading days to expiry
    equity0 = 100_000
    premium_risk_pct = 0.0075  # fraction of equity at risk via option premium
    mae_window = 60
    edge_mae_mult = 1.25       # trade when |IV - forecast| > 1.25 * MAE
    contract_mult = 100        # shares per option contract

    # transaction costs
    opt_half_spread = 0.02     # dollars per option
    opt_slippage = 0.01        # dollars per option
    opt_commission = 0.65      # dollars per contract
    stock_half_spread = 0.005  # dollars per share

    # ----- DATA -----
    df = make_synthetic_daily(seed=seed, days=days)
    df["forecast"] = naive_forecast(df["rv_next"])

    # Edge (variance risk premium): IV - forecast RV
    err = (df["rv_next"] - df["forecast"]).abs()
    mae = err.rolling(mae_window).mean()
    band = edge_mae_mult * mae
    edge = df["iv_atm"] - df["forecast"]

    # signals: short vol if IV >> forecast, long vol if IV << forecast
    signal = pd.Series(0, index=df.index)
    signal[edge >= band] = -1   # short vol (sell straddle)
    signal[edge <= -band] = +1  # long vol (buy straddle)

    # ----- BACKTEST LOOP -----
    equity = [equity0]
    pnl = [0.0]

    for i in range(len(df) - 1):
        sgn = signal.iloc[i]
        if sgn == 0:
            equity.append(equity[-1])
            pnl.append(0.0)
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

        # delta hedge at entry (delta-neutral = no directional exposure)
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

    equity_series = pd.Series(equity, index=df.index)
    stats = perf_stats(equity_series)

    print("Performance (synthetic demo):")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")

    out = pd.DataFrame({"date": df["date"], "equity": equity_series, "pnl": pnl})
    out.to_csv("backtest_results.csv", index=False)
    print("Saved backtest_results.csv")


if __name__ == "__main__":
    main()
