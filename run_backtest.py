@"
print("RUNBACKTEST: starting")

import os, sys, traceback
print("PYTHON:", sys.executable)

try:
    import yaml, numpy as np, pandas as pd
    print("IMPORTS OK")
except Exception as e:
    print("IMPORT FAIL:", e)
    raise

BASE = os.path.dirname(__file__)
CFG_PATH = os.path.join(BASE, "configs", "default.yaml")
print("CFG PATH:", CFG_PATH)

try:
    with open(CFG_PATH, "r") as f:
        CFG = yaml.safe_load(f)
    print("CFG LOADED")
except Exception as e:
    print("CFG LOAD FAIL:", e)
    traceback.print_exc()
    sys.exit(1)

try:
    from data_pipeline.loader import make_synthetic_daily
    from execution.bs import call_price, put_price, call_delta, put_delta
    print("MODULE IMPORTS OK")
except Exception as e:
    print("MODULE IMPORT FAIL:", e)
    traceback.print_exc()
    sys.exit(1)

def perf_stats(equity, freq=252):
    import numpy as _np, pandas as _pd
    ret = equity.pct_change().dropna()
    ann_ret = (equity.iloc[-1]/equity.iloc[0])**(freq/len(ret)) - 1 if len(ret)>0 else 0.0
    ann_vol = ret.std()*_np.sqrt(freq) if len(ret)>0 else 0.0
    sharpe = ann_ret/ann_vol if ann_vol>1e-9 else 0.0
    dd = (equity/equity.cummax() - 1).min()
    return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "max_drawdown": dd}

def naive_forecast(rv_series):
    return rv_series.shift(1).rolling(5).mean().bfill()

def main():
    print("MAIN: loading synthetic data")
    df = make_synthetic_daily(seed=CFG["data"]["seed"], days=CFG["data"]["synthetic_days"])
    print("MAIN: data loaded", len(df))
    df["forecast"] = naive_forecast(df["rv_next"])
    import pandas as pd
    window = 60
    mae = (df["rv_next"] - df["forecast"]).abs().rolling(window).mean()
    band = CFG["strategy"]["edge_threshold_mae_mult"] * mae
    delta = df["iv_atm"] - df["forecast"]
    signal = pd.Series(0, index=df.index)
    signal[delta >= band] = -1
    signal[delta <= -band] = +1

    r, q, t = CFG["riskfree_rate"], CFG["dividend_yield"], 2/252
    equity = [100_000.0]
    pnl = [0.0]
    mult = 100
    print("MAIN: entering loop")

    for i in range(len(df)-1):
        sgn = signal.iloc[i]
        if sgn == 0:
            equity.append(equity[-1]); pnl.append(0.0); continue
        S0, S1 = df.loc[i, "close"], df.loc[i+1, "close"]
        K = round(S0)
        iv0, iv1 = df.loc[i, "iv_atm"], df.loc[i+1, "iv_atm"]
        c0, p0 = call_price(S0,K,r,q,iv0,t), put_price(S0,K,r,q,iv0,t)
        dc0, dp0 = call_delta(S0,K,r,q,iv0,t), put_delta(S0,K,r,q,iv0,t)
        straddle = c0 + p0
        contracts = max(1, int(equity[-1]*CFG["strategy"]["sizing_premium_at_risk_pct"]/(straddle*mult)))
        entry = -sgn*straddle*contracts*mult
        net_delta = (dc0 + dp0)*contracts*mult
        hedge_shares = int(round(-net_delta))
        hedge_entry = -hedge_shares*S0
        c1, p1 = call_price(S1,K,r,q,iv1,t), put_price(S1,K,r,q,iv1,t)
        exit_val = sgn*(c1+p1)*contracts*mult
        hedge_exit = hedge_shares*S1
        opt_cost = (CFG["costs"]["option_half_spread"] + CFG["costs"]["option_slippage"] + CFG["costs"]["option_commission"]) * (2*contracts) * 2
        stk_cost = CFG["costs"]["stock_half_spread"] * abs(hedge_shares) * 2
        trade_pnl = entry + hedge_entry + exit_val + hedge_exit - opt_cost - stk_cost
        pnl.append(trade_pnl)
        equity.append(equity[-1] + trade_pnl)

    import pandas as pd
    eq = pd.Series(equity, index=df.index)
    stats = perf_stats(eq)
    print("RESULTS:", {k: round(v,4) for k,v in stats.items()})
    pd.DataFrame({"date": df["date"], "equity": eq, "pnl": pnl}).to_csv("backtest_results.csv", index=False)
    print("Saved backtest_results.csv")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("FATAL:", e)
        traceback.print_exc()
        raise
"@ | Out-File -Encoding utf8 .\run_backtest.py
