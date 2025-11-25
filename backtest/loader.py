# backtest/loader.py

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ---------- Generic price loader (for SPY or any ticker) ----------

def load_price_data(
    symbol: str,
    start: str = "2015-01-01",
    end: str | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Load daily OHLCV data for a single symbol using yfinance.
    Handles both normal and MultiIndex columns.
    Caches into <project_root>/data/.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    cache_name = f"{symbol}_{start}_{end or 'latest'}.csv"
    cache_path = data_dir / cache_name

    if cache and cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")

    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise RuntimeError(f"No data downloaded for {symbol} from {start} to {end}")

    # If MultiIndex columns (e.g. ('Adj Close','SPY')), collapse to single symbol
    if isinstance(df.columns, pd.MultiIndex):
        # try to select the last level as symbol
        if symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, level=-1, axis=1)

    cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    if not cols:
        raise RuntimeError(f"No standard price columns in dataframe for {symbol}: {list(df.columns)}")

    df = df[cols]
    df.index.name = "Date"

    if cache:
        df.to_csv(cache_path)

    return df


# ---------- Synthetic daily IV/RV dataset (for testing core engine) ----------

def make_synthetic_daily(seed: int = 42, days: int = 600) -> pd.DataFrame:
    """
    Create synthetic daily close prices, next-day realized vol, and ATM implied vol.
    Columns:
      date, close, rv_next, iv_atm
    """
    rng = np.random.default_rng(seed)

    # Synthetic price path (geometric-like random walk)
    S0 = 400.0
    mu = 0.07 / 252              # daily drift
    sigma = 0.15 / np.sqrt(252)  # daily vol

    prices = [S0]
    for _ in range(days - 1):
        shock = rng.normal(mu, sigma)
        prices.append(prices[-1] * (1 + shock))

    prices = np.array(prices, dtype=float)

    # Synthetic realized volatility (annualized)
    rv = np.abs(rng.normal(0.15, 0.05, size=days))
    rv_next = rv.copy()

    # Synthetic implied vol (IV = RV + VRP + noise)
    iv_atm = rv + rng.normal(0.02, 0.01, size=days)

    dates = pd.date_range(start="2020-01-01", periods=days, freq="B")

    df = pd.DataFrame(
        {
            "date": dates,
            "close": prices,
            "rv_next": rv_next,
            "iv_atm": iv_atm,
        }
    )

    return df


# ---------- Helpers for real IV/RV datasets ----------

def _get_price_series(df: pd.DataFrame, symbol_hint: str) -> pd.Series:
    """
    Extract a 1-D price series from a yfinance DataFrame that may have
    normal columns or MultiIndex columns like ('Adj Close','SPY').
    """
    if isinstance(df.columns, pd.MultiIndex):
        for col_name in ["Adj Close", "Close"]:
            key = (col_name, symbol_hint)
            if key in df.columns:
                s = df[key].astype(float).copy()
                s.name = symbol_hint
                return s
    else:
        for col_name in ["Adj Close", "Close"]:
            if col_name in df.columns:
                s = df[col_name].astype(float).copy()
                s.name = symbol_hint
                return s

    raise RuntimeError(f"Could not find price column for {symbol_hint} in columns: {list(df.columns)}")


# ---------- Real-data IV/RV using SPY + VIX (30-day vol) ----------

def make_spy_vix_daily(
    start: str = "2010-01-01",
    end: str | None = None,
    rv_window: int = 21,
) -> pd.DataFrame:
    """
    Real IV/RV dataset using:
      - SPY as underlying
      - ^VIX as implied vol proxy (30-day SPX IV)

    Returns columns:
      date, close, rv_next, iv_atm
    """
    spy_raw = yf.download("SPY", start=start, end=end, auto_adjust=False, progress=False)
    vix_raw = yf.download("^VIX", start=start, end=end, auto_adjust=False, progress=False)

    if spy_raw.empty or vix_raw.empty:
        raise RuntimeError("Failed to download SPY or VIX data")

    spy = _get_price_series(spy_raw, "SPY")
    vix = _get_price_series(vix_raw, "^VIX")

    df = pd.DataFrame({"SPY": spy, "VIX": vix}).dropna()
    df.index.name = "Date"

    # Realized vol from SPY log-returns
    ret = np.log(df["SPY"] / df["SPY"].shift(1))
    rv_trailing = ret.rolling(rv_window).std() * np.sqrt(252)
    rv_next = rv_trailing.shift(-1)

    # Implied vol from VIX (annualized %, convert to decimal)
    iv_atm = df["VIX"] / 100.0

    out = pd.DataFrame(
        {
            "date": df.index,
            "close": df["SPY"],
            "rv_next": rv_next,
            "iv_atm": iv_atm,
        }
    ).dropna().reset_index(drop=True)

    return out


# ---------- Real-data IV/RV using SPY + VIX9D (9-day vol) ----------

def make_spy_vix9d_daily(
    start: str = "2011-01-01",
    end: str | None = None,
    rv_window: int = 21,
) -> pd.DataFrame:
    """
    Real IV/RV dataset using:
      - SPY as underlying
      - ^VIX9D as 9-day SPX implied vol index

    Returns columns:
      date, close, rv_next, iv_atm
    """
    spy_raw = yf.download("SPY", start=start, end=end, auto_adjust=False, progress=False)
    vix_raw = yf.download("^VIX9D", start=start, end=end, auto_adjust=False, progress=False)

    if spy_raw.empty or vix_raw.empty:
        raise RuntimeError("Failed to download SPY or VIX9D data")

    spy = _get_price_series(spy_raw, "SPY")
    vix9d = _get_price_series(vix_raw, "^VIX9D")

    df = pd.DataFrame({"SPY": spy, "VIX9D": vix9d}).dropna()
    df.index.name = "Date"

    # Realized vol from SPY log-returns
    ret = np.log(df["SPY"] / df["SPY"].shift(1))
    rv_trailing = ret.rolling(rv_window).std() * np.sqrt(252)
    rv_next = rv_trailing.shift(-1)

    # Implied vol from VIX9D (annualized %, convert to decimal)
    iv_atm = df["VIX9D"] / 100.0

    out = pd.DataFrame(
        {
            "date": df.index,
            "close": df["SPY"],
            "rv_next": rv_next,
            "iv_atm": iv_atm,
        }
    ).dropna().reset_index(drop=True)

    return out


# ---------- Backwards-compat alias (if you still call make_spy_vxst_daily) ----------

def make_spy_vxst_daily(
    start: str = "2011-01-01",
    end: str | None = None,
    rv_window: int = 21,
) -> pd.DataFrame:
    """
    Alias for make_spy_vix9d_daily to keep older scripts working.
    VXST ~ VIX9D (short-term SPX IV).
    """
    return make_spy_vix9d_daily(start=start, end=end, rv_window=rv_window)
