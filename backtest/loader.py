import numpy as np
import pandas as pd

def make_synthetic_daily(seed=42, days=600):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)

    S0, mu, sigma = 450.0, 0.08, 0.18
    dt = 1 / 252
    prices = [S0]
    for _ in range(days - 1):
        z = rng.standard_normal()
        prices.append(
            prices[-1]
            * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        )
    close = np.array(prices)

    vol = [0.18]
    kappa, theta, vol_of_vol = 2.0, 0.18, 0.35
    for _ in range(days - 1):
        dv = (
            kappa * (theta - vol[-1]) * dt
            + vol_of_vol * np.sqrt(dt) * rng.standard_normal()
        )
        new_vol = np.clip(vol[-1] + dv, 0.05, 0.8)
        vol.append(new_vol)
    vol = np.array(vol)

    iv = np.clip(vol + 0.02 + 0.05 * rng.standard_normal(days), 0.05, 1.0)

    df = pd.DataFrame({
        "date": dates,
        "close": close,
        "iv_atm": iv,
        "rv_next": np.roll(vol, -1),
    }).dropna()

    df["vix"] = df["iv_atm"] * 100.0
    return df
