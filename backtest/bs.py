from math import log, sqrt, exp
from scipy.stats import norm

def _d1(S, K, r, q, sigma, t):
    return (log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt(t))

def call_price(S, K, r, q, sigma, t):
    d1 = _d1(S, K, r, q, sigma, t)
    d2 = d1 - sigma * sqrt(t)
    return S * exp(-q*t) * norm.cdf(d1) - K * exp(-r*t) * norm.cdf(d2)

def put_price(S, K, r, q, sigma, t):
    d1 = _d1(S, K, r, q, sigma, t)
    d2 = d1 - sigma * sqrt(t)
    return K * exp(-r*t) * norm.cdf(-d2) - S * exp(-q*t) * norm.cdf(-d1)

def call_delta(S, K, r, q, sigma, t):
    return exp(-q*t) * norm.cdf(_d1(S, K, r, q, sigma, t))

def put_delta(S, K, r, q, sigma, t):
    return exp(-q*t) * (norm.cdf(_d1(S, K, r, q, sigma, t)) - 1.0)
