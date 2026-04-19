"""
Derivexus — Options Pricing, Greeks & Volatility Surface Analysis Engine
Built by Marin Xhemollari | marinxhemollari.com

v2.0 — Adds quadratic smile fitting and options mispricing scanner.

Implements:
- Generalized Black-Scholes-Merton pricing with continuous dividend yield
- Full Greeks suite: Delta, Gamma, Theta, Vega, Rho
- Greeks sensitivity analysis across underlying price range
- Payoff & P&L diagrams at expiration
- Implied Volatility solver (Brent's method)
- IV Surface from live options chain data (yfinance)
- Quadratic smile fitting per expiration (NEW)
- Mispricing scanner — flags contracts with IV residuals > 2σ (NEW)
- Fair-price reconstruction from fitted IV surface (NEW)
- Put-Call Parity verification

Notes:
- Pricing assumes European exercise.
- Dividend yield q is continuously compounded.
- Smile fitting is an empirical diagnostic, not an arbitrage-free model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import brentq
import yfinance as yf
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Derivexus",
    page_icon="https://marinxhemollari.com/frog-favicon.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

:root {
    --charcoal-900: #0a0a0a;
    --charcoal-800: #0d0d0d;
    --charcoal-700: #141414;
    --charcoal-600: #1a1a1a;
    --charcoal-500: #222222;
    --charcoal-400: #2a2a2a;
    --charcoal-300: #333333;
    --emerald-500: #2ecc71;
    --emerald-400: #27ae60;
    --emerald-300: #1abc9c;
    --text-primary: #e8e8e8;
    --text-secondary: rgba(232, 232, 232, 0.6);
    --text-muted: rgba(232, 232, 232, 0.35);
    --glass-bg: rgba(20, 20, 20, 0.6);
    --glass-border: rgba(46, 204, 113, 0.12);
    --glass-border-hover: rgba(46, 204, 113, 0.25);
    --call-color: #22c55e;
    --put-color: #ef4444;
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 20px rgba(46, 204, 113, 0.08); }
    50%      { box-shadow: 0 0 40px rgba(46, 204, 113, 0.18); }
}
@keyframes logoFloat {
    0%, 100% { transform: translateY(0px); }
    50%      { transform: translateY(-6px); }
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--charcoal-900) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.main .block-container {
    padding-top: 1rem !important;
    max-width: 1400px;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--charcoal-800); }
::-webkit-scrollbar-thumb { background: var(--charcoal-400); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--emerald-400); }

.dx-header {
    background: linear-gradient(135deg,
        var(--charcoal-800) 0%, rgba(46, 204, 113, 0.06) 25%,
        var(--charcoal-700) 50%, rgba(26, 188, 156, 0.06) 75%,
        var(--charcoal-800) 100%);
    background-size: 400% 400%;
    animation: gradientShift 12s ease infinite, fadeInUp 0.8s ease-out;
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative; overflow: hidden;
}
.dx-header::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(46, 204, 113, 0.04) 0%, transparent 70%);
    pointer-events: none;
}
.dx-header-content {
    display: flex;
    align-items: center;
    gap: 2.5rem;
    position: relative;
    z-index: 1;
    min-width: 0;
}
.dx-logo {
    width: 64px; height: 64px;
    animation: logoFloat 4s ease-in-out infinite;
    filter: drop-shadow(0 0 8px rgba(46, 204, 113, 0.25));
    flex-shrink: 0;
}
.dx-title-wrap { display: flex; flex-direction: column; min-width: 0; flex: 1; }
.dx-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3rem; font-weight: 700;
    letter-spacing: -0.02em; line-height: 1.1; margin: 0;
    display: flex; align-items: baseline; flex-wrap: wrap; gap: 0.25rem;
}
.dx-title-plain { color: var(--text-primary); }
.dx-title-accent {
    color: var(--emerald-500);
    background: linear-gradient(135deg, #2ecc71 0%, #1abc9c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.dx-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem; color: var(--text-secondary);
    margin-top: 0.35rem; font-weight: 300; letter-spacing: 0.02em;
}
.dx-version {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; color: var(--emerald-500);
    background: rgba(46, 204, 113, 0.08);
    border: 1px solid rgba(46, 204, 113, 0.15);
    padding: 0.2rem 0.6rem; border-radius: 4px;
    letter-spacing: 0.05em;
    align-self: center; flex-shrink: 0; margin-left: 0.75rem;
}

div[data-testid="stMetric"] {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    padding: 1.2rem 1.4rem !important;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    animation: fadeIn 0.5s ease-out, pulseGlow 4s ease-in-out infinite;
}
div[data-testid="stMetric"]:hover {
    border-color: var(--glass-border-hover) !important;
    background: rgba(20, 20, 20, 0.8) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(46, 204, 113, 0.1);
}
div[data-testid="stMetric"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important; font-weight: 500 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important; letter-spacing: 0.08em !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.4rem !important; font-weight: 600 !important;
    color: var(--emerald-500) !important;
    white-space: nowrap !important;
    overflow: visible !important;
}

section[data-testid="stSidebar"] {
    background: var(--charcoal-800) !important;
    border-right: 1px solid rgba(46, 204, 113, 0.08) !important;
}
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 0.85rem !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
    color: var(--text-muted) !important; margin-top: 1.5rem !important;
}

h2 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 600 !important;
    animation: fadeInUp 0.6s ease-out;
}
h3, h4 { font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; }

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, var(--emerald-400), var(--emerald-300)) !important;
    color: var(--charcoal-900) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 0.85rem !important;
    letter-spacing: 0.06em !important; text-transform: uppercase !important;
    border: none !important; border-radius: 8px !important;
    box-shadow: 0 4px 16px rgba(46, 204, 113, 0.2) !important;
    transition: all 0.3s ease !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(46, 204, 113, 0.35) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px !important; background: var(--charcoal-700) !important;
    border-radius: 10px !important; padding: 4px !important;
    border: 1px solid rgba(255,255,255,0.04);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    border-radius: 8px !important; padding: 8px 16px !important;
    color: var(--text-secondary) !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(46, 204, 113, 0.1) !important;
    color: var(--emerald-500) !important;
}

.dx-label {
    font-family: 'DM Sans', sans-serif; font-size: 0.65rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.12em;
    padding: 0.25rem 0.65rem; border-radius: 4px;
    display: inline-block; margin-bottom: 0.5rem;
}
.dx-call { color: #22c55e; background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.2); }
.dx-put { color: #ef4444; background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2); }
.dx-scanner { color: #a78bfa; background: rgba(167,139,250,0.08); border: 1px solid rgba(167,139,250,0.2); }

[data-testid="stPlotlyChart"] {
    border: 1px solid var(--glass-border); border-radius: 12px;
    overflow: hidden; animation: fadeIn 0.5s ease-out;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
[data-testid="stPlotlyChart"]:hover {
    border-color: var(--glass-border-hover);
    box-shadow: 0 4px 24px rgba(46, 204, 113, 0.06);
}

hr {
    border: none !important; height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(46,204,113,0.15), transparent) !important;
    margin: 2rem 0 !important;
}

blockquote {
    border-left: 3px solid var(--emerald-500) !important;
    background: rgba(46,204,113,0.03) !important;
    padding: 0.8rem 1.2rem !important; border-radius: 0 8px 8px 0 !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.88rem !important;
    color: var(--text-secondary) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important; overflow: hidden;
    animation: fadeIn 0.5s ease-out;
}

.dx-footer {
    text-align: center; color: var(--text-muted);
    font-family: 'DM Sans', sans-serif; font-size: 0.78rem;
    padding: 2rem 0 1rem; animation: fadeIn 0.8s ease-out;
}
.dx-footer a { color: var(--emerald-500); text-decoration: none; }
.dx-footer a:hover { color: var(--emerald-300); }
.dx-footer-mono {
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    color: var(--text-muted); opacity: 0.6; margin-top: 0.5rem;
}

[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: rgba(46,204,113,0.1) !important;
    border: 1px solid rgba(46,204,113,0.2) !important;
    color: var(--emerald-500) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important; border-radius: 6px !important;
}

@media (max-width: 768px) {
    .dx-header { padding: 1.5rem 1.5rem; }
    .dx-header-content { gap: 1.25rem; flex-wrap: wrap; }
    .dx-logo { width: 48px; height: 48px; }
    .dx-title { font-size: 2rem; }
    .dx-subtitle { font-size: 0.85rem; }
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# GENERALIZED BLACK-SCHOLES-MERTON MATH
# ──────────────────────────────────────────────────────────────

def d1(S, K, T, r, sigma, q=0.0):
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma, q=0.0):
    return d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)

def bs_call(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1v = d1(S, K, T, r, sigma, q)
    d2v = d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.cdf(d1v) - K * np.exp(-r * T) * norm.cdf(d2v)

def bs_put(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1v = d1(S, K, T, r, sigma, q)
    d2v = d2(S, K, T, r, sigma, q)
    return K * np.exp(-r * T) * norm.cdf(-d2v) - S * np.exp(-q * T) * norm.cdf(-d1v)


def greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    if T <= 0 or sigma <= 0:
        return {"Delta": 0, "Gamma": 0, "Theta": 0, "Vega": 0, "Rho": 0}
    d1_val = d1(S, K, T, r, sigma, q)
    d2_val = d2(S, K, T, r, sigma, q)
    sqrt_T = np.sqrt(T)
    gamma = np.exp(-q * T) * norm.pdf(d1_val) / (S * sigma * sqrt_T)
    vega = S * np.exp(-q * T) * norm.pdf(d1_val) * sqrt_T / 100
    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1_val)
        theta = (-(S * np.exp(-q * T) * norm.pdf(d1_val) * sigma) / (2 * sqrt_T)
                 - r * K * np.exp(-r * T) * norm.cdf(d2_val)
                 + q * S * np.exp(-q * T) * norm.cdf(d1_val)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2_val) / 100
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1_val)
        theta = (-(S * np.exp(-q * T) * norm.pdf(d1_val) * sigma) / (2 * sqrt_T)
                 + r * K * np.exp(-r * T) * norm.cdf(-d2_val)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1_val)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2_val) / 100
    return {"Delta": round(delta, 4), "Gamma": round(gamma, 4),
            "Theta": round(theta, 4), "Vega": round(vega, 4), "Rho": round(rho, 4)}


def implied_vol(market_price, S, K, T, r, option_type="call", q=0.0):
    if T <= 0:
        return np.nan
    try:
        func = bs_call if option_type == "call" else bs_put
        iv = brentq(lambda sigma: func(S, K, T, r, sigma, q) - market_price,
                    0.001, 5.0, maxiter=200)
        return iv
    except (ValueError, RuntimeError):
        return np.nan


# ──────────────────────────────────────────────────────────────
# SMILE FITTING & MISPRICING SCANNER (v2.0)
# ──────────────────────────────────────────────────────────────

def fit_quadratic_smile(log_moneyness, iv_values):
    """
    Fit a quadratic polynomial σ(k) = a + b·k + c·k² to a single expiration's smile,
    where k = ln(K/F) is log-moneyness (F = forward = S·e^((r-q)T)).

    Returns:
        coeffs: array [a, b, c]
        fitted: fitted IV values at each input k
        r_squared: goodness-of-fit
        residuals: market IV - fitted IV at each input
    """
    k = np.asarray(log_moneyness, dtype=float)
    iv = np.asarray(iv_values, dtype=float)

    # Filter out non-finite values
    mask = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    if mask.sum() < 4:
        return None
    k_ = k[mask]
    iv_ = iv[mask]

    # Fit σ(k) = a + b·k + c·k²  (equivalent to np.polyfit with deg=2)
    coeffs = np.polyfit(k_, iv_, 2)  # returns [c, b, a] in descending power
    # Rearrange to [a, b, c] in ascending power
    a = coeffs[2]
    b = coeffs[1]
    c = coeffs[0]

    fitted = a + b * k + c * k**2  # evaluate at ALL k, even those we masked, for residuals

    residuals = iv - fitted  # NaN where iv was NaN; acceptable

    # R² on the fit sample
    iv_mean = iv_.mean()
    ss_res = np.sum((iv_ - (a + b * k_ + c * k_**2))**2)
    ss_tot = np.sum((iv_ - iv_mean)**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "coeffs": np.array([a, b, c]),
        "fitted": fitted,
        "residuals": residuals,
        "r_squared": r_squared,
        "n_fit": int(mask.sum()),
    }


def build_mispricing_scan(ticker, S, r, q, expirations, max_expiries=6,
                           strike_range_pct=0.15, zscore_threshold=2.0,
                           fetch_chain_fn=None):
    """
    For each expiration:
      1) Pull call chain
      2) Compute mid prices and market IVs
      3) Fit quadratic smile in log-moneyness space
      4) Compute residual z-scores
      5) Flag contracts where |z| > threshold as potentially mispriced
      6) Compute theoretical "fair" price using fitted IV

    Returns a tidy DataFrame with flagged contracts, sorted by |z| descending.
    """
    exp_subset = expirations[:max_expiries]
    flagged_rows = []
    fit_by_expiry = {}

    for exp in exp_subset:
        dte_days = max((datetime.strptime(exp, "%Y-%m-%d") - datetime.today()).days, 1)
        T = dte_days / 365.0
        fwd = S * np.exp((r - q) * T)

        calls, puts = fetch_chain_fn(ticker, exp)
        if calls is None or calls.empty:
            continue

        # Filter: only strikes within ±strike_range_pct of spot, positive volume
        mask = ((calls["strike"] >= S * (1 - strike_range_pct)) &
                (calls["strike"] <= S * (1 + strike_range_pct)) &
                (calls["volume"] > 0))
        df = calls[mask].copy()
        if len(df) < 5:
            continue

        # Mid price
        df["mid"] = np.where(df["bid"] > 0,
                             (df["bid"] + df["ask"]) / 2,
                             df["lastPrice"])
        df = df[df["mid"] > 0.05]  # filter tiny premiums (unreliable IV)
        if len(df) < 5:
            continue

        # Compute IV and log-moneyness
        df["iv_market"] = [
            implied_vol(row["mid"], S, row["strike"], T, r, "call", q)
            for _, row in df.iterrows()
        ]
        df["log_moneyness"] = np.log(df["strike"] / fwd)
        df = df.dropna(subset=["iv_market"])
        if len(df) < 5:
            continue

        # Fit smile
        fit = fit_quadratic_smile(df["log_moneyness"].values, df["iv_market"].values)
        if fit is None:
            continue

        fit_by_expiry[exp] = {
            "dte": dte_days,
            "T": T,
            "fwd": fwd,
            "strikes": df["strike"].values.copy(),
            "iv_market": df["iv_market"].values.copy(),
            "log_moneyness": df["log_moneyness"].values.copy(),
            "fitted": fit["fitted"].copy(),
            "residuals": fit["residuals"].copy(),
            "coeffs": fit["coeffs"],
            "r_squared": fit["r_squared"],
            "n_fit": fit["n_fit"],
        }

        # Z-score of residuals within this expiration
        resid = fit["residuals"]
        resid_std = np.nanstd(resid)
        if resid_std == 0 or np.isnan(resid_std):
            continue
        z = resid / resid_std

        for i, (_, row) in enumerate(df.iterrows()):
            z_i = z[i]
            if not np.isfinite(z_i):
                continue
            iv_fit = fit["fitted"][i]
            if iv_fit <= 0 or not np.isfinite(iv_fit):
                continue
            fair_price = bs_call(S, row["strike"], T, r, iv_fit, q)
            mispricing_pct = (row["mid"] - fair_price) / fair_price * 100 if fair_price > 0 else np.nan

            flagged = abs(z_i) > zscore_threshold

            flagged_rows.append({
                "Expiry": exp,
                "DTE": dte_days,
                "Strike": row["strike"],
                "Moneyness": row["strike"] / S,
                "Market Mid": row["mid"],
                "Market IV": row["iv_market"],
                "Fitted IV": iv_fit,
                "IV Residual": row["iv_market"] - iv_fit,
                "Z-Score": z_i,
                "Fair Price (fitted)": fair_price,
                "Mispricing %": mispricing_pct,
                "Volume": int(row["volume"]) if not np.isnan(row["volume"]) else 0,
                "Flagged": flagged,
            })

    scan_df = pd.DataFrame(flagged_rows) if flagged_rows else pd.DataFrame()
    return scan_df, fit_by_expiry


# ──────────────────────────────────────────────────────────────
# DATA FETCHING
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_options_chain(ticker):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        spot = info.get("regularMarketPrice") or info.get("currentPrice")
        if spot is None:
            hist = tk.history(period="1d")
            spot = float(hist["Close"].iloc[-1]) if not hist.empty else None
        div_yield = info.get("dividendYield") or 0.0
        if div_yield is None:
            div_yield = 0.0
        if div_yield > 1.0:
            div_yield = div_yield / 100.0
        expirations = list(tk.options)
        if not expirations or spot is None:
            return None, None, 0.0
        return float(spot), expirations, float(div_yield)
    except Exception:
        return None, None, 0.0


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_chain_for_expiry(ticker, expiry):
    try:
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiry)
        return chain.calls, chain.puts
    except Exception:
        return None, None


# ──────────────────────────────────────────────────────────────
# PLOTLY THEME
# ──────────────────────────────────────────────────────────────

COLORS = {
    "call": "#22c55e",
    "put": "#ef4444",
    "delta": "#22c55e",
    "gamma": "#a78bfa",
    "theta": "#f59e0b",
    "vega": "#3b82f6",
    "rho": "#ec4899",
    "bg": "rgba(0,0,0,0)",
    "grid": "rgba(46, 204, 113, 0.06)",
    "text": "rgba(232, 232, 232, 0.6)",
    "text_bright": "rgba(232, 232, 232, 0.85)",
    "fitted": "#1abc9c",
    "flag_over": "#ef4444",
    "flag_under": "#22c55e",
}

PLOT_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(color=COLORS["text"], size=12, family="DM Sans, sans-serif"),
    margin=dict(l=48, r=24, t=56, b=72),
    xaxis=dict(gridcolor=COLORS["grid"], zeroline=False,
               linecolor="rgba(46,204,113,0.1)",
               tickfont=dict(family="JetBrains Mono, monospace", size=10)),
    yaxis=dict(gridcolor=COLORS["grid"], zeroline=False,
               linecolor="rgba(46,204,113,0.1)",
               tickfont=dict(family="JetBrains Mono, monospace", size=10)),
    title_font=dict(family="DM Sans, sans-serif", size=16, color=COLORS["text_bright"]),
    hoverlabel=dict(bgcolor="rgba(14,14,14,0.95)", bordercolor="rgba(46,204,113,0.3)",
                    font=dict(family="JetBrains Mono, monospace", size=12, color="#e8e8e8")),
)


# ──────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────

def plot_payoff(S, K, premium_call, premium_put, option_type):
    s_range = np.linspace(S * 0.5, S * 1.5, 500)
    fig = go.Figure()
    if option_type in ["Call", "Both"]:
        intrinsic_call = np.maximum(s_range - K, 0)
        pnl_call = intrinsic_call - premium_call
        fig.add_trace(go.Scatter(x=s_range, y=pnl_call, mode="lines",
                                 name="Call P&L", line=dict(color=COLORS["call"], width=2.5)))
    if option_type in ["Put", "Both"]:
        intrinsic_put = np.maximum(K - s_range, 0)
        pnl_put = intrinsic_put - premium_put
        fig.add_trace(go.Scatter(x=s_range, y=pnl_put, mode="lines",
                                 name="Put P&L", line=dict(color=COLORS["put"], width=2.5)))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(232,232,232,0.2)")
    fig.add_vline(x=K, line_dash="dot", line_color="rgba(46,204,113,0.3)",
                  annotation_text=f"K={K:.0f}", annotation_position="top left",
                  annotation_font=dict(size=10, color=COLORS["text"]))
    fig.add_vline(x=S, line_dash="dot", line_color="rgba(232,232,232,0.2)",
                  annotation_text=f"S={S:.0f}", annotation_position="top right",
                  annotation_font=dict(size=10, color=COLORS["text"]))
    fig.update_layout(**PLOT_LAYOUT, title="Payoff & P&L at Expiration",
                      xaxis_title="Underlying Price at Expiry ($)",
                      yaxis_title="Profit / Loss ($)", height=450,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.28,
                                  xanchor="center", x=0.5, font=dict(family="DM Sans")))
    return fig


def plot_greeks_sensitivity(S, K, T, r, sigma, q):
    s_range = np.linspace(S * 0.5, S * 1.5, 300)
    greek_names = ["Delta", "Gamma", "Theta", "Vega"]
    greek_colors_call = [COLORS["delta"], COLORS["gamma"], COLORS["theta"], COLORS["vega"]]
    fig = make_subplots(rows=2, cols=2, subplot_titles=greek_names,
                        vertical_spacing=0.12, horizontal_spacing=0.08)
    for idx, greek_name in enumerate(greek_names):
        row = idx // 2 + 1
        col = idx % 2 + 1
        call_vals = [greeks(s, K, T, r, sigma, "call", q)[greek_name] for s in s_range]
        put_vals = [greeks(s, K, T, r, sigma, "put", q)[greek_name] for s in s_range]
        fig.add_trace(go.Scatter(x=s_range, y=call_vals, mode="lines",
                                 name=f"Call {greek_name}",
                                 line=dict(color=greek_colors_call[idx], width=2),
                                 showlegend=(idx == 0)),
                      row=row, col=col)
        fig.add_trace(go.Scatter(x=s_range, y=put_vals, mode="lines",
                                 name=f"Put {greek_name}",
                                 line=dict(color=COLORS["put"], width=2, dash="dash"),
                                 showlegend=(idx == 0)),
                      row=row, col=col)
        fig.add_vline(x=K, line_dash="dot", line_color="rgba(46,204,113,0.2)",
                      row=row, col=col)
    fig.update_layout(paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
                      font=dict(color=COLORS["text"], size=11, family="DM Sans"),
                      height=660, margin=dict(l=40, r=20, t=60, b=60),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.12,
                                  xanchor="center", x=0.5, font=dict(family="DM Sans")),
                      title=dict(text="Greeks Sensitivity vs Underlying Price",
                                 font=dict(family="DM Sans", size=16, color=COLORS["text_bright"])))
    for i in range(1, 5):
        fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False,
                         tickfont=dict(family="JetBrains Mono", size=9),
                         row=(i-1)//2+1, col=(i-1)%2+1)
        fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False,
                         tickfont=dict(family="JetBrains Mono", size=9),
                         row=(i-1)//2+1, col=(i-1)%2+1)
    return fig


def plot_greeks_vs_time(S, K, T, r, sigma, q):
    t_range = np.linspace(0.01, T, 200)
    greek_names = ["Delta", "Gamma", "Theta", "Vega"]
    greek_colors = [COLORS["delta"], COLORS["gamma"], COLORS["theta"], COLORS["vega"]]
    fig = make_subplots(rows=2, cols=2, subplot_titles=greek_names,
                        vertical_spacing=0.12, horizontal_spacing=0.08)
    for idx, greek_name in enumerate(greek_names):
        row = idx // 2 + 1
        col = idx % 2 + 1
        call_vals = [greeks(S, K, t, r, sigma, "call", q)[greek_name] for t in t_range]
        fig.add_trace(go.Scatter(x=(t_range * 365).astype(int), y=call_vals, mode="lines",
                                 name=greek_name,
                                 line=dict(color=greek_colors[idx], width=2),
                                 showlegend=True),
                      row=row, col=col)
    fig.update_layout(paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
                      font=dict(color=COLORS["text"], size=11, family="DM Sans"),
                      height=660, margin=dict(l=40, r=20, t=60, b=60),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.12,
                                  xanchor="center", x=0.5, font=dict(family="DM Sans")),
                      title=dict(text="Greeks Decay vs Days to Expiration (Call)",
                                 font=dict(family="DM Sans", size=16, color=COLORS["text_bright"])))
    for i in range(1, 5):
        fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False, title_text="DTE",
                         tickfont=dict(family="JetBrains Mono", size=9),
                         row=(i-1)//2+1, col=(i-1)%2+1)
        fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False,
                         tickfont=dict(family="JetBrains Mono", size=9),
                         row=(i-1)//2+1, col=(i-1)%2+1)
    return fig


def plot_iv_surface(iv_data):
    fig = go.Figure(data=go.Surface(
        x=iv_data["strikes"],
        y=iv_data["expirations"],
        z=iv_data["iv_matrix"],
        colorscale=[[0, "#0d0d0d"], [0.25, "#1a1a1a"], [0.5, "#1abc9c"],
                    [0.75, "#2ecc71"], [1, "#a7f3d0"]],
        colorbar=dict(title=dict(text="IV", font=dict(size=11)),
                      thickness=10, len=0.6,
                      tickfont=dict(family="JetBrains Mono", size=9)),
    ))
    fig.update_layout(paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
                      font=dict(color=COLORS["text"], size=11, family="DM Sans"),
                      title=dict(text="Implied Volatility Surface",
                                 font=dict(family="DM Sans", size=16, color=COLORS["text_bright"])),
                      scene=dict(
                          xaxis=dict(title="Strike ($)", backgroundcolor=COLORS["bg"],
                                     gridcolor=COLORS["grid"],
                                     tickfont=dict(family="JetBrains Mono", size=9)),
                          yaxis=dict(title="DTE", backgroundcolor=COLORS["bg"],
                                     gridcolor=COLORS["grid"],
                                     tickfont=dict(family="JetBrains Mono", size=9)),
                          zaxis=dict(title="IV (%)", backgroundcolor=COLORS["bg"],
                                     gridcolor=COLORS["grid"],
                                     tickfont=dict(family="JetBrains Mono", size=9)),
                          bgcolor=COLORS["bg"]),
                      height=600, margin=dict(l=0, r=0, t=50, b=0))
    return fig


def plot_iv_heatmap(iv_data):
    strike_labels = [f"${s:.0f}" for s in iv_data["strikes"]]
    fig = go.Figure(data=go.Heatmap(
        x=strike_labels,
        y=iv_data["dte_labels"],
        z=iv_data["iv_matrix"] * 100,
        colorscale=[[0, "#0d0d0d"], [0.25, "#1a1a1a"], [0.5, "#1abc9c"],
                    [0.75, "#2ecc71"], [1, "#a7f3d0"]],
        text=np.round(iv_data["iv_matrix"] * 100, 1),
        texttemplate="%{text}%",
        textfont=dict(size=11, family="JetBrains Mono"),
        colorbar=dict(title=dict(text="IV %", font=dict(size=11)),
                      thickness=12, len=0.8,
                      tickfont=dict(family="JetBrains Mono", size=10)),
        hovertemplate="Strike: %{x}<br>Expiry: %{y}<br>IV: %{text}%<extra></extra>",
    ))
    fig.update_layout(**PLOT_LAYOUT, title="Implied Volatility Heatmap",
                      xaxis_title="Strike", yaxis_title="Expiration",
                      height=max(350, len(iv_data["dte_labels"]) * 55 + 100))
    fig.update_xaxes(tickangle=-45)
    return fig


def plot_option_price_vs_vol(S, K, T, r, q):
    vol_range = np.linspace(0.05, 1.0, 200)
    call_prices = [bs_call(S, K, T, r, v, q) for v in vol_range]
    put_prices = [bs_put(S, K, T, r, v, q) for v in vol_range]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vol_range * 100, y=call_prices, mode="lines",
                             name="Call", line=dict(color=COLORS["call"], width=2.5)))
    fig.add_trace(go.Scatter(x=vol_range * 100, y=put_prices, mode="lines",
                             name="Put", line=dict(color=COLORS["put"], width=2.5)))
    fig.update_layout(**PLOT_LAYOUT, title="Option Price vs Implied Volatility",
                      xaxis_title="Volatility (%)", yaxis_title="Option Price ($)",
                      height=420,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.28,
                                  xanchor="center", x=0.5, font=dict(family="DM Sans")))
    return fig


def plot_smile_fit(fit_data, expiry_label):
    """
    Plot market IV points + fitted quadratic smile + residuals panel for a single expiration.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08, row_heights=[0.7, 0.3],
                        subplot_titles=(f"IV Smile — {expiry_label} ({fit_data['dte']} DTE)",
                                        "Residuals (Market IV − Fitted IV)"))

    # Sort by log-moneyness for a clean fitted line
    order = np.argsort(fit_data["log_moneyness"])
    k_sorted = fit_data["log_moneyness"][order]
    iv_sorted = fit_data["iv_market"][order]
    fitted_sorted = fit_data["fitted"][order]
    residuals_sorted = fit_data["residuals"][order]

    # Smooth fitted curve (evaluate on a dense grid)
    k_dense = np.linspace(k_sorted.min(), k_sorted.max(), 200)
    a, b, c = fit_data["coeffs"]
    fitted_dense = a + b * k_dense + c * k_dense**2

    # Top panel — smile
    fig.add_trace(go.Scatter(x=k_sorted, y=iv_sorted * 100, mode="markers",
                             name="Market IV",
                             marker=dict(size=8, color=COLORS["call"],
                                         line=dict(width=1, color="white"))),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=k_dense, y=fitted_dense * 100, mode="lines",
                             name="Quadratic Fit",
                             line=dict(color=COLORS["fitted"], width=2.5)),
                  row=1, col=1)

    # Bottom panel — residuals
    resid_pct = residuals_sorted * 100
    colors_r = [COLORS["flag_over"] if r > 0 else COLORS["flag_under"] for r in resid_pct]
    fig.add_trace(go.Bar(x=k_sorted, y=resid_pct, name="Residual",
                         marker_color=colors_r,
                         showlegend=False),
                  row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(232,232,232,0.3)",
                  row=2, col=1)

    fig.update_layout(
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], size=11, family="DM Sans"),
        height=580, margin=dict(l=48, r=24, t=64, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                    xanchor="center", x=0.5, font=dict(family="DM Sans")),
    )
    fig.update_xaxes(title_text="log(K/F)", gridcolor=COLORS["grid"], zeroline=False,
                     tickfont=dict(family="JetBrains Mono", size=10), row=2, col=1)
    fig.update_yaxes(title_text="IV (%)", gridcolor=COLORS["grid"], zeroline=False,
                     tickfont=dict(family="JetBrains Mono", size=10), row=1, col=1)
    fig.update_yaxes(title_text="Resid (pp)", gridcolor=COLORS["grid"], zeroline=False,
                     tickfont=dict(family="JetBrains Mono", size=10), row=2, col=1)

    # Add R² annotation
    fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.95,
                       text=f"R² = {fit_data['r_squared']*100:.1f}% · n = {fit_data['n_fit']}",
                       showarrow=False,
                       font=dict(family="JetBrains Mono", size=10,
                                 color=COLORS["text_bright"]),
                       bgcolor="rgba(14,14,14,0.6)",
                       bordercolor="rgba(46,204,113,0.25)", borderwidth=1,
                       borderpad=4)
    return fig


def plot_mispricing_scatter(scan_df):
    """Scatter: Moneyness vs Z-score, colored by flag, sized by volume."""
    if scan_df.empty:
        return None
    df = scan_df.copy()
    df["abs_z"] = df["Z-Score"].abs()
    df["color"] = df["Z-Score"].apply(
        lambda z: COLORS["flag_over"] if z > 2 else (COLORS["flag_under"] if z < -2 else "#64748b")
    )
    # Size by volume (clamped)
    vol_clamp = np.clip(df["Volume"].fillna(1).astype(float), 1, 5000)
    df["marker_size"] = 6 + np.log1p(vol_clamp) * 2

    fig = go.Figure()
    # Non-flagged
    non_flag = df[~df["Flagged"]]
    flag = df[df["Flagged"]]

    fig.add_trace(go.Scatter(
        x=non_flag["Moneyness"], y=non_flag["Z-Score"], mode="markers",
        marker=dict(size=non_flag["marker_size"], color="rgba(100,116,139,0.45)",
                    line=dict(width=0.5, color="rgba(232,232,232,0.3)")),
        name="Within fit",
        customdata=np.stack([non_flag["Expiry"], non_flag["Strike"],
                             non_flag["Market IV"] * 100, non_flag["Fitted IV"] * 100,
                             non_flag["Mispricing %"]], axis=-1),
        hovertemplate=("Expiry: %{customdata[0]}<br>"
                       "Strike: $%{customdata[1]:.2f}<br>"
                       "Moneyness: %{x:.3f}<br>"
                       "Market IV: %{customdata[2]:.1f}%<br>"
                       "Fitted IV: %{customdata[3]:.1f}%<br>"
                       "Mispricing: %{customdata[4]:.1f}%<br>"
                       "Z-score: %{y:.2f}<extra></extra>"),
    ))
    fig.add_trace(go.Scatter(
        x=flag["Moneyness"], y=flag["Z-Score"], mode="markers",
        marker=dict(size=flag["marker_size"], color=flag["color"],
                    line=dict(width=1, color="white")),
        name=f"Flagged (|z| > 2)",
        customdata=np.stack([flag["Expiry"], flag["Strike"],
                             flag["Market IV"] * 100, flag["Fitted IV"] * 100,
                             flag["Mispricing %"]], axis=-1),
        hovertemplate=("Expiry: %{customdata[0]}<br>"
                       "Strike: $%{customdata[1]:.2f}<br>"
                       "Moneyness: %{x:.3f}<br>"
                       "Market IV: %{customdata[2]:.1f}%<br>"
                       "Fitted IV: %{customdata[3]:.1f}%<br>"
                       "Mispricing: %{customdata[4]:.1f}%<br>"
                       "Z-score: %{y:.2f}<extra></extra>"),
    ))
    fig.add_hline(y=2, line_dash="dash", line_color="rgba(239,68,68,0.35)")
    fig.add_hline(y=-2, line_dash="dash", line_color="rgba(239,68,68,0.35)")
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(232,232,232,0.2)")
    fig.add_vline(x=1.0, line_dash="dot", line_color="rgba(232,232,232,0.2)",
                  annotation_text="ATM", annotation_position="top right",
                  annotation_font=dict(size=10, color=COLORS["text"]))

    fig.update_layout(**PLOT_LAYOUT,
                      title="Mispricing Map — IV Residual Z-Scores (all expirations)",
                      xaxis_title="Moneyness (K/S)",
                      yaxis_title="Residual Z-Score (σ)", height=480,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.22,
                                  xanchor="center", x=0.5, font=dict(family="DM Sans")))
    return fig


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────

st.sidebar.markdown("## Parameters")

mode = st.sidebar.radio("Mode", ["Manual", "Market Data"], horizontal=True)

if mode == "Manual":
    st.sidebar.markdown("### Option Contract")
    S = st.sidebar.number_input("Spot Price ($)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    K = st.sidebar.number_input("Strike Price ($)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    T_days = st.sidebar.slider("Days to Expiration", min_value=1, max_value=730, value=30)
    T = T_days / 365.0
    st.sidebar.markdown("### Market Conditions")
    sigma = st.sidebar.slider("Volatility (%)", min_value=1.0, max_value=150.0, value=25.0, step=0.5) / 100.0
    r = st.sidebar.slider("Risk-Free Rate (%)", min_value=0.0, max_value=15.0, value=5.0, step=0.25) / 100.0
    q = st.sidebar.slider("Dividend Yield (%)", min_value=0.0, max_value=15.0, value=0.0, step=0.1) / 100.0
    ticker_for_iv = None
    spot_price = S
else:
    st.sidebar.markdown("### Ticker")
    ticker_input = st.sidebar.text_input("Enter ticker", value="AAPL").strip().upper()
    r = st.sidebar.slider("Risk-Free Rate (%)", min_value=0.0, max_value=15.0, value=5.0, step=0.25) / 100.0
    override_div = st.sidebar.checkbox("Override dividend yield", value=False)
    if override_div:
        q = st.sidebar.slider("Dividend Yield (%)", min_value=0.0, max_value=15.0, value=0.0, step=0.1) / 100.0
    else:
        q = None
    ticker_for_iv = ticker_input

st.sidebar.markdown("### Display Options")
show_payoff = st.sidebar.checkbox("Payoff / P&L diagram", value=True)
show_greeks_sens = st.sidebar.checkbox("Greeks sensitivity", value=True)
show_greeks_time = st.sidebar.checkbox("Greeks vs time decay", value=False)
show_price_vol = st.sidebar.checkbox("Price vs volatility", value=False)
show_iv_surface = st.sidebar.checkbox("IV Surface (market data)", value=True)

st.sidebar.markdown("### Smile & Mispricing Scanner")
show_scanner = st.sidebar.checkbox("Smile fitting & mispricing scan", value=True)
if show_scanner:
    z_threshold = st.sidebar.slider("Z-score threshold (σ)", min_value=1.0, max_value=4.0,
                                    value=2.0, step=0.25,
                                    help="Contracts whose IV residuals exceed this many standard "
                                         "deviations are flagged as potentially mispriced.")
    max_expiries_scan = st.sidebar.slider("Expirations to scan", min_value=2, max_value=10,
                                          value=6, step=1)
    strike_range_pct = st.sidebar.slider("Strike range (± % of spot)",
                                          min_value=5, max_value=30, value=15, step=1) / 100.0
else:
    z_threshold = 2.0
    max_expiries_scan = 6
    strike_range_pct = 0.15

run_button = st.sidebar.button("Calculate", use_container_width=True, type="primary")

# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="dx-header">
    <div class="dx-header-content">
        <img src="https://marinxhemollari.com/frog-logo.svg"
             alt="Derivexus" class="dx-logo"
             onerror="this.style.display='none'">
        <div class="dx-title-wrap">
            <div class="dx-title">
                <span class="dx-title-plain">Derive</span><span class="dx-title-accent">xus</span>
                <span class="dx-version">v2.0</span>
            </div>
            <div class="dx-subtitle">
                Black-Scholes-Merton pricing · Greeks visualization · Volatility smile fitting & mispricing scanner
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

if run_button or "dx_results" in st.session_state:

    if mode == "Market Data":
        with st.spinner("Fetching options data..."):
            spot_price, expirations, fetched_q = fetch_options_chain(ticker_for_iv)

        if spot_price is None or expirations is None:
            st.error(f"Could not fetch data for **{ticker_for_iv}**. Check the ticker symbol.")
            st.stop()

        S = spot_price
        if q is None:
            q = fetched_q

        selected_expiry = st.selectbox(
            "Select expiration for pricing analysis",
            options=expirations[:12],
            format_func=lambda x: f"{x}  ({(datetime.strptime(x, '%Y-%m-%d') - datetime.today()).days} DTE)",
        )
        T_days = max((datetime.strptime(selected_expiry, "%Y-%m-%d") - datetime.today()).days, 1)
        T = T_days / 365.0

        with st.spinner("Loading options chain..."):
            calls_df, puts_df = fetch_chain_for_expiry(ticker_for_iv, selected_expiry)

        if calls_df is not None and not calls_df.empty:
            K = float(calls_df.iloc[(calls_df["strike"] - S).abs().argsort().iloc[0]]["strike"])
            atm_call = calls_df[calls_df["strike"] == K].iloc[0]
            mid_price = (atm_call["bid"] + atm_call["ask"]) / 2 if atm_call["bid"] > 0 else atm_call["lastPrice"]
            sigma = implied_vol(mid_price, S, K, T, r, "call", q)
            if np.isnan(sigma):
                sigma = 0.25
        else:
            K = S
            sigma = 0.25

    # ══════════ COMPUTE ══════════
    call_price = bs_call(S, K, T, r, sigma, q)
    put_price = bs_put(S, K, T, r, sigma, q)
    call_greeks = greeks(S, K, T, r, sigma, "call", q)
    put_greeks = greeks(S, K, T, r, sigma, "put", q)
    parity_lhs = call_price - put_price
    parity_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    parity_diff = abs(parity_lhs - parity_rhs)

    st.session_state["dx_results"] = True

    # ══════════ PRICING SUMMARY ══════════
    st.markdown("## Option Pricing")
    info_cols = st.columns(5)
    info_cols[0].metric("Spot Price", f"${S:.2f}")
    info_cols[1].metric("Strike", f"${K:.2f}")
    info_cols[2].metric("DTE", f"{T_days}")
    info_cols[3].metric("Volatility", f"{sigma*100:.1f}%")
    info_cols[4].metric("Div Yield", f"{q*100:.2f}%")
    st.markdown("---")

    col_call, col_put = st.columns(2)
    with col_call:
        st.markdown('<div class="dx-label dx-call">CALL OPTION</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Price", f"${call_price:.4f}")
        c2.metric("Delta", f"{call_greeks['Delta']:.4f}")
        c3, c4 = st.columns(2)
        c3.metric("Gamma", f"{call_greeks['Gamma']:.4f}")
        c4.metric("Theta", f"{call_greeks['Theta']:.4f}")
        c5, c6 = st.columns(2)
        c5.metric("Vega", f"{call_greeks['Vega']:.4f}")
        c6.metric("Rho", f"{call_greeks['Rho']:.4f}")

    with col_put:
        st.markdown('<div class="dx-label dx-put">PUT OPTION</div>', unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        p1.metric("Price", f"${put_price:.4f}")
        p2.metric("Delta", f"{put_greeks['Delta']:.4f}")
        p3, p4 = st.columns(2)
        p3.metric("Gamma", f"{put_greeks['Gamma']:.4f}")
        p4.metric("Theta", f"{put_greeks['Theta']:.4f}")
        p5, p6 = st.columns(2)
        p5.metric("Vega", f"{put_greeks['Vega']:.4f}")
        p6.metric("Rho", f"{put_greeks['Rho']:.4f}")

    st.markdown("---")

    st.markdown("### Put-Call Parity Verification")
    parity_cols = st.columns(3)
    parity_cols[0].metric("C − P", f"${parity_lhs:.4f}")
    parity_cols[1].metric("S·e⁻ᵠᵀ − K·e⁻ʳᵀ", f"${parity_rhs:.4f}")
    parity_cols[2].metric("Deviation", f"${parity_diff:.6f}")
    if parity_diff < 0.01:
        st.markdown("> Put-call parity holds — the deviation is negligible.")
    else:
        st.markdown("> Put-call parity shows a deviation — this may indicate arbitrage opportunity or data lag.")
    st.markdown("---")

    if show_payoff:
        st.markdown("## Payoff at Expiration")
        st.markdown("> P&L per contract at expiration. Dashed line = zero (breakeven).")
        st.plotly_chart(plot_payoff(S, K, call_price, put_price, "Both"), use_container_width=True)
        st.markdown("---")

    if show_greeks_sens:
        st.markdown("## Greeks Sensitivity")
        st.markdown("> How each Greek changes as the underlying moves. Gamma/Theta/Vega peak at ATM.")
        st.plotly_chart(plot_greeks_sensitivity(S, K, T, r, sigma, q), use_container_width=True)
        st.markdown("---")

    if show_greeks_time:
        st.markdown("## Greeks vs Time Decay")
        st.markdown("> Theta accelerates near expiry ('time decay cliff'). Gamma spikes, Vega drops.")
        st.plotly_chart(plot_greeks_vs_time(S, K, T, r, sigma, q), use_container_width=True)
        st.markdown("---")

    if show_price_vol:
        st.markdown("## Price vs Volatility")
        st.markdown("> Vega measures the slope of this curve at current IV.")
        st.plotly_chart(plot_option_price_vs_vol(S, K, T, r, q), use_container_width=True)
        st.markdown("---")

    # ═══ IV SURFACE ═══
    if show_iv_surface:
        st.markdown("## Implied Volatility Surface")
        st.markdown("> Maps IV across strike and days-to-expiration. Shows smile/skew and term structure.")

        if mode == "Market Data" and ticker_for_iv:
            with st.spinner("Building IV surface from market data..."):
                try:
                    exp_subset = expirations[:min(8, len(expirations))]
                    all_strikes = set()
                    chain_data = {}
                    for exp in exp_subset:
                        c, p = fetch_chain_for_expiry(ticker_for_iv, exp)
                        if c is not None and not c.empty:
                            mask = (c["strike"] >= S * 0.85) & (c["strike"] <= S * 1.15) & (c["volume"] > 0)
                            filtered = c[mask]
                            if len(filtered) >= 3:
                                chain_data[exp] = filtered
                                all_strikes.update(filtered["strike"].tolist())

                    if len(chain_data) >= 2:
                        strike_list = sorted(all_strikes)
                        if len(strike_list) > 20:
                            step = len(strike_list) // 20
                            strike_list = strike_list[::step]
                        dte_list = []
                        iv_matrix = []
                        for exp, df in chain_data.items():
                            dte = max((datetime.strptime(exp, "%Y-%m-%d") - datetime.today()).days, 1)
                            dte_list.append(dte)
                            row = []
                            for strike in strike_list:
                                match = df[df["strike"] == strike]
                                if not match.empty:
                                    mid = (match.iloc[0]["bid"] + match.iloc[0]["ask"]) / 2
                                    if mid <= 0:
                                        mid = match.iloc[0]["lastPrice"]
                                    iv_val = implied_vol(mid, S, strike, dte / 365.0, r, "call", q)
                                    row.append(iv_val if not np.isnan(iv_val) else None)
                                else:
                                    row.append(None)
                            iv_matrix.append(row)
                        iv_arr = np.array(iv_matrix, dtype=float)
                        for i in range(iv_arr.shape[0]):
                            mask_valid = ~np.isnan(iv_arr[i])
                            if mask_valid.sum() >= 2:
                                iv_arr[i] = np.interp(range(len(strike_list)),
                                                      np.where(mask_valid)[0],
                                                      iv_arr[i][mask_valid])
                        iv_data = {
                            "strikes": np.array(strike_list),
                            "expirations": np.array(dte_list),
                            "dte_labels": [f"{d} DTE" for d in dte_list],
                            "iv_matrix": iv_arr,
                        }
                        tab_3d, tab_hm = st.tabs(["3D Surface", "Heatmap"])
                        with tab_3d:
                            st.plotly_chart(plot_iv_surface(iv_data), use_container_width=True)
                        with tab_hm:
                            st.plotly_chart(plot_iv_heatmap(iv_data), use_container_width=True)
                    else:
                        st.warning("Not enough expiration data to build IV surface.")
                except Exception as e:
                    st.warning(f"Could not build IV surface: {e}")
        else:
            st.markdown("> Synthetic IV surface — switch to **Market Data** mode for real IVs.")
            strike_range = np.linspace(K * 0.85, K * 1.15, 15)
            dte_range = np.array([7, 14, 30, 60, 90, 120, 180])
            iv_matrix = []
            for dte in dte_range:
                row = []
                for strike in strike_range:
                    moneyness = np.log(S / strike)
                    smile = sigma * (1 + 0.3 * moneyness**2 + 0.05 * np.abs(moneyness))
                    term = smile * (1 - 0.1 * np.log(dte / 30))
                    row.append(max(term, 0.05))
                iv_matrix.append(row)
            iv_data = {
                "strikes": strike_range,
                "expirations": dte_range,
                "dte_labels": [f"{d} DTE" for d in dte_range],
                "iv_matrix": np.array(iv_matrix),
            }
            tab_3d, tab_hm = st.tabs(["3D Surface", "Heatmap"])
            with tab_3d:
                st.plotly_chart(plot_iv_surface(iv_data), use_container_width=True)
            with tab_hm:
                st.plotly_chart(plot_iv_heatmap(iv_data), use_container_width=True)
        st.markdown("---")

    # ═══ SMILE FITTING & MISPRICING SCANNER (v2.0) ═══
    if show_scanner:
        st.markdown('<div class="dx-label dx-scanner">V2.0 — VOLATILITY SMILE DIAGNOSTICS</div>',
                    unsafe_allow_html=True)
        st.markdown("## Smile Fitting & Mispricing Scanner")
        st.markdown(
            "> For each expiration, the observed implied volatilities across strikes are fit to a "
            "**quadratic polynomial** in log-moneyness `k = ln(K/F)`: "
            "`σ(k) = a + b·k + c·k²`. Points where the market IV deviates from the fit by more than "
            "a chosen Z-score threshold (default 2σ) are **flagged as potentially mispriced**. "
            "For each flagged contract, a theoretical 'fair' price is computed by repricing with "
            "the *fitted* IV — the difference is the implied mispricing in dollars and percent."
        )
        st.markdown(
            "> **Interpretation:** A positive Z-score (market IV > fitted IV) usually means the "
            "market is overpricing the option relative to its local smile — a candidate for selling. "
            "A negative Z-score (market IV < fitted IV) means underpricing — a candidate for buying. "
            "Always check volume, bid-ask spread, and upcoming events before acting — the quadratic "
            "fit is a *local* smoother and will miss genuine structure like earnings-driven IV spikes."
        )

        if mode != "Market Data" or not ticker_for_iv:
            st.info("Switch to **Market Data** mode and enter a ticker to run the scanner.")
        else:
            with st.spinner("Fitting smiles and scanning chain for mispricings..."):
                scan_df, fit_by_expiry = build_mispricing_scan(
                    ticker=ticker_for_iv, S=S, r=r, q=q, expirations=expirations,
                    max_expiries=max_expiries_scan,
                    strike_range_pct=strike_range_pct,
                    zscore_threshold=z_threshold,
                    fetch_chain_fn=fetch_chain_for_expiry,
                )

            if scan_df.empty or not fit_by_expiry:
                st.warning("Could not build smile fits for this ticker. Try a more liquid ticker "
                           "(SPY, QQQ, AAPL, TSLA) or widen the strike range.")
            else:
                # Summary metrics
                total_contracts = len(scan_df)
                flagged = scan_df[scan_df["Flagged"]]
                n_flagged = len(flagged)
                n_overpriced = int((flagged["Z-Score"] > 0).sum())
                n_underpriced = int((flagged["Z-Score"] < 0).sum())
                avg_r2 = np.mean([f["r_squared"] for f in fit_by_expiry.values()])

                sm_c1, sm_c2, sm_c3, sm_c4 = st.columns(4)
                sm_c1.metric("Contracts Scanned", f"{total_contracts:,}")
                sm_c2.metric(f"Flagged (|z| > {z_threshold:.1f})", f"{n_flagged}")
                sm_c3.metric("Overpriced / Underpriced", f"{n_overpriced} / {n_underpriced}")
                sm_c4.metric("Avg Fit R²", f"{avg_r2*100:.1f}%")

                st.markdown("### Mispricing Map")
                scatter_fig = plot_mispricing_scatter(scan_df)
                if scatter_fig is not None:
                    st.plotly_chart(scatter_fig, use_container_width=True)

                st.markdown("### Per-Expiration Smile Fits")
                # Select expiration to deep-dive
                exp_options = list(fit_by_expiry.keys())
                selected_fit_exp = st.selectbox(
                    "Inspect smile fit for expiration:",
                    options=exp_options,
                    format_func=lambda x: f"{x}  ({fit_by_expiry[x]['dte']} DTE · R² {fit_by_expiry[x]['r_squared']*100:.0f}%)",
                )
                st.plotly_chart(plot_smile_fit(fit_by_expiry[selected_fit_exp], selected_fit_exp),
                                use_container_width=True)

                # Flagged contracts table
                st.markdown("### Flagged Contracts")
                if n_flagged == 0:
                    st.info(f"No contracts exceeded the |z| > {z_threshold:.1f} threshold. Try lowering the threshold.")
                else:
                    display_flagged = flagged.sort_values("Z-Score", key=abs, ascending=False).copy()
                    display_flagged["Direction"] = np.where(
                        display_flagged["Z-Score"] > 0, "OVERPRICED", "UNDERPRICED"
                    )
                    # Format
                    fmt_df = pd.DataFrame({
                        "Expiry": display_flagged["Expiry"],
                        "DTE": display_flagged["DTE"],
                        "Strike": display_flagged["Strike"].map(lambda x: f"${x:.2f}"),
                        "K/S": display_flagged["Moneyness"].map(lambda x: f"{x:.3f}"),
                        "Mkt Mid": display_flagged["Market Mid"].map(lambda x: f"${x:.2f}"),
                        "Mkt IV": display_flagged["Market IV"].map(lambda x: f"{x*100:.1f}%"),
                        "Fit IV": display_flagged["Fitted IV"].map(lambda x: f"{x*100:.1f}%"),
                        "Z": display_flagged["Z-Score"].map(lambda x: f"{x:+.2f}σ"),
                        "Fair Price": display_flagged["Fair Price (fitted)"].map(lambda x: f"${x:.2f}"),
                        "Mispricing": display_flagged["Mispricing %"].map(lambda x: f"{x:+.1f}%"),
                        "Volume": display_flagged["Volume"].map(lambda x: f"{x:,}"),
                        "Direction": display_flagged["Direction"],
                    }).reset_index(drop=True)
                    st.dataframe(fmt_df, use_container_width=True, hide_index=True)

                    st.markdown(
                        "> **Caveats:** This is an *empirical* smoothing diagnostic, not a proven "
                        "arbitrage opportunity. False positives arise when (1) the quadratic fit "
                        "misses real smile structure (jumps, stickyness), (2) the IV was computed "
                        "from stale mid prices, (3) event risk (earnings, FDA, etc.) is pricing "
                        "into a single strike in a way the fit can't see. Use volume, open interest, "
                        "and bid-ask spread as additional filters before treating any flag seriously."
                    )
        st.markdown("---")

    # ═══ GREEKS REFERENCE ═══
    st.markdown("## Greeks Reference")
    greeks_ref = pd.DataFrame({
        "Greek": ["Delta (Δ)", "Gamma (Γ)", "Theta (Θ)", "Vega (ν)", "Rho (ρ)"],
        "Measures": [
            "Rate of change of option price w.r.t. underlying",
            "Rate of change of Delta w.r.t. underlying",
            "Rate of change of option price w.r.t. time (per day)",
            "Rate of change of option price w.r.t. volatility (per 1%)",
            "Rate of change of option price w.r.t. interest rate (per 1%)",
        ],
        "Call Range": ["0 to +e⁻ᵠᵀ", "≥ 0", "≤ 0 (usually)", "≥ 0", "≥ 0"],
        "Put Range": ["−e⁻ᵠᵀ to 0", "≥ 0", "≤ 0 (usually)", "≥ 0", "≤ 0"],
    }).set_index("Greek")
    st.dataframe(greeks_ref, use_container_width=True)
    st.markdown(
        "> Delta approximates the probability of finishing in-the-money (adjusted for dividend yield q). "
        "Gamma is highest for ATM options near expiration. Theta accelerates as expiration approaches."
    )

# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div class="dx-footer">
    Built by <a href="https://marinxhemollari.com" target="_blank">Marin Xhemollari</a> ·
    Black-Scholes-Merton · Quadratic Smile Fitting · Mispricing Scanner
    <div class="dx-footer-mono">derivexus v2.0 · scipy.stats.norm · brentq solver · np.polyfit · plotly.js</div>
</div>
""", unsafe_allow_html=True)
