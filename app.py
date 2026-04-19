"""
Derivexus — Options Pricing & Greeks Visualization Engine
Built by Marin Xhemollari | marinxhemollari.com

Implements:
- Black-Scholes pricing for European calls and puts
- Full Greeks suite: Delta, Gamma, Theta, Vega, Rho
- Greeks sensitivity analysis across underlying price range
- Payoff & P&L diagrams at expiration
- Implied Volatility solver (Brent's method)
- IV Surface from live options chain data (yfinance)
- Greeks heatmaps across strike × expiry
- Put-Call Parity verification
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
    page_icon="🐸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# PREMIUM CSS — Charcoal / Emerald Theme
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
    --emerald-glow: rgba(46, 204, 113, 0.15);
    --emerald-glow-strong: rgba(46, 204, 113, 0.35);
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

/* ═══ HEADER ═══ */
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
.dx-title-wrap {
    display: flex;
    flex-direction: column;
    min-width: 0;
    flex: 1;
}
.dx-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3rem; font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em; line-height: 1.1; margin: 0;
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 0.75rem;
}
.dx-title-text {
    display: inline-block;
    white-space: nowrap;
}
.dx-title span.dx-gradient {
    background: linear-gradient(135deg, var(--emerald-500), var(--emerald-300));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
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
    align-self: center;
    flex-shrink: 0;
}

/* ═══ METRICS ═══ */
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

/* ═══ SIDEBAR ═══ */
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

/* ═══ HEADINGS ═══ */
h2 { font-family: 'Cormorant Garamond', serif !important;
     font-weight: 600 !important; animation: fadeInUp 0.6s ease-out; }
h3, h4 { font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; }

/* ═══ BUTTONS ═══ */
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

/* ═══ TABS ═══ */
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

/* ═══ STRATEGY LABELS ═══ */
.dx-label {
    font-family: 'DM Sans', sans-serif; font-size: 0.65rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.12em;
    padding: 0.25rem 0.65rem; border-radius: 4px;
    display: inline-block; margin-bottom: 0.5rem;
}
.dx-call { color: #22c55e; background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.2); }
.dx-put { color: #ef4444; background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2); }
.dx-greek { color: #a78bfa; background: rgba(167,139,250,0.08); border: 1px solid rgba(167,139,250,0.2); }

/* ═══ CHARTS ═══ */
[data-testid="stPlotlyChart"] {
    border: 1px solid var(--glass-border); border-radius: 12px;
    overflow: hidden; animation: fadeIn 0.5s ease-out;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
[data-testid="stPlotlyChart"]:hover {
    border-color: var(--glass-border-hover);
    box-shadow: 0 4px 24px rgba(46, 204, 113, 0.06);
}

/* ═══ DIVIDERS ═══ */
hr {
    border: none !important; height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(46,204,113,0.15), transparent) !important;
    margin: 2rem 0 !important;
}

/* ═══ BLOCKQUOTE ═══ */
blockquote {
    border-left: 3px solid var(--emerald-500) !important;
    background: rgba(46,204,113,0.03) !important;
    padding: 0.8rem 1.2rem !important; border-radius: 0 8px 8px 0 !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.88rem !important;
    color: var(--text-secondary) !important;
}

/* ═══ DATAFRAMES ═══ */
[data-testid="stDataFrame"] {
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important; overflow: hidden;
    animation: fadeIn 0.5s ease-out;
}

/* ═══ FOOTER ═══ */
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

/* ═══ MULTISELECT ═══ */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: rgba(46,204,113,0.1) !important;
    border: 1px solid rgba(46,204,113,0.2) !important;
    color: var(--emerald-500) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important; border-radius: 6px !important;
}

/* ═══ MOBILE ═══ */
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
# BLACK-SCHOLES MATH
# ──────────────────────────────────────────────────────────────

def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    return S * norm.cdf(d1(S, K, T, r, sigma)) - K * np.exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma))

def bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    return K * np.exp(-r * T) * norm.cdf(-d2(S, K, T, r, sigma)) - S * norm.cdf(-d1(S, K, T, r, sigma))


# ──────────────────────────────────────────────────────────────
# GREEKS
# ──────────────────────────────────────────────────────────────

def greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return {"Delta": 0, "Gamma": 0, "Theta": 0, "Vega": 0, "Rho": 0}

    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)

    gamma = norm.pdf(d1_val) / (S * sigma * sqrt_T)
    vega = S * norm.pdf(d1_val) * sqrt_T / 100  # per 1% move

    if option_type == "call":
        delta = norm.cdf(d1_val)
        theta = (-(S * norm.pdf(d1_val) * sigma) / (2 * sqrt_T)
                 - r * K * np.exp(-r * T) * norm.cdf(d2_val)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2_val) / 100
    else:
        delta = norm.cdf(d1_val) - 1
        theta = (-(S * norm.pdf(d1_val) * sigma) / (2 * sqrt_T)
                 + r * K * np.exp(-r * T) * norm.cdf(-d2_val)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2_val) / 100

    return {
        "Delta": round(delta, 4),
        "Gamma": round(gamma, 4),
        "Theta": round(theta, 4),
        "Vega": round(vega, 4),
        "Rho": round(rho, 4),
    }


# ──────────────────────────────────────────────────────────────
# IMPLIED VOLATILITY SOLVER
# ──────────────────────────────────────────────────────────────

def implied_vol(market_price, S, K, T, r, option_type="call"):
    if T <= 0:
        return np.nan
    try:
        func = bs_call if option_type == "call" else bs_put
        iv = brentq(lambda sigma: func(S, K, T, r, sigma) - market_price, 0.001, 5.0, maxiter=200)
        return iv
    except (ValueError, RuntimeError):
        return np.nan


# ──────────────────────────────────────────────────────────────
# DATA FETCHING
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_options_chain(ticker):
    try:
        tk = yf.Ticker(ticker)
        spot = tk.info.get("regularMarketPrice") or tk.info.get("currentPrice")
        if spot is None:
            hist = tk.history(period="1d")
            spot = float(hist["Close"].iloc[-1]) if not hist.empty else None
        expirations = list(tk.options)
        if not expirations or spot is None:
            return None, None
        return float(spot), expirations
    except Exception:
        return None, None


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
# PLOTTING FUNCTIONS
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


def plot_greeks_sensitivity(S, K, T, r, sigma):
    s_range = np.linspace(S * 0.5, S * 1.5, 300)

    greek_names = ["Delta", "Gamma", "Theta", "Vega"]
    greek_colors_call = [COLORS["delta"], COLORS["gamma"], COLORS["theta"], COLORS["vega"]]

    fig = make_subplots(rows=2, cols=2, subplot_titles=greek_names,
                        vertical_spacing=0.12, horizontal_spacing=0.08)

    for idx, greek_name in enumerate(greek_names):
        row = idx // 2 + 1
        col = idx % 2 + 1

        call_vals = [greeks(s, K, T, r, sigma, "call")[greek_name] for s in s_range]
        put_vals = [greeks(s, K, T, r, sigma, "put")[greek_name] for s in s_range]

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

    fig.update_layout(
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], size=11, family="DM Sans"),
        height=660, margin=dict(l=40, r=20, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12,
                    xanchor="center", x=0.5, font=dict(family="DM Sans")),
        title=dict(text="Greeks Sensitivity vs Underlying Price",
                   font=dict(family="DM Sans", size=16, color=COLORS["text_bright"])),
    )

    for i in range(1, 5):
        fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False,
                         tickfont=dict(family="JetBrains Mono", size=9), row=(i-1)//2+1, col=(i-1)%2+1)
        fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False,
                         tickfont=dict(family="JetBrains Mono", size=9), row=(i-1)//2+1, col=(i-1)%2+1)

    return fig


def plot_greeks_vs_time(S, K, T, r, sigma):
    t_range = np.linspace(0.01, T, 200)
    greek_names = ["Delta", "Gamma", "Theta", "Vega"]
    greek_colors = [COLORS["delta"], COLORS["gamma"], COLORS["theta"], COLORS["vega"]]

    fig = make_subplots(rows=2, cols=2, subplot_titles=greek_names,
                        vertical_spacing=0.12, horizontal_spacing=0.08)

    for idx, greek_name in enumerate(greek_names):
        row = idx // 2 + 1
        col = idx % 2 + 1

        call_vals = [greeks(S, K, t, r, sigma, "call")[greek_name] for t in t_range]

        fig.add_trace(go.Scatter(x=(t_range * 365).astype(int), y=call_vals, mode="lines",
                                 name=greek_name,
                                 line=dict(color=greek_colors[idx], width=2),
                                 showlegend=True),
                      row=row, col=col)

    fig.update_layout(
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], size=11, family="DM Sans"),
        height=660, margin=dict(l=40, r=20, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12,
                    xanchor="center", x=0.5, font=dict(family="DM Sans")),
        title=dict(text="Greeks Decay vs Days to Expiration (Call)",
                   font=dict(family="DM Sans", size=16, color=COLORS["text_bright"])),
    )

    for i in range(1, 5):
        fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False, title_text="DTE",
                         tickfont=dict(family="JetBrains Mono", size=9), row=(i-1)//2+1, col=(i-1)%2+1)
        fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False,
                         tickfont=dict(family="JetBrains Mono", size=9), row=(i-1)//2+1, col=(i-1)%2+1)

    return fig


def plot_iv_surface(iv_data):
    fig = go.Figure(data=go.Surface(
        x=iv_data["strikes"],
        y=iv_data["expirations"],
        z=iv_data["iv_matrix"],
        colorscale=[[0, "#0d0d0d"], [0.25, "#1a1a1a"], [0.5, "#1abc9c"], [0.75, "#2ecc71"], [1, "#a7f3d0"]],
        colorbar=dict(title=dict(text="IV", font=dict(size=11)),
                      thickness=10, len=0.6,
                      tickfont=dict(family="JetBrains Mono", size=9)),
    ))

    fig.update_layout(
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], size=11, family="DM Sans"),
        title=dict(text="Implied Volatility Surface",
                   font=dict(family="DM Sans", size=16, color=COLORS["text_bright"])),
        scene=dict(
            xaxis=dict(title="Strike ($)", backgroundcolor=COLORS["bg"],
                       gridcolor=COLORS["grid"], tickfont=dict(family="JetBrains Mono", size=9)),
            yaxis=dict(title="DTE", backgroundcolor=COLORS["bg"],
                       gridcolor=COLORS["grid"], tickfont=dict(family="JetBrains Mono", size=9)),
            zaxis=dict(title="IV (%)", backgroundcolor=COLORS["bg"],
                       gridcolor=COLORS["grid"], tickfont=dict(family="JetBrains Mono", size=9)),
            bgcolor=COLORS["bg"],
        ),
        height=600, margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def plot_iv_heatmap(iv_data):
    # Format strike labels cleanly
    strike_labels = [f"${s:.0f}" for s in iv_data["strikes"]]

    fig = go.Figure(data=go.Heatmap(
        x=strike_labels,
        y=iv_data["dte_labels"],
        z=iv_data["iv_matrix"] * 100,
        colorscale=[[0, "#0d0d0d"], [0.25, "#1a1a1a"], [0.5, "#1abc9c"], [0.75, "#2ecc71"], [1, "#a7f3d0"]],
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


def plot_option_price_vs_vol(S, K, T, r):
    vol_range = np.linspace(0.05, 1.0, 200)
    call_prices = [bs_call(S, K, T, r, v) for v in vol_range]
    put_prices = [bs_put(S, K, T, r, v) for v in vol_range]

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

    ticker_for_iv = None
    spot_price = S

else:
    st.sidebar.markdown("### Ticker")
    ticker_input = st.sidebar.text_input("Enter ticker", value="AAPL").strip().upper()
    r = st.sidebar.slider("Risk-Free Rate (%)", min_value=0.0, max_value=15.0, value=5.0, step=0.25) / 100.0
    ticker_for_iv = ticker_input

st.sidebar.markdown("### Display Options")
show_payoff = st.sidebar.checkbox("Payoff / P&L diagram", value=True)
show_greeks_sens = st.sidebar.checkbox("Greeks sensitivity", value=True)
show_greeks_time = st.sidebar.checkbox("Greeks vs time decay", value=False)
show_price_vol = st.sidebar.checkbox("Price vs volatility", value=False)
show_iv_surface = st.sidebar.checkbox("IV Surface (market data)", value=True)

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
                <span class="dx-title-text">Derive<span class="dx-gradient">xus</span></span>
                <span class="dx-version">v1.0</span>
            </div>
            <div class="dx-subtitle">
                Black-Scholes pricing · Greeks visualization · Implied volatility surface
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
            spot_price, expirations = fetch_options_chain(ticker_for_iv)

        if spot_price is None or expirations is None:
            st.error(f"Could not fetch data for **{ticker_for_iv}**. Check the ticker symbol.")
            st.stop()

        S = spot_price
        # Let user pick an expiration for detailed analysis
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
            # Find ATM strike
            K = float(calls_df.iloc[(calls_df["strike"] - S).abs().argsort().iloc[0]]["strike"])

            # Get mid price for IV estimation
            atm_call = calls_df[calls_df["strike"] == K].iloc[0]
            mid_price = (atm_call["bid"] + atm_call["ask"]) / 2 if atm_call["bid"] > 0 else atm_call["lastPrice"]
            sigma = implied_vol(mid_price, S, K, T, r, "call")
            if np.isnan(sigma):
                sigma = 0.25
        else:
            K = S
            sigma = 0.25

    # ══════════ COMPUTE ══════════
    call_price = bs_call(S, K, T, r, sigma)
    put_price = bs_put(S, K, T, r, sigma)
    call_greeks = greeks(S, K, T, r, sigma, "call")
    put_greeks = greeks(S, K, T, r, sigma, "put")

    # Put-Call Parity check
    parity_lhs = call_price - put_price
    parity_rhs = S - K * np.exp(-r * T)
    parity_diff = abs(parity_lhs - parity_rhs)

    st.session_state["dx_results"] = True

    # ══════════ RESULTS ══════════

    # ─── Pricing Summary ───
    st.markdown("## Option Pricing")

    info_cols = st.columns(4)
    info_cols[0].metric("Spot Price", f"${S:.2f}")
    info_cols[1].metric("Strike", f"${K:.2f}")
    info_cols[2].metric("DTE", f"{T_days}")
    info_cols[3].metric("Volatility", f"{sigma*100:.1f}%")

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

    # ─── Put-Call Parity ───
    st.markdown("### Put-Call Parity Verification")
    parity_cols = st.columns(3)
    parity_cols[0].metric("C − P", f"${parity_lhs:.4f}")
    parity_cols[1].metric("S − K·e⁻ʳᵀ", f"${parity_rhs:.4f}")
    parity_cols[2].metric("Deviation", f"${parity_diff:.6f}")

    if parity_diff < 0.01:
        st.markdown("> Put-call parity holds — the deviation is negligible, confirming pricing consistency.")
    else:
        st.markdown("> Put-call parity shows a deviation — this may indicate arbitrage opportunity or data lag.")

    st.markdown("---")

    # ─── Payoff Diagram ───
    if show_payoff:
        st.markdown("## Payoff at Expiration")
        st.markdown("> Shows profit or loss per contract at expiration across a range of underlying prices. "
                    "The **call** profits when the stock finishes above the strike + premium paid; "
                    "the **put** profits below the strike − premium paid. The dashed line at zero marks breakeven.")
        st.plotly_chart(plot_payoff(S, K, call_price, put_price, "Both"), use_container_width=True)
        st.markdown("---")

    # ─── Greeks Sensitivity ───
    if show_greeks_sens:
        st.markdown("## Greeks Sensitivity")
        st.markdown("> How each Greek changes as the underlying price moves. "
                    "**Delta** shows directional exposure — calls go from 0 to 1, puts from −1 to 0. "
                    "**Gamma** peaks at the strike (ATM), meaning Delta changes fastest there. "
                    "**Theta** shows daily time decay — deepest at ATM near expiration. "
                    "**Vega** shows volatility sensitivity — also highest ATM.")
        st.plotly_chart(plot_greeks_sensitivity(S, K, T, r, sigma), use_container_width=True)
        st.markdown("---")

    # ─── Greeks vs Time ───
    if show_greeks_time:
        st.markdown("## Greeks vs Time Decay")
        st.markdown("> How each Greek evolves as expiration approaches (DTE = days to expiration). "
                    "**Theta** accelerates sharply in the final days — the \"time decay cliff.\" "
                    "**Gamma** spikes near expiry for ATM options, making Delta highly unstable. "
                    "**Vega** drops as expiration nears — short-dated options are less sensitive to volatility shifts.")
        st.plotly_chart(plot_greeks_vs_time(S, K, T, r, sigma), use_container_width=True)
        st.markdown("---")

    # ─── Price vs Vol ───
    if show_price_vol:
        st.markdown("## Price vs Volatility")
        st.markdown("> How option price changes with implied volatility (IV), holding all else constant. "
                    "Both calls and puts increase in value as IV rises — this is what Vega measures. "
                    "The relationship is roughly linear for ATM options and curves for deep ITM/OTM.")
        st.plotly_chart(plot_option_price_vs_vol(S, K, T, r), use_container_width=True)
        st.markdown("---")

    # ─── IV Surface ───
    if show_iv_surface:
        st.markdown("## Implied Volatility Surface")
        st.markdown("> Maps implied volatility across **strike prices** (x-axis) and **DTE — days to expiration** (y-axis). "
                    "The \"smile\" shape shows higher IV for deep OTM/ITM options. "
                    "The term structure shows how IV varies by expiration — near-term options often have elevated IV around events like earnings.")

        if mode == "Market Data" and ticker_for_iv:
            with st.spinner("Building IV surface from market data..."):
                try:
                    # Use up to 6 expirations
                    exp_subset = expirations[:min(8, len(expirations))]
                    all_strikes = set()
                    chain_data = {}

                    for exp in exp_subset:
                        c, p = fetch_chain_for_expiry(ticker_for_iv, exp)
                        if c is not None and not c.empty:
                            # Filter strikes within reasonable range of spot
                            mask = (c["strike"] >= S * 0.85) & (c["strike"] <= S * 1.15) & (c["volume"] > 0)
                            filtered = c[mask]
                            if len(filtered) >= 3:
                                chain_data[exp] = filtered
                                all_strikes.update(filtered["strike"].tolist())

                    if len(chain_data) >= 2:
                        strike_list = sorted(all_strikes)

                        # Thin strikes to max ~20 for readable heatmap
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
                                    iv_val = implied_vol(mid, S, strike, dte / 365.0, r, "call")
                                    row.append(iv_val if not np.isnan(iv_val) else None)
                                else:
                                    row.append(None)
                            iv_matrix.append(row)

                        # Convert to numpy, interpolate NaNs
                        iv_arr = np.array(iv_matrix, dtype=float)
                        # Simple forward fill for missing values
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
                        st.warning("Not enough expiration data to build IV surface. Try a more liquid ticker.")

                except Exception as e:
                    st.warning(f"Could not build IV surface: {e}")

        else:
            # Synthetic IV surface for manual mode
            st.markdown("> Synthetic IV surface — switch to **Market Data** mode for real implied volatilities.")
            strike_range = np.linspace(K * 0.85, K * 1.15, 15)
            dte_range = np.array([7, 14, 30, 60, 90, 120, 180])

            iv_matrix = []
            for dte in dte_range:
                row = []
                for strike in strike_range:
                    moneyness = np.log(S / strike)
                    # Synthetic smile: higher IV for OTM options
                    smile = sigma * (1 + 0.3 * moneyness**2 + 0.05 * np.abs(moneyness))
                    # Term structure: slightly lower IV for longer DTE
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

    # ─── Greeks Reference ───
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
        "Call Range": ["0 to +1", "≥ 0", "≤ 0 (usually)", "≥ 0", "≥ 0"],
        "Put Range": ["-1 to 0", "≥ 0", "≤ 0 (usually)", "≥ 0", "≤ 0"],
    }).set_index("Greek")

    st.dataframe(greeks_ref, use_container_width=True)

    st.markdown(
        "> Delta approximates the probability of finishing in-the-money. "
        "Gamma is highest for ATM options near expiration. "
        "Theta accelerates as expiration approaches — the \"time decay cliff.\""
    )

# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div class="dx-footer">
    Built by <a href="https://marinxhemollari.com" target="_blank">Marin Xhemollari</a> ·
    Black-Scholes Pricing Model ·
    Options data via Yahoo Finance
    <div class="dx-footer-mono">derivexus v1.0 · scipy.stats.norm · brentq solver · plotly.js</div>
</div>
""", unsafe_allow_html=True)
