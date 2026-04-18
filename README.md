# Derivexus

Options pricing and Greeks visualization engine built with Python and Streamlit. Implements Black-Scholes pricing for European options with full Greeks analysis, implied volatility surface construction, and sensitivity charts.

**[Live Demo →](#)**

---

## Screenshots

![Dashboard](dashboard.png)

![Payoff & Greeks](payoff.png)

![Greeks & Time Decay](greeks.png)

![IV Surface](iv-surface.png)

![Heatmap](heatmap.png)

---

## Features

- **Black-Scholes Pricing** — European call and put pricing with closed-form solution
- **Full Greeks Suite** — Delta, Gamma, Theta, Vega, Rho for both calls and puts
- **Greeks Sensitivity** — interactive charts showing how Greeks change across underlying price
- **Greeks Time Decay** — visualize how Greeks evolve as expiration approaches
- **Payoff & P&L Diagrams** — at-expiration payoff with premium cost overlay
- **Implied Volatility Solver** — Brent's method to back out IV from market prices
- **IV Surface** — 3D surface and heatmap from live options chain data (yfinance)
- **Put-Call Parity** — automatic verification of pricing consistency
- **Price vs Volatility** — sensitivity of option price to implied volatility changes
- **Two Modes** — manual parameter input or live market data via ticker lookup
- **Greeks Reference** — built-in reference table with interpretation guide

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Visualization | Plotly |
| Pricing Model | Black-Scholes (scipy.stats.norm) |
| IV Solver | Brent's method (scipy.optimize.brentq) |
| Data | yfinance, Pandas, NumPy |
| Deployment | Streamlit Cloud |

## How It Works

### Black-Scholes Formula

For a European call option:

C = S·N(d₁) − K·e⁻ʳᵀ·N(d₂)

Where:
- d₁ = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
- d₂ = d₁ − σ·√T
- N(·) = standard normal CDF

### Greeks

| Greek | Formula | Interpretation |
|-------|---------|----------------|
| Delta (Δ) | ∂C/∂S | Price sensitivity to underlying |
| Gamma (Γ) | ∂²C/∂S² | Delta sensitivity to underlying |
| Theta (Θ) | ∂C/∂t | Time decay per day |
| Vega (ν) | ∂C/∂σ | Sensitivity to volatility |
| Rho (ρ) | ∂C/∂r | Sensitivity to interest rate |
