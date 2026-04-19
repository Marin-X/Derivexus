# Derivexus

Options pricing, Greeks visualization, and volatility surface analysis engine built with Python and Streamlit. Implements the generalized Black-Scholes-Merton framework with continuous dividend yield, computes the full Greeks suite, builds implied volatility surfaces from live market data, fits quadratic smile models per expiration, and scans options chains for contracts whose market IV deviates meaningfully from the fitted smile.

**[Live Demo →](https://derivexus-marinx.streamlit.app)**

---

## Screenshots

![Pricing](pricing.png)
![Greeks](greeks.png)
![IV Surface](iv_surface.png)
![Smile Fit](smile_fit.png)
![Mispricing Scanner](mispricing.png)

---

## Features

### Core Pricing
- **Generalized Black-Scholes-Merton pricing** with continuous dividend yield `q`
- **Full Greeks suite** — Delta, Gamma, Theta, Vega, Rho (all dividend-adjusted)
- **Put-Call Parity verification** with dividend adjustment
- **Implied volatility solver** — Brent's method for numerically backing out σ from market prices
- **Greeks sensitivity analysis** — how each Greek changes across the underlying price range
- **Greeks time decay** — how each Greek evolves as expiration approaches
- **Payoff & P&L diagrams** at expiration for calls, puts, or both

### Market Data Integration
- **Live options chains** via `yfinance` — auto-fetches spot, chain, dividend yield
- **Implied Volatility Surface** — rendered as both 3D surface and heatmap
- **ATM strike auto-detection** and IV back-out from the actual market mid

### Smile Fitting & Mispricing Scanner (v2.0)
- **Per-expiration quadratic smile fit** — `σ(k) = a + b·k + c·k²` where `k = ln(K/F)` is log-moneyness
- **Residual z-score computation** — each contract's IV deviation from the fitted smile, expressed in standard deviations
- **Configurable flag threshold** — contracts with |z| > threshold (default 2σ) are flagged as potentially mispriced
- **Fair-price reconstruction** — each flagged contract is re-priced with its *fitted* IV, and the mispricing is reported in both dollars and percent
- **Mispricing map** — scatter plot of moneyness vs. z-score across all scanned expirations, sized by volume
- **Per-expiration diagnostic** — smile fit overlaid on market IVs with a dedicated residuals panel and R² annotation
- **Direction labeling** — flagged contracts are labeled OVERPRICED or UNDERPRICED relative to the local smile

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Visualization | Plotly (2D + 3D) |
| Pricing | SciPy (norm, brentq) |
| Smile Fit | NumPy (`np.polyfit`) |
| Market Data | yfinance |
| Deployment | Streamlit Cloud |

---

## How It Works

### Generalized Black-Scholes-Merton

For a non-dividend-paying asset, Black-Scholes (1973) gives the European call price. Merton (1973) generalized this to include a continuous dividend yield `q`:

```
d₁ = [ln(S/K) + (r − q + σ²/2)·T] / (σ·√T)
d₂ = d₁ − σ·√T
```

**Call price:**
```
C = S·e^(−qT)·N(d₁) − K·e^(−rT)·N(d₂)
```

**Put price:**
```
P = K·e^(−rT)·N(−d₂) − S·e^(−qT)·N(−d₁)
```

Where `N(·)` is the cumulative standard normal distribution, `S` is spot, `K` is strike, `T` is time to expiration in years, `r` is the continuously-compounded risk-free rate, and `q` is the continuously-compounded dividend yield.

### Greeks (Dividend-Adjusted)

```
Δ_call = e^(−qT)·N(d₁)
Δ_put  = −e^(−qT)·N(−d₁)

Γ = e^(−qT)·φ(d₁) / (S·σ·√T)

ν = S·e^(−qT)·φ(d₁)·√T    (per 1.0 change in σ; divided by 100 for per-1% convention)

Θ_call = −[S·e^(−qT)·φ(d₁)·σ] / (2·√T)
         − r·K·e^(−rT)·N(d₂)
         + q·S·e^(−qT)·N(d₁)        (divided by 365 for per-day convention)

Θ_put  = −[S·e^(−qT)·φ(d₁)·σ] / (2·√T)
         + r·K·e^(−rT)·N(−d₂)
         − q·S·e^(−qT)·N(−d₁)       (divided by 365 for per-day convention)

ρ_call =  K·T·e^(−rT)·N(d₂)   (divided by 100 for per-1% convention)
ρ_put  = −K·T·e^(−rT)·N(−d₂)  (divided by 100 for per-1% convention)
```

Where `φ(·)` is the standard normal PDF.

### Put-Call Parity (with dividends)

```
C − P = S·e^(−qT) − K·e^(−rT)
```

Violations of this identity are direct arbitrage signals (modulo bid-ask spreads and borrow costs).

### Implied Volatility

Given a market price `P_mkt`, implied vol is the σ solving:

```
BS(S, K, T, r, σ, q) − P_mkt = 0
```

Solved via Brent's method on the interval `[0.001, 5.0]` with 200 max iterations.

### Quadratic Smile Fit

For each expiration, the observed IVs across strikes are fit to:

```
σ(k) = a + b·k + c·k²
```

where `k = ln(K/F)` is log-moneyness and `F = S · e^((r−q)T)` is the forward price. Coefficients `[a, b, c]` are estimated via least squares (`numpy.polyfit`, degree=2).

**Why log-moneyness?** The smile is more symmetric and better-behaved in log-moneyness than in raw strike space, and `k = 0` corresponds to the at-the-forward option — a natural centering point.

**Why quadratic?** It captures the level (`a`), slope/skew (`b`), and curvature/convexity (`c`) of the smile without overfitting on sparse chain data. This is a simpler relative of the SVI (Stochastic Volatility Inspired) parameterization used in production at dealers.

### Mispricing Scanner

For each contract in each scanned expiration:

1. Compute the market IV from the contract's mid price via Brent IV solver
2. Compute the **residual**: `ε = σ_market − σ_fitted`
3. Compute the **z-score** within the expiration: `z = ε / std(ε)`
4. Flag the contract if `|z| > threshold` (default 2σ)
5. Compute the **fair price** by re-pricing the option with its *fitted* IV:
   `P_fair = BS(S, K, T, r, σ_fitted, q)`
6. Report the **mispricing** in percent: `(P_mid − P_fair) / P_fair × 100`

A positive z-score means the market is pricing the contract's IV *above* the local smile (the option looks *overpriced*); a negative z-score means it's priced *below* (the option looks *underpriced*).

---

## Assumptions & Limitations

- **European exercise only** — American early-exercise premium is not modeled (acceptable for most index options and non-dividend stocks, less so for deep ITM puts or dividend stocks)
- **Continuous dividend yield** — real dividends are discrete; this is a smoothing approximation (market convention for index options)
- **Constant volatility over the life of the option** — violated in reality (see IV surface tab for evidence of the smile/term structure)
- **No bid-ask spread modeling** — mid prices are used for IV back-out; wide spreads inflate apparent mispricings
- **No borrow cost / repo rate modeling** — assumes the underlying can be shorted frictionlessly at `r`
- **Smile fit is empirical, not arbitrage-free** — a quadratic fit can produce negative IVs in the wings and does not enforce no-static-arbitrage constraints the way SVI does. Use only as a *diagnostic*, not a pricing model.
- **Mispricing flags are not trading signals** — false positives arise from stale quotes, event-driven IV spikes (earnings, FDA), and genuine structure the quadratic fit smooths over. Always filter by volume, open interest, and spread before treating any flag seriously.

---

## Running Locally

```bash
git clone https://github.com/Marin-X/Derivexus.git
cd Derivexus
pip install -r requirements.txt
streamlit run app.py
```

---

## References

- Black, F. and Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637–654.
- Merton, R. C. (1973). "Theory of Rational Option Pricing." *The Bell Journal of Economics and Management Science*, 4(1), 141–183.
- Hull, J. C. (2017). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley.
- Gatheral, J. and Jacquier, A. (2014). "Arbitrage-free SVI volatility surfaces." *Quantitative Finance*, 14(1), 59–71.

---

Built by [Marin Xhemollari](https://marinxhemollari.com) · [Portfolio](https://marinxhemollari.com) · [LinkedIn](https://linkedin.com/in/marinxhemollari)
