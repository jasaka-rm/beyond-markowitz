# Portfolio Optimization Beyond Markowitz

Empirical comparison of **Mean-Variance Optimization (MVO)**, **Risk Parity (RP)**, and **Hierarchical Risk Parity (HRP)** across two asset universes, combining momentum and trend-following signals.

Developed as part of a Master's Thesis in Finance.

---

## Structure

```
├── etf_portfolio.py       # Backtest on ETF universe (SPY, EFA, EEM, TLT, GLD, HYG)
├── crypto_portfolio.py    # Backtest on crypto universe (BTC, ETH, LTC, XRP, DOGE)
├── plots/
│   ├── etf_returns.png
│   ├── etf_drawdowns.png
│   ├── etf_weights.png
│   ├── crypto_returns.png
│   ├── crypto_drawdowns.png
│   └── crypto_weights.png
└── README.md
```

---

## Methodology

Each script runs a **rolling monthly backtest** and compares three portfolio construction methods:

| Method | Description |
|--------|-------------|
| MVO | Markowitz mean-variance optimization, tilted by a combined momentum + trend signal |
| RP | Inverse-volatility risk parity — no return forecasts required |
| HRP | Hierarchical Risk Parity (López de Prado, 2016) — tree-based allocation |

**Signal layer (MVO only):** 50% 6-month momentum rank + 50% price-vs-MA trend indicator.

---

## Setup

```bash
pip install yfinance pandas numpy matplotlib scipy pypfopt
python etf_portfolio.py
python crypto_portfolio.py
```

---

## Key Parameters

| Parameter | ETF | Crypto |
|-----------|-----|--------|
| Tickers | SPY, EFA, EEM, TLT, GLD, HYG | BTC-USD, ETH-USD, LTC-USD, XRP-USD, DOGE-USD |
| Start date | 2005-01-01 | 2015-09-01 |
| Rolling window | 36 months | 12 months |
| Rebalancing | Monthly | Monthly |
| Risk-free rate | 0% | 0% |

---

## Results Summary

### ETF Universe

| Method | Ann. Return | Sharpe | Max Drawdown | Avg Turnover |
|--------|-------------|--------|--------------|--------------|
| MVO | 11.3% | 1.08 | −22.2% | 0.285 |
| RP | 7.9% | 0.88 | −21.5% | 0.017 |
| HRP | 7.4% | 0.90 | −21.4% | 0.086 |

### Crypto Universe

| Method | Ann. Return | Sharpe | Max Drawdown | Calmar | Avg Turnover |
|--------|-------------|--------|--------------|--------|--------------|
| MVO | 61.7% | 0.798 | −77.6% | 0.795 | 0.607 |
| RP | 89.7% | 0.810 | −69.9% | 1.284 | 0.073 |
| HRP | 105.5% | 0.731 | −69.6% | 1.516 | 0.203 |

---

## References

- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*.
- López de Prado, M. (2016). Building Diversified Portfolios that Outperform Out-of-Sample. *Journal of Portfolio Management*.
- Ledoit, O. & Wolf, M. (2004). Honey, I Shrunk the Sample Covariance Matrix. *Journal of Portfolio Management*.
