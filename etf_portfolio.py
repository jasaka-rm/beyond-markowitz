import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from pypfopt.hierarchical_portfolio import HRPOpt
from scipy.optimize import minimize


# -----------------------------
# 1. DATA SETTINGS
# -----------------------------
TICKERS = ["SPY", "EFA", "EEM", "TLT", "GLD", "HYG"]
START_DATE = "2005-01-01"
END_DATE = None  # use latest available data

ROLLING_WINDOW_MONTHS = 36
REBALANCE_FREQ = "M"   # monthly
RISK_FREE_RATE = 0.0   # we assume zero


# -----------------------------
# 2. DOWNLOAD DATA
# -----------------------------
def download_prices(tickers, start, end=None):
    prices = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    prices = prices.dropna()
    return prices


# -----------------------------
# 3. CALCULATE MONTHLY PRICES   
# -----------------------------
def to_monthly_prices(prices):
    monthly_prices = prices.resample("ME").last() # Converts data to monthly frequency (Month End) and takes the last price of each month
    return monthly_prices.dropna()


def compute_returns(prices):
    return prices.pct_change().dropna()


# -----------------------------
# 4. SIGNAL CONSTRUCTION
# -----------------------------
def compute_momentum_signal(monthly_prices, lookback=12):
    # "How much has this asset gone up or down in the last 12 months"
    # This is a simple momentum signal: the percentage change over the last 12 months for each date
    return monthly_prices.pct_change(lookback)


def compute_trend_signal(monthly_prices, ma_window=12):
    # Simple trend-following signal: 1 if the price is above its 12-month moving average, 0 otherwise
    moving_avg = monthly_prices.rolling(ma_window).mean()
    return (monthly_prices > moving_avg).astype(float)


def combine_signals(momentum, trend):
    # Simple combination: positive momentum + trend confirmation
    mom_rank = momentum.rank(axis=1, pct=True)
    combined = 0.5 * mom_rank + 0.5 * trend
    return combined


# -----------------------------
# 5. WEIGHT FUNCTIONS
# -----------------------------
def normalize_weights(weights):
    weights = pd.Series(weights)
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if weights.sum() == 0:
        weights[:] = 1.0 / len(weights)
    else:
        weights = weights / weights.sum()
    return weights


def mvo_weights(train_returns, signal_tilt=None):
    mu = train_returns.mean()
    cov = train_returns.cov()

    # Optional signal tilt
    if signal_tilt is not None:
        mu = mu * signal_tilt.reindex(mu.index).fillna(1.0)

    try:
        raw = np.linalg.pinv(cov.values) @ mu.values
        weights = pd.Series(raw, index=train_returns.columns)
    except Exception:
        weights = pd.Series(1.0, index=train_returns.columns)

    # Long-only simplification
    weights[weights < 0] = 0 # zeroing out any negative weights before normalization.
    return normalize_weights(weights)


def risk_parity_weights(train_returns, vol_floor=1e-4):
    vol = train_returns.std()
    vol = vol.clip(lower=vol_floor)   # prevents extreme inv_vol (e.g. if an asset has near-zero volatility, we don't want to assign it an outsized weight)
    inv_vol = 1 / vol
    return normalize_weights(inv_vol)

# def risk_parity_weights_true(train_returns):
#     cov = train_returns.cov().values
#     n = cov.shape[0]

#     def risk_contributions(w):
#         """Compute each asset's risk contribution"""
#         portfolio_vol = np.sqrt(w @ cov @ w)
#         marginal_rc = cov @ w                    # marginal risk contribution
#         rc = w * marginal_rc / portfolio_vol     # absolute risk contribution
#         return rc

#     def objective(w):
#         """Minimize variance of risk contributions (want them all equal)"""
#         rc = risk_contributions(w)
#         # Penalize differences between all pairs of risk contributions
#         return sum((rc[i] - rc[j])**2 
#                    for i in range(n) 
#                    for j in range(i+1, n))

#     # Constraints and bounds
#     constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]  # sum = 1
#     bounds = [(0.01, 1.0)] * n                                       # long-only + min weight
#     w0 = np.ones(n) / n                                              # start from equal weights

#     try:
#         result = minimize(
#             objective,
#             w0,
#             method="SLSQP",
#             bounds=bounds,
#             constraints=constraints,
#             options={"ftol": 1e-12, "maxiter": 1000}
#         )
#         if result.success:
#             weights = pd.Series(result.x, index=train_returns.columns)
#         else:
#             raise ValueError("Optimizer did not converge")

#     except Exception:
#         # Fallback to inverse vol if optimizer fails
#         vol = train_returns.std()
#         inv_vol = 1 / vol.replace(0, np.nan).fillna(0.0)
#         weights = inv_vol

#     return normalize_weights(weights)


def hrp_weights(train_returns):
# Step 1 — TREE CLUSTERING: Builds a dendrogram via hierarchical clustering on the distance matrix.
# Step 2 — QUASI-DIAGONALIZATION: Reorders the covariance matrix so correlated assets sit next to each other.
# Step 3 — RECURSIVE BISECTION
# Splits the portfolio top-down, allocating risk at each branch of the tree.
    try:
        hrp = HRPOpt(train_returns)
        weights_dict = hrp.optimize()
        weights = pd.Series(weights_dict).reindex(train_returns.columns).fillna(0.0)
    except Exception:
        weights = pd.Series(1.0 / train_returns.shape[1], index=train_returns.columns)
    return normalize_weights(weights)


# -----------------------------
# 6. ROLLING BACKTEST
# -----------------------------
def run_backtest(monthly_returns, combined_signal, window=36):
    dates = monthly_returns.index

    strategy_returns = {
        "MVO": [],
        "RP": [],
        "HRP": []
    }

    weights_history = {
        "MVO": [],
        "RP": [],
        "HRP": []
    }

    rebalance_dates = []

    for t in range(window, len(monthly_returns) - 1):
        train = monthly_returns.iloc[t - window:t] # rolling window of 36 months to estimate cov and momentum
        next_ret = monthly_returns.iloc[t + 1] # The out-of-sample return, what actually happened the month after weights were computed

        signal_t = combined_signal.iloc[t].reindex(train.columns) # Takes the signal at the current time t — this is what we will use to tilt the MVO weights. We reindex to ensure it matches the order of assets in the training data.

        w_mvo = mvo_weights(train, signal_tilt=signal_t)
        w_rp = risk_parity_weights(train)
        w_hrp = hrp_weights(train)

        r_mvo = np.dot(w_mvo, next_ret) # dot product, weighted sum of individual asset returns
        r_rp = np.dot(w_rp, next_ret)
        r_hrp = np.dot(w_hrp, next_ret)

        strategy_returns["MVO"].append(r_mvo)
        strategy_returns["RP"].append(r_rp)
        strategy_returns["HRP"].append(r_hrp)

        weights_history["MVO"].append(w_mvo)
        weights_history["RP"].append(w_rp)
        weights_history["HRP"].append(w_hrp)

        rebalance_dates.append(monthly_returns.index[t + 1])

    returns_df = pd.DataFrame(strategy_returns, index=rebalance_dates)

    weights_df = {}
    for method in weights_history:
        weights_df[method] = pd.DataFrame(weights_history[method], index=rebalance_dates)

    return returns_df, weights_df


# -----------------------------
# 7. METRICS
# -----------------------------
def max_drawdown(returns):
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1
    return dd.min()


def sharpe_ratio(returns, rf=0.0):
    excess = returns - rf / 12
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(12)


def turnover(weights_df):
    changes = weights_df.diff().abs().sum(axis=1)
    return changes.mean()


def weight_stability(weights_df):
    # lower std of weights through time = more stable
    return weights_df.std().mean()


def summary_table(returns_df, weights_dict):
    rows = []
    for method in returns_df.columns:
        rows.append({
            "Method": method,
            "Annualized Return": returns_df[method].mean() * 12,
            "Annualized Volatility": returns_df[method].std() * np.sqrt(12),
            "Sharpe Ratio": sharpe_ratio(returns_df[method]),
            "Max Drawdown": max_drawdown(returns_df[method]),
            "Average Turnover": turnover(weights_dict[method]),
            "Weight Stability": weight_stability(weights_dict[method]),
        })
    return pd.DataFrame(rows).set_index("Method")


# -----------------------------
# 8. PLOTS
# -----------------------------
def plot_cumulative_returns(returns_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    cumulative = (1 + returns_df).cumprod()
    cumulative.plot(ax=ax)
    ax.set_title("Cumulative Portfolio Returns")
    ax.set_ylabel("Growth of 1€")
    ax.grid(True)
    plt.show(block=False)


def plot_drawdowns(returns_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    dd_df = pd.DataFrame(index=returns_df.index)
    for col in returns_df.columns:
        wealth = (1 + returns_df[col]).cumprod()
        dd_df[col] = wealth / wealth.cummax() - 1
    dd_df.plot(ax=ax)
    ax.set_title("Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.grid(True)
    plt.show(block=False)  

def plot_weights_over_time(weights_dict):
    methods = list(weights_dict.keys())

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)  # wider for outside legend

    for i, method in enumerate(methods):
        weights = weights_dict[method]
        weights.plot(ax=axes[i])
        axes[i].set_title(f"{method} Weights Over Time")
        axes[i].set_ylabel("Weight")
        axes[i].grid(True)
        axes[i].legend(            # ← move inside the loop, applies to ALL subplots
            title="Assets",
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            borderaxespad=0
        )

    axes[-1].set_xlabel("Date")

    plt.tight_layout()             # respects the outside legends automatically
    plt.show()

# -----------------------------
# 9. MAIN
# -----------------------------
if __name__ == "__main__":
    import time

    print("Step 1: Downloading prices...")
    t0 = time.time()
    prices_daily = download_prices(TICKERS, START_DATE, END_DATE)
    print(f"  ✓ Done in {time.time()-t0:.1f}s — shape: {prices_daily.shape}")

    print("Step 2: Converting to monthly...")
    t0 = time.time()
    prices_monthly = to_monthly_prices(prices_daily)
    monthly_returns = compute_returns(prices_monthly)
    print(f"  ✓ Done in {time.time()-t0:.1f}s — shape: {monthly_returns.shape}")

    print("Step 3: Computing signals...")
    t0 = time.time()
    momentum = compute_momentum_signal(prices_monthly, lookback=12)
    trend = compute_trend_signal(prices_monthly, ma_window=12)
    combined_signal = combine_signals(momentum, trend)
    combined_signal = combined_signal.reindex(monthly_returns.index)
    print(f"  ✓ Done in {time.time()-t0:.1f}s")

    print("Step 4: Running backtest...")
    t0 = time.time()
    returns_df, weights_dict = run_backtest(
        monthly_returns=monthly_returns,
        combined_signal=combined_signal,
        window=ROLLING_WINDOW_MONTHS
    )
    print(f"  ✓ Done in {time.time()-t0:.1f}s — {len(returns_df)} periods")

    print("Step 5: Computing metrics...")
    results = summary_table(returns_df, weights_dict)
    print(results.round(4))

    print("Step 6: Plotting...")
    plot_cumulative_returns(returns_df)
    plot_drawdowns(returns_df)
    plot_weights_over_time(weights_dict)
    print("  ✓ All done")
    
    plt.show(block=True)