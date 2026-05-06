import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from pypfopt.hierarchical_portfolio import HRPOpt


# -----------------------------
# 1. DATA
# -----------------------------
def download_prices(tickers, start, end=None):
    prices = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    prices = prices.dropna()
    return prices


def to_monthly_prices(prices):
    monthly_prices = prices.resample("ME").last()
    return monthly_prices.dropna()


def compute_returns(prices):
    return prices.pct_change().dropna()


# -----------------------------
# 2. SIGNAL CONSTRUCTION
# -----------------------------
def compute_momentum_signal(prices, lookback):
    return prices.pct_change(lookback)


def compute_trend_signal(prices, ma_window):
    moving_avg = prices.rolling(ma_window).mean()
    return (prices > moving_avg).astype(float)


def combine_signals(momentum, trend):
    mom_rank = momentum.rank(axis=1, pct=True)
    combined = 0.5 * mom_rank + 0.5 * trend
    return combined


# -----------------------------
# 3. WEIGHT FUNCTIONS
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

    if signal_tilt is not None:
        mu = mu * signal_tilt.reindex(mu.index).fillna(1.0)

    try:
        raw = np.linalg.pinv(cov.values) @ mu.values
        weights = pd.Series(raw, index=train_returns.columns)
    except Exception:
        weights = pd.Series(1.0, index=train_returns.columns)

    weights[weights < 0] = 0
    return normalize_weights(weights)


def risk_parity_weights(train_returns, vol_floor=1e-4):
    vol = train_returns.std()
    vol = vol.clip(lower=vol_floor)

    inv_vol = 1 / vol
    return normalize_weights(inv_vol)


def hrp_weights(train_returns):
    try:
        hrp = HRPOpt(train_returns)
        weights_dict = hrp.optimize()
        weights = pd.Series(weights_dict).reindex(train_returns.columns).fillna(0.0)
    except Exception:
        weights = pd.Series(1.0 / train_returns.shape[1], index=train_returns.columns)

    return normalize_weights(weights)


# -----------------------------
# 4. ROLLING BACKTEST
# -----------------------------
def run_backtest(monthly_returns, combined_signal, window):
    strategy_returns = {"MVO": [], "RP": [], "HRP": []}
    weights_history = {"MVO": [], "RP": [], "HRP": []}
    rebalance_dates = []

    for t in range(window, len(monthly_returns) - 1):
        train = monthly_returns.iloc[t - window:t]
        next_ret = monthly_returns.iloc[t + 1]

        signal_t = combined_signal.iloc[t].reindex(train.columns)

        w_mvo = mvo_weights(train, signal_tilt=signal_t)
        w_rp = risk_parity_weights(train)
        w_hrp = hrp_weights(train)

        r_mvo = np.dot(w_mvo, next_ret)
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

    weights_dict = {}
    for method in weights_history:
        weights_dict[method] = pd.DataFrame(weights_history[method], index=rebalance_dates)

    return returns_df, weights_dict


# -----------------------------
# 5. METRICS
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


def calmar_ratio(returns):
    ann_return = returns.mean() * 12
    mdd = abs(max_drawdown(returns))

    if mdd == 0:
        return np.nan

    return ann_return / mdd


def turnover(weights_df):
    changes = weights_df.diff().abs().sum(axis=1)
    return changes.mean()


def weight_stability(weights_df):
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
            "Calmar Ratio": calmar_ratio(returns_df[method]),
            "Average Turnover": turnover(weights_dict[method]),
            "Weight Stability": weight_stability(weights_dict[method]),
        })

    return pd.DataFrame(rows).set_index("Method")


# -----------------------------
# 6. PLOTS
# -----------------------------
def plot_cumulative_returns(returns_df, title):
    fig, ax = plt.subplots(figsize=(10, 6))

    cumulative = (1 + returns_df).cumprod()
    cumulative.plot(ax=ax)

    ax.set_title(title)
    ax.set_ylabel("Growth of 1€")
    ax.grid(True)

    plt.show()


def plot_drawdowns(returns_df, title):
    fig, ax = plt.subplots(figsize=(10, 6))

    dd_df = pd.DataFrame(index=returns_df.index)

    for col in returns_df.columns:
        wealth = (1 + returns_df[col]).cumprod()
        dd_df[col] = wealth / wealth.cummax() - 1

    dd_df.plot(ax=ax)

    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.grid(True)

    plt.show()


def plot_weights_over_time(weights_dict):
    methods = list(weights_dict.keys())

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for i, method in enumerate(methods):
        weights = weights_dict[method]
        weights.plot(ax=axes[i])
        axes[i].set_title(f"{method} Weights Over Time")
        axes[i].set_ylabel("Weight")
        axes[i].set_ylim(0, 1)   # 🔥 FIX: force same scale
        axes[i].grid(True)

    axes[-1].set_xlabel("Date")

    plt.tight_layout()
    plt.show()