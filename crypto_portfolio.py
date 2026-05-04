import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from pypfopt.hierarchical_portfolio import HRPOpt
from scipy.optimize import minimize


# -----------------------------
# 1. DATA SETTINGS
# -----------------------------
TICKERS = ["BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD", "DOGE-USD"]
START_DATE = "2015-09-01"   # ETH-USD data starts ~Aug 2015
END_DATE = None              # use latest available data

# 12 months is more appropriate for crypto:
#   - Crypto bull/bear cycles are much shorter than equities
#   - Avoids eating too much of the available history (only ~10 years total)
#   - Makes the covariance estimate more reactive to current regime
ROLLING_WINDOW_MONTHS = 12
REBALANCE_FREQ = "ME"   # monthly
RISK_FREE_RATE = 0.0    # no agreed-upon risk-free rate for crypto


# -----------------------------
# 2. DOWNLOAD DATA
# -----------------------------
def download_prices(tickers, start, end=None):
    # Crypto trades 24/7 so no missing weekends — dropna() is safe
    prices = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    prices = prices.dropna()
    return prices


# -----------------------------
# 3. CALCULATE MONTHLY PRICES
# -----------------------------
def to_monthly_prices(prices):
    monthly_prices = prices.resample("ME").last()
    return monthly_prices.dropna()


def compute_returns(prices):
    return prices.pct_change().dropna()


# -----------------------------
# 4. SIGNAL CONSTRUCTION
# -----------------------------
def compute_momentum_signal(monthly_prices, lookback=6):
    # Reduced from 12 to 6 months: crypto momentum decays faster than equities.
    # A 12-month lookback would mix two very different market regimes in crypto.
    return monthly_prices.pct_change(lookback)


def compute_trend_signal(monthly_prices, ma_window=6):
    # 6-month MA instead of 12: more responsive to the faster crypto cycles
    moving_avg = monthly_prices.rolling(ma_window).mean()
    return (monthly_prices > moving_avg).astype(float)


def combine_signals(momentum, trend):
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

    if signal_tilt is not None:
        mu = mu * signal_tilt.reindex(mu.index).fillna(1.0)

    try:
        raw = np.linalg.pinv(cov.values) @ mu.values
        weights = pd.Series(raw, index=train_returns.columns)
    except Exception:
        weights = pd.Series(1.0, index=train_returns.columns)

    weights[weights < 0] = 0
    return normalize_weights(weights)


def risk_parity_weights(train_returns, vol_floor=1e-3):
    # vol_floor raised from 1e-4 to 1e-3:
    # Crypto monthly vols are typically 20-80%+ annualised, so 1e-4 was
    # too tight and could cause numerical issues near-zero vol assets.
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
# 6. ROLLING BACKTEST
# -----------------------------
def run_backtest(monthly_returns, combined_signal, window=12):
    strategy_returns = {"MVO": [], "RP": [], "HRP": []}
    weights_history  = {"MVO": [], "RP": [], "HRP": []}
    rebalance_dates  = []

    for t in range(window, len(monthly_returns) - 1):
        train    = monthly_returns.iloc[t - window:t]
        next_ret = monthly_returns.iloc[t + 1]
        signal_t = combined_signal.iloc[t].reindex(train.columns)

        w_mvo = mvo_weights(train, signal_tilt=signal_t)
        w_rp  = risk_parity_weights(train)
        w_hrp = hrp_weights(train)

        strategy_returns["MVO"].append(np.dot(w_mvo, next_ret))
        strategy_returns["RP"].append(np.dot(w_rp,  next_ret))
        strategy_returns["HRP"].append(np.dot(w_hrp, next_ret))

        weights_history["MVO"].append(w_mvo)
        weights_history["RP"].append(w_rp)
        weights_history["HRP"].append(w_hrp)

        rebalance_dates.append(monthly_returns.index[t + 1])

    returns_df = pd.DataFrame(strategy_returns, index=rebalance_dates)

    weights_df = {
        method: pd.DataFrame(weights_history[method], index=rebalance_dates)
        for method in weights_history
    }

    return returns_df, weights_df


# -----------------------------
# 7. METRICS
# -----------------------------
def max_drawdown(returns):
    wealth = (1 + returns).cumprod()
    peak   = wealth.cummax()
    return (wealth / peak - 1).min()


def sharpe_ratio(returns, rf=0.0):
    excess = returns - rf / 12
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(12)


def turnover(weights_df):
    return weights_df.diff().abs().sum(axis=1).mean()


def weight_stability(weights_df):
    return weights_df.std().mean()


def calmar_ratio(returns):
    """Annualised return divided by absolute max drawdown — useful for crypto
    because Sharpe alone doesn't capture the severity of drawdown cycles."""
    ann_return = returns.mean() * 12
    mdd = abs(max_drawdown(returns))
    return ann_return / mdd if mdd != 0 else np.nan


def summary_table(returns_df, weights_dict):
    rows = []
    for method in returns_df.columns:
        rows.append({
            "Method":             method,
            "Annualized Return":  returns_df[method].mean() * 12,
            "Annualized Vol":     returns_df[method].std() * np.sqrt(12),
            "Sharpe Ratio":       sharpe_ratio(returns_df[method]),
            "Max Drawdown":       max_drawdown(returns_df[method]),
            "Calmar Ratio":       calmar_ratio(returns_df[method]),  # added for crypto
            "Avg Turnover":       turnover(weights_dict[method]),
            "Weight Stability":   weight_stability(weights_dict[method]),
        })
    return pd.DataFrame(rows).set_index("Method")


# -----------------------------
# 8. PLOTS
# -----------------------------
def plot_cumulative_returns(returns_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    (1 + returns_df).cumprod().plot(ax=ax)
    ax.set_title("Cumulative Portfolio Returns — Crypto Universe")
    ax.set_ylabel("Growth of 1€")
    ax.grid(True)
    plt.tight_layout()
    plt.show(block=False)


def plot_drawdowns(returns_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    dd_df = pd.DataFrame(index=returns_df.index)
    for col in returns_df.columns:
        wealth      = (1 + returns_df[col]).cumprod()
        dd_df[col]  = wealth / wealth.cummax() - 1
    dd_df.plot(ax=ax)
    ax.set_title("Drawdowns — Crypto Universe")
    ax.set_ylabel("Drawdown")
    ax.grid(True)
    plt.tight_layout()
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


def plot_rolling_volatility(returns_df, window=6):
    fig, ax = plt.subplots(figsize=(10, 6))
    rolling_vol = returns_df.rolling(window).std() * np.sqrt(12)
    rolling_vol.plot(ax=ax)
    ax.set_title(f"Rolling {window}-Month Annualised Volatility — Crypto Universe")
    ax.set_ylabel("Annualised Volatility")
    ax.grid(True)
    plt.tight_layout()
    plt.show(block=False)

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
    momentum        = compute_momentum_signal(prices_monthly, lookback=6)
    trend           = compute_trend_signal(prices_monthly, ma_window=6)
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
    plot_rolling_volatility(returns_df, window=6)
    print("  ✓ All done")

    plt.show(block=True)