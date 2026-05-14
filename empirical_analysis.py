import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from core import (
    download_prices,
    to_monthly_prices,
    compute_returns,
    run_backtest, 
    summary_table,
)

from config_etf import (
    ETF_TICKERS,
    ETF_START_DATE,
    ETF_LOOKBACK,
    ETF_MA_WINDOW,
    ETF_ASSET_LABELS,
)

from config_crypto import (
    CRYPTO_TICKERS,
    CRYPTO_START_DATE,
    CRYPTO_LOOKBACK,
    CRYPTO_MA_WINDOW,
    CRYPTO_ASSET_LABELS,
)


# -----------------------------
# 1. SHARED STYLE
# -----------------------------
STYLE = {
    "axes.facecolor":    "#F8FAFC",
    "figure.facecolor":  "#FFFFFF",
    "axes.grid":         True,
    "grid.alpha":        0.4,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
}
plt.rcParams.update(STYLE)

colors = ["#7C3AED", "#2563EB", "#16A34A"]

# -----------------------------
# 2. HELPERS
# -----------------------------
def ensure_plots_dir():
    os.makedirs("plots", exist_ok=True)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join("plots", filename), dpi=150, bbox_inches="tight")
    # plt.show()
    plt.close()
    print(f"  ✓ Saved: plots/{filename}")



# -----------------------------
# 3. PLOTS (5.2 Allocation Stability Analysis)
# -----------------------------
# Weights evolution
def plot_weights_over_time(weights_dict, asset_labels, universe_name, filename):
    methods = list(weights_dict.keys())

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for i, method in enumerate(methods):
        weights = weights_dict[method].rename(columns=asset_labels)
        weights.plot(ax=axes[i], legend=False)

        axes[i].set_title(f"{method} Weights — {universe_name}")
        axes[i].set_ylabel("Weight")
        axes[i].set_ylim(0, 1)
        axes[i].grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Assets", loc="center left", bbox_to_anchor=(1.01, 0.5))

    axes[-1].set_xlabel("Date")
    fig.suptitle(f"Evolution of Portfolio Weights — {universe_name}", fontsize=14)

    save_plot(filename)

# Estability bar chart
def plot_quantitative_stability(results, universe_name, filename):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    results["Weight Stability"].plot(
        kind="bar",
        ax=ax,
        color=colors
    )

    ax.set_title(f"Weight Stability — {universe_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Std of weights", fontsize=10)
    ax.set_xlabel("Method", fontsize=10)
    ax.set_ylim(0, 0.22)
    ax.grid(axis="y")

    for i, v in enumerate(results["Weight Stability"].values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    save_plot(filename)



# -----------------------------
# 3. PLOTS (5.3 Drawdown Analysis)
# -----------------------------
# Drawdown + Time Under Water
def plot_drawdown_with_recovery(returns_df, universe_name, filename):
    fig, ax = plt.subplots(figsize=(12, 6))

    recovery_stats = {}

    for col, color in zip(returns_df.columns, colors):
        returns = returns_df[col]

        wealth = (1 + returns).cumprod()
        peak = wealth.cummax()
        drawdown = wealth / peak - 1

        ax.plot(drawdown.index, drawdown, label=col, color=color)

        # --- Compute recovery time ---
        trough = drawdown.idxmin()
        peak_before = wealth.loc[:trough].idxmax()

        recovery = wealth.loc[trough:]
        recovery_date = recovery[recovery >= wealth.loc[peak_before]].index

        if len(recovery_date) > 0:
            recovery_date = recovery_date[0]
            time_under_water = (recovery_date - trough).days / 30
        else:
            time_under_water = np.nan

        recovery_stats[col] = time_under_water

    ax.set_title(f"Drawdowns with Recovery — {universe_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")

    # plt.show()
    plt.close()   
    plt.close(fig)

    print(f"  ✓ Saved: plots/{filename}")

    return recovery_stats


# Recovery Time Table
def print_drawdown_table(recovery_stats, avg_drawdown_stats, universe_name):
    print("\n" + "═" * 70)
    print(f"  DRAWDOWN METRICS — {universe_name}")
    print("═" * 70)
    print(f"{'Method':<10} {'Max Time Under Water':>22} {'Avg Drawdown Depth':>22}")
    print("─" * 70)

    for method in recovery_stats.keys():
        recovery = recovery_stats[method]
        avg_dd = avg_drawdown_stats.get(method, np.nan)

        if np.isnan(recovery):
            recovery_str = "Not recovered"
        else:
            recovery_str = f"{recovery:.1f} months"

        if np.isnan(avg_dd):
            avg_dd_str = "NaN"
        else:
            avg_dd_str = f"{avg_dd:.4f}"

        print(f"{method:<10} {recovery_str:>22} {avg_dd_str:>22}")

    print("═" * 70 + "\n")


# -----------------------------
# 3. PLOTS (5.4 Risk-adjusted performance)
# -----------------------------
def plot_sharpe_comparison(results, universe_name, filename):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    results["Sharpe Ratio"].plot(kind="bar", ax=ax, color=colors)

    ax.set_title(f"Sharpe Ratio Comparison — {universe_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio", fontsize=10)
    ax.set_xlabel("Method", fontsize=10)
    ax.grid(axis="y")

    for i, v in enumerate(results["Sharpe Ratio"].values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    save_plot(filename)


def compute_rolling_sharpe(returns, window=12, rf=0.0):
    excess = returns - rf / 12
    rolling_mean = excess.rolling(window).mean()
    rolling_std = excess.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(12)
    return rolling_sharpe


def plot_rolling_sharpe(returns_df, universe_name, filename, window=12):
    fig, ax = plt.subplots(figsize=(13, 5))

    rolling_data = {}

    for col in returns_df.columns:
        rolling_data[col] = compute_rolling_sharpe(returns_df[col], window=window)

    rolling_df = pd.DataFrame(rolling_data, index=returns_df.index)
    rolling_df.plot(ax=ax, linewidth=1.8, color=colors)

    ax.axhline(0, color="#94A3B8", linestyle="--", linewidth=1)
    ax.set_title(f"Rolling {window}-Month Sharpe Ratio — {universe_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Rolling Sharpe", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.grid(True)
    ax.legend(title="Method", fontsize=9)

    save_plot(filename)


def print_risk_adjusted_table(results, universe_name, filename_csv):
    table = results[[
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Calmar Ratio",
    ]].copy()

    print("\n" + "═" * 60)
    print(f"  RISK-ADJUSTED PERFORMANCE — {universe_name}")
    print("═" * 60)
    print(table.round(4))
    print("═" * 60 + "\n")

    #table.round(4).to_csv(os.path.join("plots", filename_csv))
    print(f"  ✓ Saved: plots/{filename_csv}")


# -----------------------------
# 3. PLOTS (5.5 Turnover)
# -----------------------------
def compute_turnover_series(weights_df):
    turnover_series = weights_df.diff().abs().sum(axis=1)
    return turnover_series.dropna()


def plot_annual_turnover(results, weights_dict, universe_name, filename):
    annual_turnover = {}

    for method in weights_dict:
        turnover_series = compute_turnover_series(weights_dict[method])
        annual_turnover[method] = turnover_series.mean() * 12

    annual_turnover = pd.Series(annual_turnover)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    annual_turnover.plot(kind="bar", ax=ax, color=colors)

    ax.set_title(f"Annual Turnover — {universe_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Annualized turnover", fontsize=10)
    ax.set_xlabel("Method", fontsize=10)
    ax.grid(axis="y")

    for i, v in enumerate(annual_turnover.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    save_plot(filename)


def compute_regimes(monthly_returns, window=6):
    """
    Simple regime classification based on rolling volatility of an equal-weight
    universe proxy. Months above the median rolling volatility are classified
    as high-volatility regimes; the rest are low-volatility regimes.
    """
    universe_proxy = monthly_returns.mean(axis=1)
    rolling_vol = universe_proxy.rolling(window).std()

    threshold = rolling_vol.median()
    regimes = pd.Series(index=monthly_returns.index, dtype="object")
    regimes.loc[rolling_vol <= threshold] = "Low-vol regime"
    regimes.loc[rolling_vol > threshold] = "High-vol regime"

    return regimes.dropna()



def plot_turnover_by_regime(weights_dict, monthly_returns, universe_name, filename):
    regimes = compute_regimes(monthly_returns, window=6)

    rows = []
    for method in weights_dict:
        turnover_series = compute_turnover_series(weights_dict[method])
        aligned_regimes = regimes.reindex(turnover_series.index).dropna()
        aligned_turnover = turnover_series.reindex(aligned_regimes.index)

        for regime_name in ["Low-vol regime", "High-vol regime"]:
            mask = aligned_regimes == regime_name
            avg_turnover = aligned_turnover.loc[mask].mean() * 12
            rows.append({
                "Method": method,
                "Regime": regime_name,
                "Annualized Turnover": avg_turnover,
            })

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="Method", columns="Regime", values="Annualized Turnover")
    pivot = pivot.reindex(["MVO", "RP", "HRP"])

    fig, ax = plt.subplots(figsize=(10, 4.8))

    color_map = {"MVO": "#7C3AED", "RP": "#2563EB", "HRP": "#16A34A",    }

    methods = pivot.index.tolist()
    x = np.arange(len(methods))
    width = 0.35

    bars_high = ax.bar(x - width/2, pivot["High-vol regime"], width, label="High-vol regime", color=[color_map[m] for m in methods], alpha=1.0)
    bars_low = ax.bar(x + width/2, pivot["Low-vol regime"], width, label="Low-vol regime", color=[color_map[m] for m in methods], alpha=0.5)
    
    # Add labels
    for bar in bars_high:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.2f}",
            ha='center',
            va='bottom',
            fontsize=9
        )

    for bar in bars_low:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.2f}",
            ha='center',
            va='bottom',
            fontsize=9
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)

    ax.set_title(f"Turnover Across Regimes — {universe_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Annualized turnover", fontsize=10)
    ax.set_xlabel("Method", fontsize=10)
    ax.grid(axis="y")
    ax.legend(title="Regime", fontsize=9)

    save_plot(filename)

def print_turnover_table(results, weights_dict, universe_name, filename_csv):
    annual_turnover = {}

    for method in weights_dict:
        turnover_series = compute_turnover_series(weights_dict[method])
        annual_turnover[method] = turnover_series.mean() * 12

    turnover_df = pd.DataFrame({
        "Annualized Turnover": annual_turnover
    })

    print("\n" + "═" * 60)
    print(f"  TURNOVER AND IMPLEMENTATION REALISM — {universe_name}")
    print("═" * 60)
    print(turnover_df.round(4))
    print("═" * 60 + "\n")

    turnover_df.round(4).to_csv(os.path.join("plots", filename_csv))
    print(f"  ✓ Saved: plots/{filename_csv}")


# -----------------------------
# 3. PLOTS (5.5 Bootstrap)
# -----------------------------
def plot_bootstrap_stability(results_df, results_boot, universe_name, filename):
    import matplotlib.pyplot as plt
    import numpy as np

    methods = results_df.index.tolist()

    # Colors per method
    colors = {
        "MVO": "#7C3AED",
        "RP": "#2563EB",
        "HRP": "#16A34A"
    }

    # Data
    std_vals = results_df["Weight Stability"].values
    boot_vals = results_boot["Weight Stability"].values

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Plot bars manually
    for i, method in enumerate(methods):
        color = colors.get(method, "#333333")

        # Standard (full opacity)
        ax.bar(
            x[i] - width/2,
            std_vals[i],
            width,
            color=color,
            alpha=1.0,
            label="Standard" if i == 0 else ""
        )

        # Bootstrap (lighter)
        ax.bar(
            x[i] + width/2,
            boot_vals[i],
            width,
            color=color,
            alpha=0.5,
            label="Bootstrap" if i == 0 else ""
        )

    # Labels & title
    ax.set_title(f"Allocation Stability: Standard vs Bootstrap — {universe_name}", fontweight="bold")
    ax.set_ylabel("Weight Stability (Std Dev)")
    ax.set_xlabel("Method")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.grid(axis="y")

    # Value labels
    for i in range(len(methods)):
        ax.text(x[i] - width/2, std_vals[i] + 0.002, f"{std_vals[i]:.3f}", ha="center", fontsize=8)
        ax.text(x[i] + width/2, boot_vals[i] + 0.002, f"{boot_vals[i]:.3f}", ha="center", fontsize=8)

    # Legend (only once)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"plots/{filename}", dpi=150)
    plt.close()

    print(f"✓ Saved: plots/{filename}")


def plot_bootstrap_turnover(results_df, results_boot, universe_name, filename):

    methods = results_df.index.tolist()

    # Colors per method
    colors = {
        "MVO": "#7C3AED",
        "RP": "#2563EB",
        "HRP": "#16A34A"
    }

    # Data
    std_vals = results_df["Average Turnover"].values
    boot_vals = results_boot["Average Turnover"].values

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Plot bars manually
    for i, method in enumerate(methods):
        color = colors.get(method, "#333333")

        # Standard (full opacity)
        ax.bar(
            x[i] - width/2,
            std_vals[i],
            width,
            color=color,
            alpha=1.0,
            label="Standard" if i == 0 else ""
        )

        # Bootstrap (lighter)
        ax.bar(
            x[i] + width/2,
            boot_vals[i],
            width,
            color=color,
            alpha=0.5,
            label="Bootstrap" if i == 0 else ""
        )

    # Labels & title
    ax.set_title(f"Turnover: Standard vs Bootstrap — {universe_name}", fontweight="bold")
    ax.set_ylabel("Average Turnover")
    ax.set_xlabel("Method")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.grid(axis="y")

    # Value labels
    for i in range(len(methods)):
        ax.text(x[i] - width/2, std_vals[i] + 0.02, f"{std_vals[i]:.2f}", ha="center", fontsize=8)
        ax.text(x[i] + width/2, boot_vals[i] + 0.02, f"{boot_vals[i]:.2f}", ha="center", fontsize=8)

    # Legend
    ax.legend(title="Estimation")

    plt.tight_layout()
    plt.savefig(f"plots/{filename}", dpi=150)
    plt.close()

    print(f"✓ Saved: plots/{filename}")

# -----------------------------
# 4. ANALYSIS PIPELINE
# -----------------------------
def run_signal_validation(tickers, start_date, lookback, ma_window, asset_labels, universe_name, file_prefix):
    print(f"\n{universe_name}")
    print("Step 1: Downloading prices...")
    prices_daily = download_prices(tickers, start_date)
    prices = to_monthly_prices(prices_daily)

    monthly_returns = compute_returns(prices)


    print("Step 2: Running allocation analysis (standard)...")

    returns_df, weights_dict = run_backtest(
        monthly_returns=monthly_returns,
        window=36 if "ETF" in universe_name else 12,
        mode="standard"
    )

    print("Step 3: Running allocation analysis (bootstrap)...")

    returns_boot, weights_boot = run_backtest(
        monthly_returns=monthly_returns,
        window=36 if "ETF" in universe_name else 12,
        mode="bootstrap"
    )

    results = summary_table(returns_df, weights_dict)
    print(results.round(4))
    
    print("Step 4: Generating plots 5.1...")
    plot_weights_over_time(
    weights_dict,
    asset_labels,
    universe_name,
    f"{file_prefix}_weights.png",
)

    plot_quantitative_stability(
        results,
        universe_name,
        f"{file_prefix}_stability_metrics.png",
    )

    print("Step 5: Generating plots 5.2...")
    recovery_stats = plot_drawdown_with_recovery(
        returns_df,
        universe_name,
        f"{file_prefix}_drawdowns.png"
    )

    avg_drawdown_stats = results["Average Drawdown Depth"].to_dict()
    print_drawdown_table(recovery_stats, avg_drawdown_stats, universe_name)


    print("Step 6: Generating plots 5.3 (risk-adjusted performance plots)...")

    plot_sharpe_comparison(
        results,
        universe_name,
        f"{file_prefix}_sharpe_comparison.png",
    )

    plot_rolling_sharpe(
        returns_df,
        universe_name,
        f"{file_prefix}_rolling_sharpe.png",
        window=12,
    )

    print_risk_adjusted_table(
        results,
        universe_name,
        f"{file_prefix}_risk_adjusted_table.csv",
    )

    print("Step 7: Generating plots 5.4 (Generating turnover plots)...")

    plot_annual_turnover(
        results,
        weights_dict,
        universe_name,
        f"{file_prefix}_annual_turnover.png",
    )

    plot_turnover_by_regime(
        weights_dict,
        monthly_returns,
        universe_name,
        f"{file_prefix}_turnover_by_regime.png",
    )

    print_turnover_table(
        results,
        weights_dict,
        universe_name,
        f"{file_prefix}_turnover_table.csv",
    )

    print("Step 8: Generating plots 5.5 (Generating bootstrap plots)...")

    results_std = summary_table(returns_df, weights_dict)
    results_boot = summary_table(returns_boot, weights_boot)

    print("\nStandard results:")
    print(results_std.round(4))

    print("\nBootstrap results:")
    print(results_boot.round(4))

    plot_bootstrap_stability(
        results_std, 
        results_boot, 
        universe_name,
        f"{file_prefix}_bootstrap_stability.png",
    )

    plot_bootstrap_turnover(
        results_std, 
        results_boot, 
        universe_name,
        f"{file_prefix}_bootstrap_turnover.png"
    )

# -----------------------------
# 5. MAIN
# -----------------------------
if __name__ == "__main__":
    ensure_plots_dir()

    t0 = time.time()
    run_signal_validation(
        tickers=ETF_TICKERS,
        start_date=ETF_START_DATE,
        lookback=ETF_LOOKBACK,
        ma_window=ETF_MA_WINDOW,
        asset_labels=ETF_ASSET_LABELS,
        universe_name="ETF Universe",
        file_prefix="etf",
    )
    print(f"  ✓ Done in {time.time() - t0:.1f}s")

    t0 = time.time()
    run_signal_validation(
        tickers=CRYPTO_TICKERS,
        start_date=CRYPTO_START_DATE,
        lookback=CRYPTO_LOOKBACK,
        ma_window=CRYPTO_MA_WINDOW,
        asset_labels=CRYPTO_ASSET_LABELS,
        universe_name="Crypto Universe",
        file_prefix="crypto",
    )
    print(f"  ✓ Done in {time.time() - t0:.1f}s")