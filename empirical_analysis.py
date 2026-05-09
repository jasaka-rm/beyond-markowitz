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
    compute_momentum_signal,
    compute_trend_signal,
    combine_signals,
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


def print_signal_summary(combined_signal, momentum_signal, trend_signal, asset_labels, universe_name):
    print("\n" + "═" * 60)
    print(f"  SIGNAL VALIDATION — SUMMARY STATISTICS — {universe_name}")
    print("═" * 60)

    dispersion = combined_signal.std(axis=1)
    ranks = combined_signal.rank(axis=1)
    rank_chg = ranks.diff().abs().sum(axis=1).dropna()
    avg_signal = combined_signal.mean(axis=1)

    stats = {
        "Mean combined signal":       avg_signal.mean(),
        "Std combined signal (time)": avg_signal.std(),
        "Mean cross-sect. dispersion": dispersion.mean(),
        "Std cross-sect. dispersion":  dispersion.std(),
        "Pct months signal > 0.5":    (avg_signal > 0.5).mean(),
        "Mean ranking turnover":      rank_chg.mean(),
        "Std ranking turnover":       rank_chg.std(),
    }

    for k, v in stats.items():
        print(f"  {k:<38} {v:.4f}")

    print("\n  Per-asset mean combined signal:")
    per_asset = combined_signal.rename(columns=asset_labels).mean()
    for asset, val in per_asset.items():
        print(f"    {asset:<6}  {val:.4f}")

    print("═" * 60 + "\n")


# -----------------------------
# 3. PLOTS (5.1 Signal Validation)
# -----------------------------
def plot_signal_heatmap(combined_signal, asset_labels, title, filename):
    data = combined_signal.rename(columns=asset_labels).T

    fig, ax = plt.subplots(figsize=(14, 3.8))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        center=0.5,
        vmin=0,
        vmax=1,
        linewidths=0.3,
        linecolor="#E2E8F0",
        cbar_kws={
            "label": "Signal strength (0 = bearish · 1 = bullish)",
            "shrink": 0.8,
        },
    )

    n_cols = data.shape[1]
    step = max(1, n_cols // 20)
    ticks = range(0, n_cols, step)
    labels = [data.columns[i].strftime("%Y-%m") for i in ticks]
    ax.set_xticks([t + 0.5 for t in ticks])
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("")

    save_plot(filename)


def plot_average_signal(momentum_signal, trend_signal, combined_signal, title, filename):
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)

    # More interpretable than averaging ranks:
    # share of assets with positive momentum at each date
    momentum_breadth = (momentum_signal > 0).astype(float)

    datasets = [
        (momentum_breadth, "Momentum Breadth", "#2563EB"),
        (trend_signal, "Trend Signal", "#16A34A"),
        (combined_signal, "Combined Signal", "#7C3AED"),
    ]

    for ax, (data, label, color) in zip(axes, datasets):
        mean_signal = data.mean(axis=1)
        roll_mean = mean_signal.rolling(3).mean()

        ax.fill_between(
            mean_signal.index,
            0.5,
            mean_signal,
            where=mean_signal >= 0.5,
            alpha=0.25,
            color=color,
            label="_nolegend_",
        )
        ax.fill_between(
            mean_signal.index,
            mean_signal,
            0.5,
            where=mean_signal < 0.5,
            alpha=0.25,
            color="#DC2626",
            label="_nolegend_",
        )

        ax.plot(mean_signal.index, mean_signal, color=color, alpha=0.55, linewidth=1.2, label="Monthly avg")
        ax.plot(roll_mean.index, roll_mean, color=color, linewidth=2.0, label="3-month MA")
        ax.axhline(0.5, color="#94A3B8", linestyle="--", linewidth=1)

        ax.set_ylabel(label, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9, loc="upper left")

    axes[-1].set_xlabel("Date", fontsize=10)
    save_plot(filename)


# def plot_signal_dispersion(combined_signal, title, filename):
#     dispersion = combined_signal.std(axis=1)

#     fig, ax = plt.subplots(figsize=(13, 4.5))

#     low_threshold = dispersion.quantile(0.25)
#     low_share = (dispersion <= low_threshold).mean() * 100

#     ax.fill_between(dispersion.index, 0, dispersion, alpha=0.2, color="#7C3AED")
#     ax.plot(dispersion.index, dispersion, color="#7C3AED", linewidth=1.2, label="Monthly std")

#     ax.axhline(
#         low_threshold,
#         color="#DC2626",
#         linestyle=":",
#         linewidth=1.3,
#         label=f"25th pct ({low_threshold:.2f})",
#     )

#     ax.fill_between(
#         dispersion.index,
#         0,
#         dispersion,
#         where=dispersion <= low_threshold,
#         alpha=0.35,
#         color="#FCA5A5",
#         label=f"Low differentiation ({low_share:.1f}%)",
#     )

#     # ax.set_xlim(left=pd.Timestamp("2018-01-01"))
#     ax.set_title(title, fontsize=14, fontweight="bold")
#     ax.set_ylabel("Std of combined signal across assets", fontsize=10)
#     ax.set_xlabel("Date", fontsize=10)
#     ax.legend(fontsize=9)

#     save_plot(filename)


def plot_ranking_turnover(combined_signal, title, filename):
    ranks = combined_signal.rank(axis=1)
    rank_changes = ranks.diff().abs().sum(axis=1).dropna()
    roll_turn = rank_changes.rolling(3).mean()

    fig, ax = plt.subplots(figsize=(13, 4.5))

    ax.fill_between(rank_changes.index, 0, rank_changes, alpha=0.2, color="#D97706")
    ax.plot(rank_changes.index, rank_changes, color="#D97706", alpha=0.5, linewidth=1.2, label="Monthly turnover")
    ax.plot(roll_turn.index, roll_turn, color="#D97706", linewidth=2.2, label="3-month MA")

    mean_turn = rank_changes.mean()
    ax.axhline(mean_turn, color="#94A3B8", linestyle="--", linewidth=1.3, label=f"Mean ({mean_turn:.2f})")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Sum of absolute rank changes across assets", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.legend(fontsize=9)

    ax.text(
        rank_changes.index[-1],
        mean_turn * 1.08,
        f"Avg: {mean_turn:.1f}",
        fontsize=9,
        color="#64748B",
        ha="right",
    )

    save_plot(filename)



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
# 4. ANALYSIS PIPELINE
# -----------------------------
def run_signal_validation(tickers, start_date, lookback, ma_window, asset_labels, universe_name, file_prefix):
    print(f"\n{universe_name}")
    print("Step 1: Downloading prices...")
    t0 = time.time()
    prices_daily = download_prices(tickers, start_date)
    prices = to_monthly_prices(prices_daily)
    print(f"  ✓ Done in {time.time() - t0:.1f}s — shape: {prices.shape}")

    print("Step 2: Computing signals...")
    t0 = time.time()
    momentum = compute_momentum_signal(prices, lookback)
    trend = compute_trend_signal(prices, ma_window)
    combined_signal = combine_signals(momentum, trend).dropna()

    momentum = momentum.reindex(combined_signal.index)
    trend = trend.reindex(combined_signal.index)
    monthly_returns = compute_returns(prices).reindex(combined_signal.index)


    print(f"  ✓ Done in {time.time() - t0:.1f}s")

    print_signal_summary(combined_signal, momentum, trend, asset_labels, universe_name)

    print("Step 3: Generating plots...")
    plot_signal_heatmap(
        combined_signal,
        asset_labels,
        f"Combined Signal Heatmap — {universe_name}",
        f"{file_prefix}_signal_heatmap.png",
    )
    plot_average_signal(
        momentum,
        trend,
        combined_signal,
        f"Average Signal Strength Over Time — {universe_name}",
        f"{file_prefix}_signal_average.png",
    )
    # plot_signal_dispersion(
    #     combined_signal,
    #     f"Cross-Sectional Signal Dispersion — {universe_name}",
    #     f"{file_prefix}_signal_dispersion.png",
    # )
    plot_ranking_turnover(
        combined_signal,
        f"Signal Ranking Turnover Over Time — {universe_name}",
        f"{file_prefix}_signal_ranking_turnover.png",
    )

    print(f"\n✓ All plots 5.1 saved to /plots for {universe_name}")

    print("Step 4: Running allocation analysis...")

    returns_df, weights_dict = run_backtest(
        monthly_returns=monthly_returns,
        combined_signal=combined_signal,
        window=36 if "ETF" in universe_name else 12,
    )

    results = summary_table(returns_df, weights_dict)
    print(results.round(4))
    
    print("Step 5: Generating plots 5.2...")
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

    print("Step 6: Generating plots 5.3...")
    recovery_stats = plot_drawdown_with_recovery(
        returns_df,
        universe_name,
        f"{file_prefix}_drawdowns.png"
    )

    avg_drawdown_stats = results["Average Drawdown Depth"].to_dict()
    print_drawdown_table(recovery_stats, avg_drawdown_stats, universe_name)


    print("Step 7: Generating plots 5.4 (risk-adjusted performance plots)...")

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

    print("Step 8: Generating plots 5.5 (Generating turnover plots)...")

    plot_annual_turnover(
        results,
        weights_dict,
        universe_name,
        f"{file_prefix}_annual_turnover.png",
    )

    print_turnover_table(
        results,
        weights_dict,
        universe_name,
        f"{file_prefix}_turnover_table.csv",
    )


# -----------------------------
# 5. MAIN
# -----------------------------
if __name__ == "__main__":
    ensure_plots_dir()

    run_signal_validation(
        tickers=ETF_TICKERS,
        start_date=ETF_START_DATE,
        lookback=ETF_LOOKBACK,
        ma_window=ETF_MA_WINDOW,
        asset_labels=ETF_ASSET_LABELS,
        universe_name="ETF Universe",
        file_prefix="etf",
    )

    run_signal_validation(
        tickers=CRYPTO_TICKERS,
        start_date=CRYPTO_START_DATE,
        lookback=CRYPTO_LOOKBACK,
        ma_window=CRYPTO_MA_WINDOW,
        asset_labels=CRYPTO_ASSET_LABELS,
        universe_name="Crypto Universe",
        file_prefix="crypto",
    )
    