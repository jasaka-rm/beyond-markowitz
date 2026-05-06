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


# -----------------------------
# 2. HELPERS
# -----------------------------
def ensure_plots_dir():
    os.makedirs("plots", exist_ok=True)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join("plots", filename), dpi=150, bbox_inches="tight")
    plt.show()
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
# 3. PLOTS
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

    datasets = [
        (momentum_signal.rank(axis=1, pct=True), "Momentum Rank", "#2563EB"),
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


def plot_signal_dispersion(combined_signal, title, filename):
    dispersion = combined_signal.std(axis=1)
    roll_disp = dispersion.rolling(3).mean()

    fig, ax = plt.subplots(figsize=(13, 4.5))

    ax.fill_between(dispersion.index, 0, dispersion, alpha=0.2, color="#7C3AED")
    ax.plot(dispersion.index, dispersion, color="#7C3AED", alpha=0.5, linewidth=1.2, label="Monthly std")
    ax.plot(roll_disp.index, roll_disp, color="#7C3AED", linewidth=2.2, label="3-month MA")

    low_threshold = dispersion.quantile(0.25)
    ax.axhline(low_threshold, color="#DC2626", linestyle=":", linewidth=1.3, label=f"25th pct ({low_threshold:.2f})")
    ax.fill_between(
        dispersion.index,
        0,
        dispersion,
        where=dispersion <= low_threshold,
        alpha=0.35,
        color="#FCA5A5",
        label="Low differentiation",
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Std of combined signal across assets", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.legend(fontsize=9)

    save_plot(filename)


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


def plot_signal_strategy(combined_signal, monthly_returns, title, filename, top_n=2):
    returns_list = []

    for date in combined_signal.index:
        if date not in monthly_returns.index:
            continue

        signal_t = combined_signal.loc[date].dropna()
        ret_t = monthly_returns.loc[date].reindex(signal_t.index)

        long_assets = signal_t.nlargest(top_n).index
        long_ret = ret_t[long_assets].mean()
        bench_ret = ret_t.mean()

        returns_list.append({
            "date": date,
            "signal_long": long_ret,
            "equal_weight": bench_ret,
        })

    df = pd.DataFrame(returns_list).set_index("date")
    cumulative = (1 + df).cumprod()

    fig, ax = plt.subplots(figsize=(13, 5))
    cumulative["signal_long"].plot(ax=ax, color="#2563EB", linewidth=2.2, label=f"Long top-{top_n} (signal)")
    cumulative["equal_weight"].plot(ax=ax, color="#94A3B8", linewidth=1.8, linestyle="--", label="Equal weight (benchmark)")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Growth of $1", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.legend(fontsize=10)

    save_plot(filename)


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
    # monthly_returns = compute_returns(to_monthly_prices(prices)).reindex(combined_signal.index)
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
    plot_signal_dispersion(
        combined_signal,
        f"Cross-Sectional Signal Dispersion — {universe_name}",
        f"{file_prefix}_signal_dispersion.png",
    )
    plot_ranking_turnover(
        combined_signal,
        f"Signal Ranking Turnover Over Time — {universe_name}",
        f"{file_prefix}_signal_ranking_turnover.png",
    )
    plot_signal_strategy(
        combined_signal,
        monthly_returns,
        f"Signal Strategy: Long Top-2 Assets vs Equal Weight — {universe_name}",
        f"{file_prefix}_signal_strategy.png",
        top_n=2,
    )

    print(f"\n✓ All plots saved to /plots for {universe_name}")


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