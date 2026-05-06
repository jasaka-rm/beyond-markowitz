from core import (
    download_prices,
    to_monthly_prices,
    compute_returns,
    compute_momentum_signal,
    compute_trend_signal,
    combine_signals,
    run_backtest,
    summary_table,
    plot_cumulative_returns,
    plot_drawdowns,
    plot_weights_over_time,
)

from config_crypto import (
    CRYPTO_TICKERS,
    CRYPTO_START_DATE,
    CRYPTO_LOOKBACK,
    CRYPTO_MA_WINDOW,
    CRYPTO_ROLLING_WINDOW,
    CRYPTO_ASSET_LABELS,
)

import time


if __name__ == "__main__":
    print("Step 1: Downloading crypto prices...")
    t0 = time.time()
    prices_daily = download_prices(CRYPTO_TICKERS, CRYPTO_START_DATE)
    print(f"  ✓ Done in {time.time() - t0:.1f}s — shape: {prices_daily.shape}")

    print("Step 2: Converting to monthly...")
    t0 = time.time()
    prices_monthly = to_monthly_prices(prices_daily)
    monthly_returns = compute_returns(prices_monthly)
    print(f"  ✓ Done in {time.time() - t0:.1f}s — shape: {monthly_returns.shape}")

    print("Step 3: Computing signals...")
    t0 = time.time()
    momentum = compute_momentum_signal(prices_monthly, CRYPTO_LOOKBACK)
    trend = compute_trend_signal(prices_monthly, CRYPTO_MA_WINDOW)
    combined_signal = combine_signals(momentum, trend)
    combined_signal = combined_signal.reindex(monthly_returns.index)

    momentum = momentum.reindex(monthly_returns.index)
    trend = trend.reindex(monthly_returns.index)
    print(f"  ✓ Done in {time.time() - t0:.1f}s")

    print("Step 4: Running backtest...")
    t0 = time.time()
    returns_df, weights_dict = run_backtest(
        monthly_returns=monthly_returns,
        combined_signal=combined_signal,
        window=CRYPTO_ROLLING_WINDOW,
    )
    print(f"  ✓ Done in {time.time() - t0:.1f}s — {len(returns_df)} periods")

    print("Step 5: Computing metrics...")
    results = summary_table(returns_df, weights_dict)
    print(results.round(4))

    print("Step 6: Plotting...")
    plot_cumulative_returns(returns_df, "Cumulative Portfolio Returns — Crypto Universe")
    plot_drawdowns(returns_df, "Drawdowns — Crypto Universe")
    plot_weights_over_time(weights_dict)
    print("  ✓ All done")