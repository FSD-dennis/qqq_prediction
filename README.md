# QQQ 5-Day Forward Return Pipeline

End-to-end Python project that predicts **QQQ 5-trading-day forward return** from daily OHLCV data and backtests a long-only strategy.

## Overview

- **Ticker:** QQQ
- **Date range:** 2010-01-01 to 2025-12-31
- **Target:** 5-day forward return using Adjusted Close
- **Models:** Linear Regression, Random Forest, LightGBM
- **Strategy:** long when prediction is in the top 30% of test-period predictions
- **Outputs:** `results.csv`, `cumulative_returns.png`

## Project Files

- `qqq_pipeline.py` — full pipeline (data, features, models, backtest, reporting)
- `requirements.txt` — pinned Python package versions
- `.python-version` — pinned Python interpreter version (3.12.12)
- `results.csv` — final model + strategy metrics table
- `cumulative_returns.png` — strategy vs buy-and-hold cumulative curve
- `qqq_daily_data.csv` — local cached daily data used by the pipeline

## Data Source Behavior

The pipeline tries data sources in this order:

1. **Yahoo Finance (`yfinance`)** with retries
2. **Stooq fallback** if Yahoo is rate-limited/unavailable
3. Writes successful data to local cache: `qqq_daily_data.csv`

If cache exists, it is reused for faster and reproducible runs.

## Feature Engineering (No Leakage)

Features at day \(t\) use only information available up to \(t\):

- `mom_5` = AdjClose(t) / AdjClose(t-5) - 1
- `vol_20` = rolling std of daily AdjClose returns over last 20 trading days
- `volchg_10` = Volume(t) / SMA10(Volume)(t) - 1
- `rsi_14` = RSI(14) using **Wilder's smoothing**

Target:

- `target_5d_fwd_ret` = AdjClose(t+5) / AdjClose(t) - 1

Rows with missing feature/target values from rolling windows or forward shift are dropped.

## Train/Validation/Test Split

Time-series split (no shuffle):

- **Train:** 2010-01-01 to 2019-12-31
- **Validation:** 2020-01-01 to 2022-12-31
- **Test:** 2023-01-01 to 2025-12-31

Scaling is applied only for Linear Regression via sklearn `Pipeline` (fit on train only).

## Backtest Logic

On test period:

- Compute predictions \(\hat{y}(t)\)
- Compute threshold = 70th percentile of test predictions
- Signal ON when \(\hat{y}(t)\) >= threshold (top 30%)
- Enter long trade at day \(t\), hold for next 5 trading days
- Allow overlapping trades, equal-weight active trade returns daily

Benchmark: buy-and-hold QQQ over the same test period.

## Metrics Reported

Per model on test period:

- `R2`
- `MAE`
- `Rank IC (Spearman)`
- `Strategy CAGR`
- `Sharpe`
- `Max Drawdown`

## Model Analysis (Why LightGBM is Best in This Project)

Based on the latest pipeline outputs (`results.csv` and `data/best_model_details.json`), **LightGBM** is selected as the best model because it achieves the strongest strategy performance on the test window:

- Highest **Strategy CAGR** (~0.2955)
- Highest **Sharpe** (~1.5181)
- Competitive **Max Drawdown** (~-0.2133)

Why this can happen even when regression fit is modest:

- The trading rule uses a **top-30% prediction filter** (70th percentile threshold), so the key objective is ranking/selecting stronger opportunities, not only minimizing global prediction error.
- LightGBM can model non-linear relationships and feature interactions (`mom_5`, `vol_20`, `volchg_10`, `rsi_14`) better than a linear model.
- This often improves decision quality in the prediction tail, which is exactly what the strategy uses.

### Important caveat

The current script chooses the final “best model” by **test-period Strategy CAGR**. This is practical for comparison, but it can introduce selection bias. For stricter evaluation, select the winning model on validation (or walk-forward folds), then run one final test-only report.

## Environment Setup (Python 3.12.12)


```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
cd /Users/dennis/Desktop/src/qqq
python qqq_pipeline.py
```

## Expected Outputs

After a successful run:

- `results.csv` with model/strategy metrics
- `cumulative_returns.png` with strategy vs buy-and-hold performance

## Reproducibility Notes

- Fixed random seed: `SEED = 42`
- Deterministic split without shuffling
- Feature windows are backward-looking only
- Scaling fit on train only (for Linear Regression)
- Fallback + cache support keeps pipeline runnable when Yahoo API is rate-limited
