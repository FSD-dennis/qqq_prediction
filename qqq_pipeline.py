from __future__ import annotations

import warnings
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:
    raise ImportError(
        "lightgbm is required. Install with: pip install lightgbm"
    ) from exc


SEED = 42
np.random.seed(SEED)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class SplitData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


FEATURE_COLS = ["mom_5", "vol_20", "volchg_10", "rsi_14"]


def download_data(
    ticker: str = "QQQ",
    start: str = "2010-01-01",
    end: str = "2025-12-31",
    cache_path: str = "qqq_daily_data.csv",
) -> pd.DataFrame:
    """Download OHLCV data with retry/cache and fallback source."""
    cache_file = Path(cache_path)

    if cache_file.exists():
        cached = pd.read_csv(cache_file, parse_dates=["Date"], index_col="Date")
        if not cached.empty:
            return cached.sort_index()

    df = pd.DataFrame()
    retry_waits = [2, 5, 10]
    for wait_s in retry_waits:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )
        if not df.empty:
            break
        time.sleep(wait_s)

    # Fallback source if Yahoo is rate-limited/unavailable
    if df.empty:
        fallback_url = "https://stooq.com/q/d/l/?s=qqq.us&i=d"
        fallback = pd.read_csv(fallback_url)
        fallback.columns = [c.strip() for c in fallback.columns]
        rename_map = {
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        }
        fallback = fallback[list(rename_map.keys())].rename(columns=rename_map)
        fallback["Date"] = pd.to_datetime(fallback["Date"])
        fallback = fallback[(fallback["Date"] >= pd.to_datetime(start)) & (fallback["Date"] <= pd.to_datetime(end))]
        fallback = fallback.set_index("Date").sort_index()

        # Stooq does not provide a separate Adj Close column in this endpoint.
        # Use Close as Adj Close in fallback mode.
        fallback["Adj Close"] = fallback["Close"]
        df = fallback

    if df.empty:
        raise ValueError("No data downloaded from primary or fallback source.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    required_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols].copy()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.to_datetime(df.index)

    df.to_csv(cache_file, index=True, index_label="Date")
    return df


def rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe features available at time t."""
    out = df.copy()

    out["ret_1"] = out["Adj Close"].pct_change()
    out["mom_5"] = out["Adj Close"] / out["Adj Close"].shift(5) - 1.0
    out["vol_20"] = out["ret_1"].rolling(window=20, min_periods=20).std()

    vol_sma10 = out["Volume"].rolling(window=10, min_periods=10).mean()
    out["volchg_10"] = out["Volume"] / vol_sma10 - 1.0

    out["rsi_14"] = rsi_wilder(out["Adj Close"], period=14)

    return out


def build_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Build 5-day forward return target y(t)=AdjClose(t+h)/AdjClose(t)-1."""
    out = df.copy()
    out["target_5d_fwd_ret"] = out["Adj Close"].shift(-horizon) / out["Adj Close"] - 1.0
    return out


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Build features+target and drop rows with unavailable values."""
    out = build_features(df)
    out = build_target(out, horizon=5)

    needed_cols = FEATURE_COLS + ["target_5d_fwd_ret", "ret_1", "Adj Close"]
    out = out.dropna(subset=needed_cols).copy()

    if out.index.has_duplicates:
        raise ValueError("Duplicate dates found after cleaning.")

    if out[FEATURE_COLS + ["target_5d_fwd_ret"]].isna().any().any():
        raise ValueError("NaN values remain in features/target after cleaning.")

    return out


def split_data(df: pd.DataFrame) -> SplitData:
    """Time-series split: train/val/test with no shuffle."""
    train_mask = (df.index >= "2010-01-01") & (df.index <= "2019-12-31")
    val_mask = (df.index >= "2020-01-01") & (df.index <= "2022-12-31")
    test_mask = (df.index >= "2023-01-01") & (df.index <= "2025-12-31")

    train = df.loc[train_mask]
    val = df.loc[val_mask]
    test = df.loc[test_mask]

    if train.empty or val.empty or test.empty:
        raise ValueError("One or more split periods are empty. Check data coverage.")

    return SplitData(
        X_train=train[FEATURE_COLS],
        y_train=train["target_5d_fwd_ret"],
        X_val=val[FEATURE_COLS],
        y_val=val["target_5d_fwd_ret"],
        X_test=test[FEATURE_COLS],
        y_test=test["target_5d_fwd_ret"],
    )


def save_training_dataset(
    full_dataset: pd.DataFrame,
    output_path: str = "data/training_dataset.csv",
) -> Path:
    """Save train-period dataset (2010-2019) including features and target."""
    train_mask = (full_dataset.index >= "2010-01-01") & (full_dataset.index <= "2019-12-31")
    train_df = full_dataset.loc[train_mask].copy()

    export_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "ret_1",
        *FEATURE_COLS,
        "target_5d_fwd_ret",
    ]
    train_df = train_df[export_cols]

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_file, index=True, index_label="Date")
    return out_file


def _score_regression(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    rank_ic = pd.Series(y_pred, index=y_true.index).corr(y_true, method="spearman")
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Rank IC (Spearman)": float(rank_ic) if pd.notna(rank_ic) else np.nan,
    }


def train_models(data: SplitData) -> Dict[str, object]:
    """Train three regressors and do lightweight val-based tuning for tree models."""
    models: Dict[str, object] = {}

    # Linear Regression with scaling (scaling fit only on train via pipeline)
    lin = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    lin.fit(data.X_train, data.y_train)
    models["LinearRegression"] = lin

    # Random Forest: small deterministic search on validation MAE
    rf_grid = [
        {"n_estimators": 300, "max_depth": 4, "min_samples_leaf": 5},
        {"n_estimators": 500, "max_depth": 5, "min_samples_leaf": 3},
        {"n_estimators": 700, "max_depth": 6, "min_samples_leaf": 3},
    ]

    best_rf = None
    best_rf_mae = np.inf
    for params in rf_grid:
        candidate = RandomForestRegressor(
            random_state=SEED,
            n_jobs=-1,
            **params,
        )
        candidate.fit(data.X_train, data.y_train)
        val_pred = candidate.predict(data.X_val)
        val_mae = mean_absolute_error(data.y_val, val_pred)
        if val_mae < best_rf_mae:
            best_rf_mae = val_mae
            best_rf = candidate

    models["RandomForest"] = best_rf

    # LightGBM: expanded deterministic search on validation MAE
    lgb_grid = [
        {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
        },
        {
            "n_estimators": 500,
            "learning_rate": 0.02,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 30,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
        },
        {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 6,
            "min_child_samples": 20,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
        },
        {
            "n_estimators": 800,
            "learning_rate": 0.01,
            "num_leaves": 15,
            "max_depth": 6,
            "min_child_samples": 30,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 0.5,
        },
        {
            "n_estimators": 700,
            "learning_rate": 0.015,
            "num_leaves": 31,
            "max_depth": 8,
            "min_child_samples": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
        {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "max_depth": 5,
            "min_child_samples": 40,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
        {
            "n_estimators": 1000,
            "learning_rate": 0.01,
            "num_leaves": 63,
            "max_depth": 10,
            "min_child_samples": 25,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
        },
        {
            "n_estimators": 600,
            "learning_rate": 0.02,
            "num_leaves": 31,
            "max_depth": 7,
            "min_child_samples": 50,
            "subsample": 1.0,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
    ]

    best_lgb = None
    best_lgb_mae = np.inf
    for params in lgb_grid:
        candidate = LGBMRegressor(
            random_state=SEED,
            objective="regression",
            n_jobs=-1,
            **params,
        )
        candidate.fit(data.X_train, data.y_train)
        val_pred = candidate.predict(data.X_val)
        val_mae = mean_absolute_error(data.y_val, val_pred)
        if val_mae < best_lgb_mae:
            best_lgb_mae = val_mae
            best_lgb = candidate

    models["LightGBM"] = best_lgb
    return models


def backtest_strategy(
    test_prices: pd.Series,
    test_predictions: pd.Series,
    hold_days: int = 5,
    top_quantile: float = 0.70,
) -> Tuple[pd.Series, Dict[str, float], pd.Series, pd.Series]:
    """
    Backtest long-only strategy with overlapping 5-day trades and equal-weight active trades.

    Signal rule (one-asset interpretation of top 30%):
    - Global threshold on test predictions = 70th percentile over all test dates.
    - Signal ON at date t if pred(t) >= threshold.
    - Enter a new trade at t, hold for next `hold_days` daily returns (t+1..t+hold_days).
    - Daily portfolio return = average of active trade daily returns on that day.
    """
    test_prices = test_prices.copy()
    test_predictions = test_predictions.copy()

    test_ret_1d = test_prices.pct_change().fillna(0.0)
    threshold = float(test_predictions.quantile(top_quantile))
    entry_signal = (test_predictions >= threshold).astype(int)

    active_trade_returns: Dict[pd.Timestamp, List[float]] = {d: [] for d in test_prices.index}
    idx = test_prices.index

    for i, dt in enumerate(idx):
        if entry_signal.loc[dt] != 1:
            continue

        for k in range(1, hold_days + 1):
            j = i + k
            if j >= len(idx):
                break
            day = idx[j]
            active_trade_returns[day].append(float(test_ret_1d.iloc[j]))

    strategy_ret = pd.Series(index=idx, dtype=float)
    strategy_ret.iloc[0] = 0.0
    for d in idx[1:]:
        trades = active_trade_returns[d]
        strategy_ret.loc[d] = float(np.mean(trades)) if len(trades) > 0 else 0.0

    strategy_curve = (1.0 + strategy_ret).cumprod()

    # Buy-and-hold benchmark over same test window
    buyhold_ret = test_ret_1d.copy()
    buyhold_curve = (1.0 + buyhold_ret).cumprod()

    ann_factor = 252
    n_days = len(strategy_ret)
    years = n_days / ann_factor if n_days > 0 else np.nan

    total_return = strategy_curve.iloc[-1] - 1.0
    cagr = (strategy_curve.iloc[-1] ** (1 / years) - 1.0) if years and years > 0 else np.nan

    daily_mean = strategy_ret.mean()
    daily_std = strategy_ret.std(ddof=0)
    sharpe = (np.sqrt(ann_factor) * daily_mean / daily_std) if daily_std > 0 else np.nan

    running_max = strategy_curve.cummax()
    drawdown = strategy_curve / running_max - 1.0
    max_dd = float(drawdown.min())

    stats = {
        "Strategy Total Return": float(total_return),
        "Strategy CAGR": float(cagr),
        "Strategy Sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "Strategy Max Drawdown": max_dd,
        "Signal Threshold (70th pct)": threshold,
        "Signal Days": int(entry_signal.sum()),
    }

    return strategy_curve, stats, buyhold_curve, strategy_ret


def evaluate_metrics(
    models: Dict[str, object],
    data: SplitData,
    full_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate regression + strategy metrics for each model on test set."""
    rows = []

    test_index = data.X_test.index
    test_prices = full_df.loc[test_index, "Adj Close"]

    for name, model in models.items():
        pred_test = model.predict(data.X_test)
        reg_metrics = _score_regression(data.y_test, pred_test)

        pred_series = pd.Series(pred_test, index=test_index, name="pred")
        _, bt_stats, _, _ = backtest_strategy(
            test_prices=test_prices,
            test_predictions=pred_series,
            hold_days=5,
            top_quantile=0.70,
        )

        rows.append(
            {
                "Model": name,
                "R2": reg_metrics["R2"],
                "MAE": reg_metrics["MAE"],
                "Rank IC (Spearman)": reg_metrics["Rank IC (Spearman)"],
                "Strategy CAGR": bt_stats["Strategy CAGR"],
                "Sharpe": bt_stats["Strategy Sharpe"],
                "Max Drawdown": bt_stats["Strategy Max Drawdown"],
            }
        )

    results = pd.DataFrame(rows).sort_values(by="Strategy CAGR", ascending=False)
    return results


def plot_curves(
    strategy_curve: pd.Series,
    buyhold_curve: pd.Series,
    title: str = "QQQ Test Period: Strategy vs Buy-and-Hold",
    out_path: str = "cumulative_returns.png",
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(strategy_curve.index, strategy_curve.values, label="Strategy (Top 30% Pred)", linewidth=2)
    plt.plot(buyhold_curve.index, buyhold_curve.values, label="Buy & Hold QQQ", linewidth=2, alpha=0.8)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Growth of $1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_best_model_details(
    best_model_name: str,
    best_model: object,
    results: pd.DataFrame,
    output_path: str = "data/best_model_details.json",
) -> Path:
    """Save best model metrics and key parameters to a separate JSON file."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    best_row = results.loc[results["Model"] == best_model_name].iloc[0].to_dict()

    params = best_model.get_params() if hasattr(best_model, "get_params") else {}
    params = {
        k: v
        for k, v in params.items()
        if isinstance(v, (str, int, float, bool, type(None)))
    }

    payload = {
        "best_model": best_model_name,
        "selection_criterion": "highest Strategy CAGR on test set",
        "metrics": best_row,
        "parameters": params,
    }

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out_file


def run_pipeline() -> None:
    raw = download_data("QQQ", "2010-01-01", "2025-12-31")
    dataset = prepare_dataset(raw)
    splits = split_data(dataset)
    training_dataset_path = save_training_dataset(dataset, output_path="data/training_dataset.csv")

    models = train_models(splits)
    results = evaluate_metrics(models, splits, dataset)

    # Strategy curve for best model by Strategy CAGR
    best_model_name = results.iloc[0]["Model"]
    best_model = models[best_model_name]
    best_model_details_path = save_best_model_details(
        best_model_name=best_model_name,
        best_model=best_model,
        results=results,
        output_path="data/best_model_details.json",
    )

    test_index = splits.X_test.index
    pred_best = pd.Series(best_model.predict(splits.X_test), index=test_index)

    strategy_curve, bt_stats, buyhold_curve, _ = backtest_strategy(
        test_prices=dataset.loc[test_index, "Adj Close"],
        test_predictions=pred_best,
        hold_days=5,
        top_quantile=0.70,
    )

    results_path = Path("results.csv")
    results.to_csv(results_path, index=False)

    plot_curves(strategy_curve, buyhold_curve, out_path="cumulative_returns.png")

    print("\n=== Model Results (Test 2023-2025) ===")
    print(results.to_string(index=False))

    print("\n=== Best Model Backtest Details ===")
    print(f"Best model: {best_model_name}")
    for k, v in bt_stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print("\nSaved outputs:")
    print("- results.csv")
    print("- cumulative_returns.png")
    print(f"- {training_dataset_path.as_posix()}")
    print(f"- {best_model_details_path.as_posix()}")


if __name__ == "__main__":
    run_pipeline()
