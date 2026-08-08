#!/usr/bin/env python3
"""
Robust recursive monthly forecast using LightGBM with dynamic lag selection and fallback.
Usage:
  python src/bizintel/forecast_lightgbm_jun2025_dec2025.py \
    --csv data/reporting/sales_reporting_cjjade.csv \
    --start 2025-06 --end 2025-12 \
    --out forecasts_2025-06_2025-12_lgb.csv
"""

from abc import ABC  # noqa: I001
import argparse
import sys

import joblib  # type: ignore
import lightgbm as lgb  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/reporting/sales_reporting_cjjade.csv")
    p.add_argument("--start", default="2025-06", help="Forecast start month YYYY-MM")
    p.add_argument("--end", default="2025-12", help="Forecast end month YYYY-MM")
    p.add_argument("--out", default="forecasts_2025-06_2025-12_lgb.csv")
    return p.parse_args()


def load_monthly(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["SaleDate"])
    df["SaleAmount"] = pd.to_numeric(df["SaleAmount"], errors="coerce").fillna(0.0)  # type: ignore
    df["SaleDate"] = pd.to_datetime(df["SaleDate"])
    ts = df.resample("ME", on="SaleDate").SaleAmount.sum().rename("y").to_frame()
    ts.index.name = "ds"
    return ts


def make_features_from_series(series, lags):
    X = pd.DataFrame(index=series.index)
    X["y"] = series
    for lag in lags:
        X[f"lag_{lag}"] = series.shift(lag)
    X["rolling_3"] = series.shift(1).rolling(3).mean()
    X["rolling_6"] = series.shift(1).rolling(6).mean()
    X["rolling_12"] = series.shift(1).rolling(12).mean()
    X["month"] = series.index.month
    X["year"] = series.index.year
    return X


class candidate_lags(ABC):  # noqa: B024
    def __init__(self, candidate_lag_values=(1, 2, 3, 6, 12)):
        self.candidate_lag_values = tuple(candidate_lag_values)

    def choose(self, n_months):
        if n_months < 0:
            raise ValueError("n_months must be non-negative")
        chosen = self._valid_lags(n_months)
        if chosen:
            return chosen
        return self._fallback_lags(n_months)

    def _valid_lags(self, n_months):
        max_possible = max(0, n_months - 1)
        return [lag for lag in self.candidate_lag_values if lag <= max_possible]

    def _fallback_lags(self, n_months):
        return [1] if n_months >= 2 else []


def choose_lags(n_months, candidate_lag_values=(1, 2, 3, 6, 12)):
    chooser = candidate_lags(candidate_lag_values)
    return chooser.choose(n_months)


def train_model(ts, chosen_lags):
    if not chosen_lags:
        return None  # signal to caller: not enough history to build lags
    X = make_features_from_series(ts["y"], lags=chosen_lags)
    X_train = X.dropna().copy()
    if X_train.shape[0] < 2:
        # not enough rows to train
        return None
    y_train = X_train.pop("y")
    dtrain = lgb.Dataset(X_train, label=y_train)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbosity": -1,
        "seed": 42,
    }
    bst = lgb.train(params, dtrain, num_boost_round=500)
    return bst


def make_feature_row_for_date(target_date, history_series, chosen_lags):
    row = {}
    for lag in chosen_lags:
        lag_date = pd.to_datetime(target_date) - pd.tseries.offsets.MonthEnd(lag)
        row[f"lag_{lag}"] = history_series.get(lag_date, np.nan)
    # rolling windows reference up to 12 previous months (fill may occur later)
    prev_index = pd.date_range(
        end=(pd.to_datetime(target_date) - pd.tseries.offsets.MonthEnd(1)),
        periods=12,
        freq="ME",
    )
    prev_vals = history_series.reindex(prev_index)
    row["rolling_3"] = prev_vals[-3:].mean()
    row["rolling_6"] = prev_vals[-6:].mean()
    row["rolling_12"] = prev_vals[-12:].mean()
    row["month"] = pd.to_datetime(target_date).month
    row["year"] = pd.to_datetime(target_date).year
    return pd.Series(row)


def recursive_forecast(model, ts, start_ym, end_ym, chosen_lags):
    last_obs = ts.index.max()
    start_month_end = pd.to_datetime(f"{start_ym}-01") + pd.offsets.MonthEnd(0)
    end_month_end = pd.to_datetime(f"{end_ym}-01") + pd.offsets.MonthEnd(0)
    history = ts["y"].copy()
    first_future = last_obs + pd.offsets.MonthEnd(1)
    start = max(first_future, start_month_end)
    future_index = pd.date_range(start=start, end=end_month_end, freq="ME")
    results = []
    for dt in future_index:
        if model is None:
            # fallback: simple rule — repeat last observed value (or mean if last is NaN)
            last_val = history.dropna().iloc[-1] if history.dropna().size > 0 else 0.0
            yhat = float(last_val)
        else:
            feat = make_feature_row_for_date(dt, history, chosen_lags)
            if feat.isnull().any():
                feat = feat.fillna(history.mean())
            X_row = feat.to_frame().T
            yhat = float(model.predict(X_row)[0])
        history.loc[pd.to_datetime(dt)] = yhat
        results.append({"ds": pd.to_datetime(dt), "yhat": yhat})
    return pd.DataFrame(results)


def main():
    args = parse_args()
    ts = load_monthly(args.csv)
    if ts.empty:
        raise SystemExit(
            "No data found after aggregation. Check CSV and SaleDate column."
        )
    n_months = len(ts)
    print(
        f"Loaded monthly series with {n_months} months: {ts.index.min().date()} -> {ts.index.max().date()}",
        file=sys.stderr,
    )
    chosen_lags = choose_lags(n_months)
    print(f"Chosen lags based on history length: {chosen_lags}", file=sys.stderr)
    model = train_model(ts, chosen_lags)
    if model is None:
        print(
            "Not enough data to train LightGBM with the chosen lags; falling back to simple last-value forecasts.",
            file=sys.stderr,
        )
    else:
        print("LightGBM trained successfully.", file=sys.stderr)
        joblib.dump(model, "lgb_model_monthly.joblib")
    forecast_df = recursive_forecast(model, ts, args.start, args.end, chosen_lags)
    forecast_df.to_csv(args.out, index=False)
    print(f"Wrote {len(forecast_df)} rows to {args.out}")
    print(forecast_df.to_string(index=False))


if __name__ == "__main__":
    main()
