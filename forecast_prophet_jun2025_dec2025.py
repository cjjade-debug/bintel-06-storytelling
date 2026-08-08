#!/usr/bin/env python3
"""
Recursive monthly forecast using LightGBM.
Produces forecasts for YYYY-MM range (month granularity, end-of-month timestamps).

Usage:
  python src/bizintel/forecast_lightgbm_jun2025_dec2025.py \
    --csv data/reporting/sales_reporting_cjjade.csv \
    --start 2025-06 --end 2025-12 \
    --out forecasts_2025-06_2025-12_lgb.csv
"""

import argparse  # noqa: I001
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import lightgbm as lgb  # type: ignore
import joblib  # type: ignore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/reporting/sales_reporting_cjjade.csv")
    p.add_argument("--start", default="2025-06", help="Forecast start month YYYY-MM")
    p.add_argument("--end", default="2025-12", help="Forecast end month YYYY-MM")
    p.add_argument("--out", default="forecasts_2025-06_2025-12_lgb.csv")
    return p.parse_args()


def load_monthly(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["SaleDate"])
    df["SaleAmount"] = pd.to_numeric(df["SaleAmount"], errors="coerce").fillna(0.0)  # pyright: ignore[reportAttributeAccessIssue]
    df["SaleDate"] = pd.to_datetime(df["SaleDate"])
    ts = df.resample("M", on="SaleDate").SaleAmount.sum().rename("y").to_frame()
    ts.index.name = "ds"
    return ts


def make_features_from_series(series, lags=(1, 2, 3, 6, 12)):
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


def train_model(ts):
    X = make_features_from_series(ts["y"])
    X_train = X.dropna().copy()
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
    bst = lgb.train(params, dtrain, num_boost_round=500, verbose_eval=False)
    return bst


def make_feature_row_for_date(target_date, history_series, lags=(1, 2, 3, 6, 12)):
    target_date = pd.to_datetime(target_date)
    row = {}
    for lag in lags:
        lag_date = target_date - pd.tseries.offsets.MonthEnd(lag)
        row[f"lag_{lag}"] = history_series.get(lag_date, np.nan)
    prev_index = pd.date_range(
        end=(target_date - pd.tseries.offsets.MonthEnd(1)),
        periods=12,
        freq="M",
    )
    prev_vals = history_series.reindex(prev_index)
    row["rolling_3"] = prev_vals[-3:].mean()
    row["rolling_6"] = prev_vals[-6:].mean()
    row["rolling_12"] = prev_vals[-12:].mean()
    row["month"] = target_date.month
    row["year"] = target_date.year
    return pd.Series(row)


def recursive_forecast(model, ts, start_ym, end_ym):
    last_obs = ts.index.max()
    start_month_end = pd.to_datetime(f"{start_ym}-01") + pd.offsets.MonthEnd(0)
    end_month_end = pd.to_datetime(f"{end_ym}-01") + pd.offsets.MonthEnd(0)
    history = ts["y"].copy()
    first_future = last_obs + pd.offsets.MonthEnd(1)
    start = max(first_future, start_month_end)
    future_index = pd.date_range(start=start, end=end_month_end, freq="M")
    results = []
    for dt in future_index:
        feat = make_feature_row_for_date(dt, history)
        if feat.isnull().any():
            feat = feat.fillna(history.mean())
        X_row = feat.to_frame().T
        yhat = model.predict(X_row)[0]
        history.loc[pd.to_datetime(dt)] = yhat
        results.append({"ds": pd.to_datetime(dt), "yhat": float(yhat)})
    return pd.DataFrame(results)


def main():
    args = parse_args()
    ts = load_monthly(args.csv)
    if ts.empty:
        raise SystemExit(
            "No data found after aggregation. Check CSV and SaleDate column."
        )
    model = train_model(ts)
    joblib.dump(model, "lgb_model_monthly.joblib")
    forecast_df = recursive_forecast(model, ts, args.start, args.end)
    forecast_df.to_csv(args.out, index=False)
    print(f"Wrote {len(forecast_df)} rows to {args.out}")
    print(forecast_df.to_string(index=False))


if __name__ == "__main__":
    main()
