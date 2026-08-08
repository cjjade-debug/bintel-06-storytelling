#!/usr/bin/env python3
"""
forecast_prophet_jun2025_dec2025.py

Usage:
  python forecast_prophet_jun2025_dec2025.py \
    --csv data/reporting/sales_reporting_cjjade.csv \
    --start 2025-06 --end 2025-12 \
    --out forecasts_2025-06_2025-12.csv

Defaults: start=2025-06, end=2025-12
"""

import argparse  # noqa: I001
import pandas as pd  # type: ignore
import numpy as np  # type: ignore  # noqa: F401
from prophet import Prophet  # type: ignore
import sys
from pandas.tseries.offsets import MonthEnd  # type: ignore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        default="data/reporting/sales_reporting_cjjade.csv",
        help="Path to sales CSV",
    )
    p.add_argument("--start", default="2025-06", help="Forecast start month YYYY-MM")
    p.add_argument("--end", default="2025-12", help="Forecast end month YYYY-MM")
    p.add_argument(
        "--out",
        default="forecasts_2025-06_2025-12.csv",
        help="Output CSV for forecasts",
    )
    p.add_argument(
        "--freq",
        default="M",
        choices=["D", "M"],
        help="Frequency: 'D' for daily, 'M' for monthly",
    )
    return p.parse_args()


def load_and_agg(csv_path, freq="M"):
    df = pd.read_csv(csv_path, parse_dates=["SaleDate"])
    # ensure numeric
    df["SaleAmount"] = pd.to_numeric(df["SaleAmount"], errors="coerce").fillna(0.0)  # pyright: ignore[reportAttributeAccessIssue]
    df["SaleDate"] = pd.to_datetime(df["SaleDate"])
    if freq == "M":
        ts = df.resample("M", on="SaleDate").SaleAmount.sum().reset_index()
    else:
        ts = df.resample("D", on="SaleDate").SaleAmount.sum().reset_index()
    ts = ts.rename(columns={"SaleDate": "ds", "SaleAmount": "y"})
    return ts


def make_future_dates(last_obs, start_ym, end_ym, freq="M"):
    # parse YYYY-MM to month-end timestamp
    start_month_end = pd.to_datetime(f"{start_ym}-01") + MonthEnd(0)
    end_month_end = pd.to_datetime(f"{end_ym}-01") + MonthEnd(0)
    # if last_obs >= start_month_end we still create range starting after last_obs
    first_future = (
        (last_obs + pd.offsets.MonthEnd(1))
        if freq == "M"
        else (last_obs + pd.Timedelta(days=1))
    )
    start = max(first_future, start_month_end)
    return pd.date_range(
        start=start, end=end_month_end, freq="M" if freq == "M" else "D"
    )


def main():
    args = parse_args()
    ts = load_and_agg(args.csv, freq=args.freq)

    if ts.empty:
        print(
            "No data after aggregation — check CSV and SaleDate column.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Fit on all available data
    m = Prophet(
        yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False
    )
    # If you have campaign flags or other regressors, add them here: m.add_regressor('CampaignFlag')
    m.fit(ts)

    last_obs = ts["ds"].max()
    future_dates = make_future_dates(last_obs, args.start, args.end, freq=args.freq)
    if len(future_dates) == 0:
        print(
            "No future months generated. Check the start/end and last date in the data.",
            file=sys.stderr,
        )
        sys.exit(1)

    future_df = pd.DataFrame({"ds": future_dates})
    forecast = m.predict(future_df)

    # keep only requested months
    out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    # Add a helpful column with predicted total (rounded)
    out["yhat"] = out["yhat"].round(2)
    out["yhat_lower"] = out["yhat_lower"].round(2)
    out["yhat_upper"] = out["yhat_upper"].round(2)

    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} forecast rows to {args.out}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
