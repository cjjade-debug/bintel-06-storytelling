# src/bizintel/viz_forecast_static.py
from pathlib import Path  # noqa: I001
import sys

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore

# Paths (adjust if different)
DATA_CSV = Path("data/reporting/sales_reporting_cjjade.csv")
FORECAST_CSV = Path("forecasts_2025-06_2025-12_lgb.csv")  # or your forecast file


def load_monthly_sales(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["SaleDate"])
    df["SaleAmount"] = pd.to_numeric(df["SaleAmount"], errors="coerce").fillna(0.0)  # pyright: ignore[reportAttributeAccessIssue]
    ts = df.resample("ME", on="SaleDate").SaleAmount.sum().rename("y").to_frame()
    ts.index.name = "ds"
    return ts


def main():
    ts = load_monthly_sales(DATA_CSV)
    if not FORECAST_CSV.exists():
        print(f"Forecast file not found: {FORECAST_CSV}", file=sys.stderr)
        sys.exit(1)
    fc = pd.read_csv(FORECAST_CSV, parse_dates=["ds"])
    fc = fc.set_index("ds")

    # Combine for plotting
    combined = ts.join(fc[["yhat"]], how="outer")
    combined["type"] = combined["y"].notna().map({True: "history"})
    combined.loc[fc.index, "type"] = "forecast"

    # rolling smoothing for visual context (optional)
    combined["rolling_3"] = combined["y"].rolling(3, min_periods=1).mean()

    # plot
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    # history line
    ax.plot(ts.index, ts["y"], label="Historical sales", color="#1f77b4", linewidth=2)
    # forecast line
    ax.plot(
        fc.index,
        fc["yhat"],
        label="Forecast (LightGBM)",
        color="#ff7f0e",
        linewidth=2,
        linestyle="--",
        marker="o",
    )
    # optional rolling
    ax.plot(
        combined.index,
        combined["rolling_3"],
        label="3-mo rolling (history)",
        color="#2ca02c",
        linewidth=1,
        alpha=0.8,
    )

    # vertical line at forecast start
    forecast_start = fc.index.min()
    ax.axvline(forecast_start, color="gray", linestyle=":", linewidth=1)
    ax.text(
        forecast_start,
        ax.get_ylim()[1] * 0.95,
        "Forecast start",
        rotation=90,
        va="top",
        ha="right",
        color="gray",
    )

    ax.set_title("Monthly Sales — Historical and Forecast (Jun–Dec 2025)")
    ax.set_ylabel("Sales (total)")
    ax.set_xlabel("Month (month-end)")
    ax.legend()
    plt.tight_layout()

    out_path = Path("outputs")
    out_path.mkdir(parents=True, exist_ok=True)
    fig_file = out_path / "sales_forecast_2025_jun_dec.png"
    plt.savefig(fig_file, dpi=200)
    print(f"Saved static plot to {fig_file}")
    plt.show()


if __name__ == "__main__":
    main()
