# src/bizintel/viz_forecast_interactive.py
from pathlib import Path  # noqa: I001
import sys

import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore

DATA_CSV = Path("data/reporting/sales_reporting_cjjade.csv")
FORECAST_CSV = Path("forecasts_2025-06_2025-12_lgb.csv")


def load_monthly_sales(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["SaleDate"])
    df["SaleAmount"] = pd.to_numeric(df["SaleAmount"], errors="coerce").fillna(0.0)  # pyright: ignore[reportAttributeAccessIssue]
    ts = (
        df.resample("ME", on="SaleDate")
        .SaleAmount.sum()
        .rename("y")
        .to_frame()
        .reset_index()
    )
    return ts


def main():
    ts = load_monthly_sales(DATA_CSV)
    if not FORECAST_CSV.exists():
        print(f"Forecast file not found: {FORECAST_CSV}", file=sys.stderr)
        sys.exit(1)

    fc = pd.read_csv(FORECAST_CSV, parse_dates=["ds"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts["SaleDate"],
            y=ts["y"],
            mode="lines+markers",
            name="Historical",
            line=dict(color="#1f77b4"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc["ds"],
            y=fc["yhat"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#ff7f0e", dash="dash"),
        )
    )

    if "yhat_lower" in fc.columns and "yhat_upper" in fc.columns:
        fig.add_trace(
            go.Scatter(
                x=list(fc["ds"]) + list(fc["ds"][::-1]),
                y=list(fc["yhat_upper"]) + list(fc["yhat_lower"][::-1]),
                fill='toself',
                fillcolor='rgba(255,127,14,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='Forecast interval',
            )
        )

    fig.update_layout(
        title="Monthly Sales — Historical and Forecast (Jun–Dec 2025)",
        xaxis_title="Month (month-end)",
        yaxis_title="Sales",
        template="plotly_white",
        width=1000,
        height=600,
    )

    out_html = Path("outputs") / "sales_forecast_2025_jun_dec.html"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs='cdn')
    print(f"Wrote interactive plot to {out_html}")
    fig.show()


if __name__ == "__main__":
    main()
