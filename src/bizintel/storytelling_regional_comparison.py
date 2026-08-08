"""storytelling_regional_comparison.py - Regional Sales Analysis for 2024-2025.

Compare sales performance across all regions for 2024 and 2025.

It demonstrates how to adapt the storytelling workflow to answer
a different business question while maintaining best practices.

An overall BI project often flows like this:

prepared data
    ↓
data warehouse
    ↓
reporting-ready dataset from the dw
    ↓
analysis and OLAP (slice, dice, roll-up, drill-down)
    ↓
insight
    ↓
story and recommended action <--- this is our final addition

Author: [Your Name]
Date: 2026-08

Storytelling process:

1. Define one clear business question.
2. Use the reporting-ready CSV created earlier.
3. Gather only the data needed to answer the question.
4. Create connected charts.
5. Identify the key results supported by the charts.
6. Write the story and recommendation in docs/index.md.

Business Question:
- How do sales compare across all regions in 2024 vs 2025?
- Which region had the strongest growth?

Data Source:
- data/reporting/sales_reporting_case.csv

Output:
- docs/images/regional_sales_2024_2025.png
- docs/images/regional_growth_comparison.png
- docs/images/regional_year_comparison.png

Terminal command to run this file from the root project folder:

uv run python -m bizintel.storytelling_regional_comparison

OBS:
  This is a modified version of storytelling_case.py
  adapted to compare all regions across 2024-2025.
"""

# === Section 1. Import dependencies and set up constants ===

# === IMPORTS ===

from numbers import Real  # noqa: F401, I001
from pathlib import Path
from typing import Final

from datafun_toolkit.logger import log_path  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore

from bizintel.utils_logger import LOG, log_header  # type: ignore
from bizintel.utils_viz import plot_bar, plot_line  # noqa: F401

# === DECLARE CONSTANTS ===

# Storytelling input folder.
DATA_REPORTING: Final[Path] = Path("data/reporting")

# Storytelling input file (created earlier).
REPORTING_FILE: Final[Path] = DATA_REPORTING / "sales_reporting_case.csv"

# Storytelling charts output folder so they can appear in our narrative.
CHARTS_OUTPUT: Final[Path] = Path("docs/images")

# Chart files shown in docs/index.md.
REGIONAL_SALES_CHART: Final[Path] = CHARTS_OUTPUT / "regional_sales_2024_2025.png"
REGIONAL_GROWTH_CHART: Final[Path] = CHARTS_OUTPUT / "regional_growth_comparison.png"
REGIONAL_YEAR_CHART: Final[Path] = CHARTS_OUTPUT / "regional_year_comparison.png"

# Year range for this analysis.
ANALYSIS_YEARS: Final[list[str]] = ["2024", "2025"]


# === Section 2. Define Reusable Functions ===

# === Section 2.1 LOAD REPORTING DATA FUNCTION ===


def load_reporting_data(file_path: Path) -> pd.DataFrame:
    """Load and verify the reporting-ready data.

    WHY: Storytelling begins with trusted, reporting-ready data.
    The earlier modules created this file from the data warehouse.
    We do not repeat the preparation, warehouse, or ETL work here.

    Args:
        file_path: Path to the reporting-ready CSV file.

    Returns:
        Reporting-ready pandas DataFrame.
    """
    LOG.info("Loading reporting-ready data")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Reporting-ready data file not found: {file_path}. "
            "Run the earlier workflow first."
        )

    df_reporting: pd.DataFrame = pd.read_csv(file_path)

    # Define the set of columns required for this analysis.
    required_columns: set[str] = {
        "YearMonth",  # categorical dimension
        "Region",  # categorical dimension
        "SaleAmount",  # numeric measure
    }

    # COLUMN QUALITY CHECKS.
    missing_columns: set[str] = required_columns - set(df_reporting.columns)

    if missing_columns:
        raise ValueError(
            f"Reporting data is missing required columns: {sorted(missing_columns)}"
        )

    if df_reporting.empty:
        raise ValueError("The reporting data contains no rows.")

    # NUMERIC COLUMN QUALITY CHECKS.
    df_reporting["SaleAmount"] = pd.to_numeric(
        df_reporting["SaleAmount"],
        errors="coerce",
    )

    if df_reporting["SaleAmount"].isna().any():
        raise ValueError("SaleAmount contains missing or nonnumeric values.")

    # Extract year from YearMonth for filtering.
    # YearMonth format is assumed to be "YYYY-MM" or "YYYYMM".
    df_reporting["Year"] = df_reporting["YearMonth"].astype(str).str[:4]

    LOG.info(f"  Loaded {df_reporting.shape[0]} reporting rows")
    LOG.info(f"  Verified {df_reporting.shape[1]} reporting columns")
    return df_reporting


# === Section 2.2 FILTER DATA BY YEARS ===


def filter_by_years(
    df_reporting: pd.DataFrame,
    years: list[str],
) -> pd.DataFrame:
    """Filter reporting data to include only specified years.

    Args:
        df_reporting: Complete reporting-ready sales data.
        years: List of years to include (e.g., ["2024", "2025"]).

    Returns:
        DataFrame filtered to the specified years.
    """
    LOG.info(f"Filtering data for years: {years}")

    df_filtered: pd.DataFrame = df_reporting.loc[
        df_reporting["Year"].isin(years)
    ].copy()  # type: ignore

    if df_filtered.empty:
        raise ValueError(
            f"No sales data found for years {years}. "
            "Update ANALYSIS_YEARS to match available data."
        )

    LOG.info(f"  Filtered to {df_filtered.shape[0]} rows for {len(years)} year(s)")
    return df_filtered


# === Section 2.3 SUMMARIZE SALES BY REGION AND YEAR ===


def summarize_regional_sales_by_year(
    df_filtered: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize total sales by region and year.

    WHY: This comparison shows which regions performed best
    and how each region's performance changed year-over-year.

    Args:
        df_filtered: Sales data filtered to analysis years.

    Returns:
        DataFrame with Region, Year, and TotalSales columns.
    """
    LOG.info("Summarizing regional sales by year")

    df_regional_yearly: pd.DataFrame = (
        df_filtered.groupby(["Region", "Year"], as_index=False)
        .agg(TotalSales=("SaleAmount", "sum"))
        .sort_values(["Year", "TotalSales"], ascending=[True, False])
    )

    df_regional_yearly["TotalSales"] = df_regional_yearly["TotalSales"].round(2)

    LOG.info(f"  Regional-yearly combinations: {df_regional_yearly.shape[0]}")
    return df_regional_yearly


# === Section 2.4 CALCULATE YEAR-OVER-YEAR GROWTH ===


def calculate_yoy_growth(
    df_regional_yearly: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate year-over-year growth by region.

    WHY: Growth rates show which regions are expanding fastest.

    Args:
        df_regional_yearly: Regional sales summarized by year.

    Returns:
        DataFrame with Region, Growth_Percent, and Sales_2024, Sales_2025.
    """
    LOG.info("Calculating year-over-year growth")

    # Pivot to get years as columns for easier comparison.
    df_pivot: pd.DataFrame = df_regional_yearly.pivot(
        index="Region",
        columns="Year",
        values="TotalSales",
    ).reset_index()

    # Handle missing years (if a region has only one year of data).
    for year in ANALYSIS_YEARS:
        if year not in df_pivot.columns:
            df_pivot[year] = 0.0

    # Calculate growth percentage.
    # Growth = ((2025 - 2024) / 2024) * 100
    # Guard against division by zero using numpy.clip
    sales_2024 = pd.Series(df_pivot.get("2024", 0).values).values  # type: ignore
    sales_2025 = pd.Series(df_pivot.get("2025", 0).values).values  # type: ignore

    # Use numpy.clip to ensure no division by zero
    denominator = np.clip(sales_2024, a_min=0.1, a_max=None)

    df_pivot["Growth_Percent"] = (
        ((sales_2025 - sales_2024) / denominator) * 100  # type: ignore
    ).round(2)

    # Sort by growth rate descending.
    df_pivot = df_pivot.sort_values("Growth_Percent", ascending=False)

    LOG.info(f"  Growth rates calculated for {df_pivot.shape[0]} regions")
    return df_pivot


# === Section 2.5 IDENTIFY KEY RESULTS ===


def identify_key_results(
    df_regional_yearly: pd.DataFrame,
    df_growth: pd.DataFrame,
) -> None:
    """Identify and log key factual results from the analysis.

    WHY: Results are values directly supported by the data.
    The complete analytical story is written in docs/index.md
    with both charts.

    Args:
        df_regional_yearly: Regional sales by year.
        df_growth: Year-over-year growth by region.

    Returns:
        None
    """
    LOG.info("Identifying key results")

    # Total sales across all regions and years.
    total_all_sales: float = float(df_regional_yearly["TotalSales"].sum())
    LOG.info(f"  Total sales (all regions, all years): ${total_all_sales:,.2f}")

    # Best performing region in 2025.
    df_2025 = df_regional_yearly[df_regional_yearly["Year"] == "2025"]
    if not df_2025.empty:
        best_region_2025: str = str(df_2025.iloc[0]["Region"])
        best_sales_2025: float = float(df_2025.iloc[0]["TotalSales"])
        LOG.info(f"  Best region in 2025: {best_region_2025} (${best_sales_2025:,.2f})")

    # Fastest growing region.
    fastest_growth_region: str = str(df_growth.iloc[0]["Region"])
    fastest_growth_rate: float = float(df_growth.iloc[0]["Growth_Percent"])
    LOG.info(
        f"  Fastest growing region: {fastest_growth_region} "
        f"({fastest_growth_rate:+.2f}%)"
    )

    # Any declining regions.
    df_declining = df_growth[df_growth["Growth_Percent"] < 0]
    if not df_declining.empty:
        LOG.info(f"  Regions with negative growth: {df_declining.shape[0]}")
        for _, row in df_declining.iterrows():
            LOG.info(f"    - {row['Region']}: {row['Growth_Percent']:+.2f}%")


# === MAIN FUNCTION ===


def main() -> None:
    """Main function to run the regional comparison analysis."""

    log_header(LOG, "BI")

    LOG.info("========================")
    LOG.info("START Regional Comparison Analysis")
    LOG.info("========================")

    log_path(LOG, "Input reporting data from:", REPORTING_FILE)

    # STEP 1: DEFINE ONE CLEAR BUSINESS QUESTION.
    #
    # How do sales compare across all regions in 2024 vs 2025?
    # Which region had the strongest growth?

    LOG.info("CALL a function to load reporting-ready data........")
    df_reporting = load_reporting_data(REPORTING_FILE)

    # STEP 2: FILTER TO ANALYSIS YEARS.
    LOG.info("CALL a function to filter data by year........")
    df_filtered = filter_by_years(df_reporting, ANALYSIS_YEARS)

    # STEP 3: SUMMARIZE SALES BY REGION AND YEAR.
    LOG.info("CALL a function to summarize regional sales by year........")
    df_regional_yearly = summarize_regional_sales_by_year(df_filtered)

    # STEP 4: CALCULATE GROWTH RATES.
    LOG.info("CALL a function to calculate year-over-year growth........")
    df_growth = calculate_yoy_growth(df_regional_yearly)

    # STEP 5: CREATE AND SAVE THE CHARTS.
    CHARTS_OUTPUT.mkdir(parents=True, exist_ok=True)

    # CHART 1: Regional Sales by Year (Grouped Bar Chart).
    LOG.info("CALL a function to plot regional sales by year........")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=df_regional_yearly,
        x="Region",
        y="TotalSales",
        hue="Year",
        ax=ax,
        palette="Set2",
    )
    ax.set_title(
        "Regional Sales Comparison: 2024 vs 2025", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Region", fontsize=12)
    ax.set_ylabel("Total Sales ($)", fontsize=12)
    ax.legend(title="Year")
    plt.tight_layout()
    plt.savefig(REGIONAL_SALES_CHART, bbox_inches="tight", dpi=300)
    log_path(LOG, "Saved regional sales chart:", REGIONAL_SALES_CHART)
    plt.close()

    # CHART 2: Year-over-Year Growth Rate (Bar chart).
    LOG.info("CALL a function to plot growth rates........")
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["green" if x >= 0 else "red" for x in df_growth["Growth_Percent"]]
    ax.bar(df_growth["Region"], df_growth["Growth_Percent"], color=colors, alpha=0.7)
    ax.set_title(
        "Year-over-Year Sales Growth by Region (2024 to 2025)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Region", fontsize=12)
    ax.set_ylabel("Growth Rate (%)", fontsize=12)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(REGIONAL_GROWTH_CHART, bbox_inches="tight", dpi=300)
    log_path(LOG, "Saved growth rates chart:", REGIONAL_GROWTH_CHART)
    plt.close()

    # CHART 3: Regional Monthly Trend (Line chart across all regions).
    LOG.info("CALL a function to plot regional monthly trends........")
    df_monthly_regional = (
        df_filtered.groupby(["YearMonth", "Region"], as_index=False)
        .agg(TotalSales=("SaleAmount", "sum"))
        .sort_values(["YearMonth"])
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    for region in df_monthly_regional["Region"].unique():
        df_region_data = df_monthly_regional[df_monthly_regional["Region"] == region]
        ax.plot(
            df_region_data["YearMonth"],
            df_region_data["TotalSales"],
            marker="o",
            label=region,
            linewidth=2,
        )

    ax.set_title(
        "Monthly Sales Trend by Region (2024-2025)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Total Sales ($)", fontsize=12)
    ax.legend(title="Region", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(REGIONAL_YEAR_CHART, bbox_inches="tight", dpi=300)
    log_path(LOG, "Saved regional monthly trend chart:", REGIONAL_YEAR_CHART)
    plt.close()

    # STEP 6: IDENTIFY THE KEY FACTUAL RESULTS.
    LOG.info("CALL a function to identify key results........")
    identify_key_results(df_regional_yearly, df_growth)

    LOG.info("Regional comparison workflow complete")
    LOG.info("========================")
    LOG.info("Executed successfully!")
    LOG.info("========================")


# === CONDITIONAL EXECUTION GUARD ===

if __name__ == "__main__":
    main()
