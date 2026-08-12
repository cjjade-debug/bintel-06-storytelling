# bintel-06-storytelling

[![Workflow Guide](https://img.shields.io/badge/Pro--Guide-pro--analytics--02-green)](https://denisecase.github.io/pro-analytics-02/workflow-b-apply-example-project/)
[![Python 3.14](https://img.shields.io/badge/python-3.14%2B-blue?logo=python)](./pyproject.toml)
[![MIT](https://img.shields.io/badge/license-see%20LICENSE-yellow.svg)](./LICENSE)

> Professional Python project: BI storytelling with smart sales data.

## CJ Jade's Project Description

This project focuses on the sales of 2024, and the sales of the first half of 2025 (current year),
with the goal of investigating, analyzing and forecasting the strongest and weakest regions.

Within this project we will explore the dataset of sales_reporting_cjjade.csv, using both
Python and PowerBI to analyze the current sales data and forecast future sales for June to December of 2025.

## Working Files

We'll work with these areas:

- **data/reporting** - sales_reporting_cjjade.csv
- **docs/** - Project Narrative and Documentation
- **src/bizintel/** - app_cjjade.py, forecast_lightgbm_jun2025_dec2025.py, storytelling_cjjade.py, storytelling_regional_comparison.py,
viz_forecast_interactive.py, & viz_forecast_static.py
- **pyproject.toml** - Update authorship & Links
- **zensical.toml** - Update authorship & Links

## Instructions (pro-analytics-02)

Follow the
[step-by-step workflow guide](https://denisecase.github.io/pro-analytics-02/workflow-b-apply-example-project/)
to complete:

1. Phase 1. **Start & Run**
2. Phase 2. **Change Authorship**
3. Phase 3. **Read & Understand**
4. Phase 4. **Modify**
5. Phase 5. **Apply**

## Project Instructions

# To run this code

- **uv run python -m bizintel.storytelling_cjjade**
- **uv run python -m bizintel.storytelling_regional_comparison** (graphs do not generate but are saved to docs/image)
- **uv run python -m bizintel.forecast_lightgbm_jun2025_dec2025** (This is the file that generates the forecasting .csv file and report, no graphs generate)
- **uv run python -m bizintel.viz_forecast_static** (Forecasting Graph will generate)

## Custom Commands

- New Images (Charts from PowerBI)
- Regional Comparisons of Sales and Trends
- Forecasting of June 2025 - Dec 2025
- Forecasting CSV File
- Forecasting Statsic Charts
- Forecasting Interactive Visualization
- Outputs Folder (includes images and HTML files)

## Command Reference

<details>
<summary>Show command reference</summary>

### In a machine terminal (open in your `Repos` folder)

After you get a copy of this repo in your own GitHub account,
open a machine terminal in your `Repos` folder:

```shell
# Replace username with YOUR GitHub username.
git clone https://github.com/cjjade-debug/bintel-06-storytelling

cd bintel-06-storytelling
code .
```

### In a VS Code terminal

These are listed for convenience.
For best results, follow the detailed instructions in
[pro-analytics-02 guide](https://denisecase.github.io/pro-analytics-02/).

```shell
uv self update
uv python pin 3.14
uv lock --upgrade
uv sync --extra dev --extra docs --upgrade

uvx pre-commit install
uvx pre-commit autoupdate

git add -A
uvx pre-commit run --all-files
# repeat if changes were made
uvx pre-commit run --all-files

# OPTIONAL: run the example module
uv run python -m bizintel.app_cjjade

# TASK 1: run the example storytelling module for an example problem
uv run python -m bizintel.storytelling_cjjade

# TASK 2: run your own storytelling module that looks at a different problem
# add your command in the line below


# run common chores
uv run ruff format .
uv run ruff check . --fix
uv run python -m pyright
uv run python -m pytest
uv run python -m zensical build

# save progress
git add -A
git commit -m "update"
git push -u origin main
```

</details>

## Notes

- Use the **UP ARROW** and **DOWN ARROW** in the terminal to scroll through past commands.
- Use `CTRL+f` to find (and replace) text within a file.
- You do not need to add to or modify `tests/`. They are provided for example only.
- Many files are silent helpers. Explore as you like, but nothing is required.
- You do NOT need to understand everything; understanding builds naturally over time.

## Troubleshooting >>>

If you see something like this in your terminal: `>>>` or `...`
You accidentally started Python interactive mode.
It happens.
Press `Ctrl+c` (both keys together) or `Ctrl+Z` then `Enter` on Windows.

## Findings and Visuals

**Displayed below are the images that CJ Jade generated through Python,**
**the images that were created by PowerBI can be found on the Project Documentation Page**

[Project Documentation Page (Index.md)](https://cjjade-debug.github.io/bintel-06-storytelling/)

![Regional Growth Comparison](./docs/images/regional_growth_comparison_cjjade.png)

![Regional Sales 2024-2025](./docs/images/regional_sales_2024_2025_cjjade.png)

![Regional Year Comparison](./docs/images/regional_year_comparison_cjjade.png)

![Monthly Forecasting](./docs/images/storytelling_monthly_forecast_cjjade_03.png)

## Project Documentation

Additional project instructions, terms, and notes:

[docs/index.md](docs/index.md)

## Citation

[CITATION.cff](./CITATION.cff)

## License

[MIT](./LICENSE)
