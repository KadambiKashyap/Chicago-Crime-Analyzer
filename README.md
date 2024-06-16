# Chicago Crime Dashboard

This Streamlit application provides a dashboard for analyzing Chicago crime data, allowing users to explore temporal and spatial patterns, view heatmaps, line charts, bar charts, and generate detailed reports.

## Overview

The Chicago Crime Dashboard is built using Streamlit, a Python framework for building interactive web applications. It leverages data visualization techniques to provide insights into crime trends based on user-selected filters such as year, crime type, and weekday.

## Features

- **Temporal and Spatial Patterns:**
  - Displays scatter plots and 2D histograms of crime locations.
  - Allows exploration of crime trends over time.

- **Heatmaps, Line Charts, Bar Charts:**
  - Offers visualizations such as heatmaps of crime occurrences, line charts of rolling sums, and bar charts showing crime counts.
  - Users can select filters like crime type, year, and weekday to customize visualizations.

- **Detailed Report:**
  - Provides findings and recommendations based on analyzed crime data.
  - Aimed at law enforcement and public safety policymakers.

- **Interactive Dashboard:**
  - Future development will integrate real-time monitoring features for crime data.


## Usage

1. **Run the Streamlit App:**
   ```bash
   streamlit run main.py
    ```
2. Use the sidebar filters to select specific years, crime types, and weekdays for customized data views.

3. Each visualization is interactive and updates dynamically based on user selections.


## Data Source

The data used in this application is sourced from [Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2).

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
