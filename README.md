# COVID19-Case-Trend

Data analysis and visualization project exploring global COVID-19 confirmed case trends. This assignment focuses on data preprocessing, aggregation, and both static and interactive visualizations to compare case trajectories across countries and over time.

## Overview

This project analyzes global COVID-19 confirmed case data to examine temporal trends and cross-country differences. It emphasizes robust data cleaning, error handling, and visualization design using both Matplotlib and Plotly.

## Features

- Data preprocessing: cleaning malformed rows and invalid dates, handling missing and negative values, merging duplicate country columns, and converting wide-format data to long-format
- Static visualizations (Matplotlib): daily case scatter plots by country and time range, horizontal bar chart of total cases by country, custom axis formatting and annotations
- Interactive visualizations (Plotly): interactive time-series trends by country, dropdown-based country comparison, hover tooltips, and zooming

## Dataset

- global_confirmed_cases.csv — Global confirmed COVID-19 case counts by country and date

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Plotly

## Project Files

- akto2022-assignment1.py — Data preprocessing and visualization pipeline
- covid_case_trend.py — Visualization functions and analysis logic
- global_confirmed_cases.csv — COVID-19 dataset
- README.md

## How to Run

1. Install dependencies:
   pip install pandas numpy matplotlib plotly
2. Run the analysis script:
   python akto2022-assignment1.py
3. Interactive Plotly figures will open in your browser or notebook environment.

## Notes

- Some visualizations may take time to render due to dataset size.
- Date ranges and countries can be customized within the script.

## Author

Aiko Kato  
Pomona College — Computer Science & Digital Media Studies  
CS181DV: Interactive Data Visualization
