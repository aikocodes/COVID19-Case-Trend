"""
    CS181DV Assignment 1: Fundamentals with Matplotlib and Plotly

    Author: AIKO KATO

    Date: 02/09/2025
    
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.graph_objects as go
import plotly.express as px
from typing import List

# =================================================
# Task 1: Data Preprocessing 
# =================================================

def load_covid_data(filename: str) -> pd.DataFrame:
    """
    Load and preprocess the COVID-19 data.

    Args:
        filename (str): Path to CSV file

    Returns:
        tuple: A tuple containing cleaned DataFrame, daily totals, country-wise pivot table, and rolling average statistics
    """
    # Load the dataset
    try:
        df = pd.read_csv(filename)
    # If the file is not found or the CSV file has a parsing error, raise a ValueError
    except (FileNotFoundError, pd.errors.ParserError):
        raise ValueError(f"Error: Unable to load '{filename}'. Check if the file exists and is correctly formatted.")

    # Drop the 'Province/State' row to clean
    df = df[df.iloc[:, 0] != "Province/State"].reset_index(drop = True)

    # Rename the first column to 'Date'
    df.rename(columns = {df.columns[0]: "Date"}, inplace = True)
    
    # Ensure "Date" column exists, and if missing, raise a ValueError with a clear error message
    if "Date" not in df.columns:
        raise ValueError("Error: 'Date' column not found in dataset.")

    # Ensure dates are formatted properly before processing
    df["Date"] = pd.to_datetime(df["Date"], format = "%m/%d/%y", errors = "coerce")
    
    # Drop any rows where "Date" is still NaT (invalid dates)
    df.dropna(subset = ["Date"], inplace = True)

    # Apply cleaning function
    df_cleaned = clean_data(df)

    # Apply formatting and aggregation function
    daily_totals, country_pivot, rolling_avg = format_aggregate(df_cleaned)

    # Return all processed data
    return df_cleaned, daily_totals, country_pivot, rolling_avg


def clean_data(df):
    """
    Clean the dataset by handling missing values, negative values, and merging duplicate country columns.

    Args:
        df (pd.DataFrame): Raw COVID-19 dataset

    Returns:
        pd.DataFrame: Cleaned dataset with fixed missing values and merged country data
    """
    # Create a copy to preserve the original data
    df_clean = df.copy()

    # Convert all date columns to numeric not to crash the program
    date_columns = df_clean.columns[1:]  # Skipping 'Date'
    df_clean[date_columns] = df_clean[date_columns].apply(pd.to_numeric, errors = 'coerce')

    # Merge duplicate country columns by summing them
    df_clean = df_clean.T.groupby(level=0).sum().T  # I used ChatGPT for this line

    # Fill negative values with median per country
    for col in date_columns:
        # Check if the column exists in the cleaned dataframe before processing
        if col not in df_clean.columns:
            continue  # Skip non-existent columns
        median_value = df_clean[col].median()  # Compute the median for the entire column (country's data)
        df_clean[col] = df_clean[col].apply(lambda x: median_value if x < 0 else x)  # Replace negative numbers with median values

    # Return cleaned data
    return df_clean


def format_aggregate(df):
    """
    Format and aggregate the data to make it suitable for time-series analysis.

    Args:
        df (pd.DataFrame): Cleaned dataset.

    Returns:
        tuple: Daily totals, pivoted country data, and rolling averages.
    """
    # Ensure necessary columns exist
    if "Date" not in df.columns:
        raise ValueError("Error: 'Date' column missing after cleaning.")
    
    # Convert wide-format data into long-format using melt
    df_melted = df.melt(id_vars=["Date"], var_name="Country/Region", value_name="Cases")  # I used ChatGPT for this line

    # Standardize country names before aggregation
    df_melted["Country/Region"] = df_melted["Country/Region"].str.replace(r"\.\d+", "", regex=True)  # I used ChatGPT for this line (I added this after having an error for Task3)

    # Ensure "Date" column is correctly formatted before further processing
    df_melted["Date"] = pd.to_datetime(df_melted["Date"], format = "%Y-%m-%d", errors = "coerce")

    # Aggregate total cases by country and date
    daily_totals = df_melted.groupby(["Date", "Country/Region"], as_index = False)["Cases"].sum()

    # Pivot the data to create a country-by-date matrix, useful for comparison
    country_pivot = daily_totals.pivot(index = "Date", columns = "Country/Region", values = "Cases")

    # Calculate weekly rolling averages and ensure long format output
    rolling_avg = (daily_totals.groupby(["Country/Region", "Date"])["Cases"].rolling(window = 7, min_periods = 1).mean().reset_index())

    # Return all the stats
    return daily_totals, country_pivot, rolling_avg


# Load the dataset
filename = "global_confirmed_cases.csv"
covid_data, daily_totals, country_pivot, rolling_avg = load_covid_data(filename)


# =================================================
# Task 2: Matplotlib - Daily Cases Scatter Plot
# =================================================

def create_daily_cases_scatter(data: pd.DataFrame, countries: list, start_date: str, end_date: str) -> plt.Figure:
    """
    Create scatter plot of daily COVID-19 cases.

    Args:
        data (pd.DataFrame): Processed COVID-19 data
        countries (list): List of countries to include in the plot
        start_date (str): Starting date (YYYY-MM-DD format)
        end_date (str): Ending date (YYYY-MM-DD format)
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    # Get a set of all unique country names available in the dataset
    available_countries = set(data["Country/Region"].unique())
    
    # Identify any countries from the input list that are not present in the dataset
    invalid_countries = [c for c in countries if c not in available_countries]
    
    # If there are any invalid (missing) countries, display a warning message
    if invalid_countries:
        print(f"Warning: The following countries were not found in the dataset: {invalid_countries}")

    # Filter data based on the given date range
    filtered_data = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)]

    # Create a scatter plot for each selected country
    plt.figure(figsize = (12, 6))
    
    # Loop through each country in the given list of selected countries
    for country in countries:
        # Check if the country exists in the dataset to avoid errors
        if country in available_countries:
            # Filter the dataset to include only rows where "Country/Region" matches the current country
            country_data = filtered_data[filtered_data["Country/Region"] == country]
            # Create a scatter plot for the selected country's data
            plt.scatter(country_data["Date"], country_data["Cases"], label = country, alpha = 0.6)

    # Formatting the plot
    plt.xlabel("Date", fontsize = 12)
    plt.ylabel("Daily Cases", fontsize = 12)
    plt.title("Daily COVID-19 Cases by Country", fontsize = 14)
    plt.xticks(rotation = 45)
    plt.grid(True, linestyle = "--", alpha = 0.5)
    plt.legend(title = "Countries", fontsize = 10, loc = "upper left")

    # Show the plot
    plt.show()


# Load the dataset
filename = "global_confirmed_cases.csv"
covid_data, daily_totals, country_pivot, rolling_avg = load_covid_data(filename)


# Define parameters for the scatter plot
selected_countries = ["US", "India", "Brazil", "Russia", "Japan"]
start_date = "2020-03-01"
end_date = "2021-03-01"


# Generate the scatter plot
create_daily_cases_scatter(daily_totals, selected_countries, start_date, end_date)


# =================================================
# Task 3: Matplotlib - Total Cases Bar Chart
# =================================================

def create_total_cases_bar(data: pd.DataFrame, n_countries: int = 10) -> plt.Figure:
    """
    Create bar chart of total cases by country.

    Args:
        data (pd.DataFrame): Processed COVID-19 dataset
        n_countries (int): Number of top countries to display
    
    Returns:
        plt.Figure: Matplotlib figure
    """
    # Aggregate total cases by country, summing cases for duplicate country entries
    total_cases = data.groupby("Country/Region", as_index = False)["Cases"].sum()

    # Ensure the dataset contains enough countries to display
    if len(total_cases) < n_countries:
        print(f"Warning: Only {len(total_cases)} countries available. Adjusting the number of countries shown.")
        n_countries = len(total_cases)  # Avoids IndexError
    
    # Ensure "Cases" column is numeric before sorting
    if not np.issubdtype(total_cases["Cases"].dtype, np.number):
        # Raise an error if the data type of the "Cases" column is not numeric
        raise ValueError("Error: 'Cases' column contains non-numeric data.")
    
    # Select the top n_countries with the highest total cases
    top_countries = total_cases.nlargest(n_countries, "Cases")

    # Create the bar chart
    plt.figure(figsize = (12, 6))
    plt.barh(top_countries["Country/Region"], top_countries["Cases"], color = 'red')

    # Add labels to each bar
    for index, value in enumerate(top_countries["Cases"]):
        plt.text(value, index, f'{int(value):,}', va = 'center')

    # Formatting the plot
    plt.xlabel("Total Cases", fontsize = 12)
    plt.ylabel("Country", fontsize = 12)
    plt.title("Top 10 Countries with Highest COVID-19 Cases", fontsize = 14)
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest case on top
    plt.grid(axis = 'x', linestyle = "--", alpha = 0.5)
    
    # Explicitly set y-tick labels to avoid additional numbering
    plt.yticks(ticks = range(len(top_countries)), labels = top_countries["Country/Region"].tolist())
    
    # Set x-axis limit to prevent text overlap with grid lines
    plt.xlim(0, top_countries["Cases"].max() * 1.2)  # I used ChatGPT for this line
    
    # Format x-axis to display numbers in Millions (M) instead of scientific notation
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x / 1e6:.0f}M'))  # I used ChatGPT for this line
    
    # Show the plot
    plt.show()


# Run the updated function to merge duplicate country names and remove numbering issues
create_total_cases_bar(daily_totals)


# =================================================
# Task 4: Interactive Cases Trend (Plotly)
# =================================================

def create_interactive_trends(data: pd.DataFrame, countries: List[str]) -> go.Figure:
    """
    Create interactive time series plot.

    Args: 
        data (pd.DataFrame): Processed COVID-19 dataset
        countries (list): List of countries to include in the plot

    Returns:
        go.Figure: Plotly figure
    """
    # Extract all unique country names from the dataset and store them in a set
    available_countries = set(data["Country/Region"].unique())
    
    # Identify any requested countries that are not present in the dataset
    invalid_countries = [c for c in countries if c not in available_countries]
    
    # If there are any invalid (missing) countries, display a warning message
    if invalid_countries:
        print(f"Warning: The following countries were not found: {invalid_countries}")
        
    # Filter only existing countries
    valid_countries = [c for c in countries if c in available_countries]
    
    # Raise an error if no valid countries are found in the dataset
    if not valid_countries:
        raise ValueError("Error: No valid countries were found in the dataset.")
    
    # Initialize an empty Plotly Figure object
    fig = go.Figure()
    
    # Add a trace for each selected country
    for country in countries:
        country_data = data[data["Country/Region"] == country]
        fig.add_trace(go.Scatter(
            x = country_data["Date"],
            y = country_data["Cases"],
            mode = 'lines',
            name = country,
            hoverinfo = 'x+y'
        ))
    
    # Customize layout
    fig.update_layout(
        title = "COVID-19 Cases Trend",
        xaxis_title = "Date",
        yaxis_title = "Number of Cases",
        hovermode = "x unified",
        template = "plotly_white"
    )
    
    # Return the finalized interactive figure
    return fig


# Call the function to generate an interactive time-series plot
fig = create_interactive_trends(daily_totals, ["US", "India", "France"])
fig.show()


# =================================================
# Task 5: Interactive Country Comparison (Plotly)
# =================================================

def create_country_comparison(data: pd.DataFrame, metric: str = "Cases") -> go.Figure:
    """
    Create interactive country comparison

    Args:
        data (pd.DataFrame): Processed COVID-19 dataset
        metric (str): Comparison metric

    Returns:
        go.Figure: Plotly figure
    """
    # Raise an error if the metric is not found, providing a list of available columns for reference
    if metric not in data.columns:
        raise ValueError(f"Error: The metric '{metric}' does not exist in the dataset. Available columns: {list(data.columns)}")
    
    # Aggregate total cases per country
    total_cases = data.groupby("Country/Region", as_index = False)[metric].sum()
    
    # Raise an error if the metric column contains non-numeric values
    if not np.issubdtype(total_cases[metric].dtype, np.number):
        raise ValueError(f"Error: The column '{metric}' contains non-numeric values and cannot be used for comparison.")

    # Sort by highest case count
    total_cases = total_cases.sort_values(by = metric, ascending = False)
    
    # Get the top 10 countries
    top_10_countries = total_cases.nlargest(10, metric)["Country/Region"].tolist()
    
    # Get all countries sorted alphabetically
    all_countries = sorted(total_cases["Country/Region"].tolist())
    
    # Create an interactive bar chart
    fig = px.bar(
        total_cases,
        x = "Country/Region",
        y = metric,
        title = f"Total {metric} by Country",
        labels = {"Country/Region": "Country", metric: "Total Cases"},
        hover_name = "Country/Region",
        color = metric,
        color_discrete_sequence = px.colors.qualitative.Set1  # Use a categorical palette
    )
    
    # Create dropdown buttons
    dropdown_buttons = [
        {"label": "Show All", "method": "update", "args": 
            [{"x": [total_cases["Country/Region"]], "y": [total_cases[metric]]}]},
        {"label": "Top 10 Countries", "method": "update", "args": 
            [{"x": [top_10_countries], "y": [total_cases[total_cases["Country/Region"].isin(top_10_countries)][metric]]}]},
    ]
    
    # Add each country to the dropdown in alphabetical order
    for country in all_countries:
        dropdown_buttons.append({
            "label": country,
            "method": "update",
            "args": [{"x": [[country]], "y": [[total_cases[total_cases["Country/Region"] == country][metric].values[0]]]}]
        })
    
    # Apply dropdown menu to figure with improved positioning
    fig.update_layout(
        xaxis_title = "Country",
        yaxis_title = f"Total {metric}",
        xaxis_tickangle = -45,
        updatemenus = [
            {
                "buttons": dropdown_buttons,
                "direction": "down",
                "showactive": True,
                "x": -0.4,  # Move left to avoid overlapping with the title
                "y": 1.05,  # Lower to avoid overlapping with the title
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        margin=dict(l=180)  # Increase left margin further to separate menu from the chart
    )

    # Return the finalized interactive figure
    return fig


# Call the function to generate an interactive country comparison bar chart
fig = create_country_comparison(daily_totals)
fig.show()
