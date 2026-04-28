import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

def plot_histogram(df: pd.DataFrame, columm: str, bins: int = 30):
    plt.hist(df[columm], bins=bins, edgecolor='black')
    plt.title(f'Histogram of {columm}')
    plt.xlabel(columm)
    plt.ylabel('Frequency')
    plt.show()


def plot_correlation(df, x_col, y_col):
    plt.scatter(df[x_col], df[y_col], alpha=0.5)
    plt.title(f'Scatter Plot of {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)

    # Calculate and plot the line of best fit
    slope, intercept, r_value, p_value, std_err = linregress(df[x_col], df[y_col])
    line = slope * df[x_col] + intercept
    plt.plot(df[x_col], line, color='red', label=f'Best Fit Line (R²={r_value**2:.2f})')
    plt.legend()
    plt.show()


def print_nan_summary(df: pd.DataFrame, name: str):
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)

    if nan_cols.empty:
        print(f"\n{name}: No NaNs")
        return

    print(f"\n{name}: Columns with NaNs")
    for col, count in nan_cols.items():
        print(f"{col}: {count} ({count / len(df):.2%})")