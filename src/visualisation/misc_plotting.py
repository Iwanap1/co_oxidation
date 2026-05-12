import matplotlib.pyplot as plt
import pymatviz as pmv
from typing import List
import pandas as pd
from scipy.stats import linregress
from src.data.element_attributes import METALS

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


def feature_presence_bar_chart(df: pd.DataFrame, features: List[str]):
    freq = {}
    for col in features:
        if col not in df.columns:
            continue

        s = df[col]

        if pd.api.types.is_numeric_dtype(s):
            count = ((s.notna()) & (s != 0)).sum()
        else:
            count = ((s.notna()) & (s.astype(str).str.strip() != "")).sum()

        freq[col] = count

    freq_df = (
        pd.DataFrame({
            "column": list(freq.keys()),
            "count": list(freq.values())
        })
        .sort_values("count", ascending=False)
    )

    # Plot
    plt.figure(figsize=(14, 8))

    plt.bar(freq_df["column"], freq_df["count"])

    plt.xticks(rotation=90)
    plt.ylabel("Frequency")
    plt.xlabel("Column")
    plt.title("Non-zero / non-null frequency of MATERIAL_COLS")

    plt.tight_layout()
    plt.show()


def plot_periodic_table_frequency(
    df: pd.DataFrame,
    elements=METALS,
    *,
    log: bool = False,
    title: str = "Element frequency in dataset",
    cbar_title: str = "No. materials",
    include_ce: bool = True,
):
    
    if not include_ce and "Ce" in elements:
        elements = [el for el in elements if el != "Ce"]

    elements = [el for el in elements if el in df.columns]
    freq = {}

    for el in elements:
        if el not in df.columns:
            continue

        s = pd.to_numeric(df[el], errors="coerce")
        freq[el] = int(((s.notna()) & (s != 0)).sum())

    fig = pmv.ptable_heatmap_plotly(
        freq,
        log=log,
        fmt=".0f",
        colorbar={"title": cbar_title},
    )

    fig.update_layout(title=title)
    fig.show()

    return freq