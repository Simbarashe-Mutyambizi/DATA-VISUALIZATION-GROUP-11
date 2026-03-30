import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence

def plot_columns_line_subplots(df: pd.DataFrame, columns: Sequence[str], figsize=(12, 8),
                               sharex=True, sharey=False, style='-o', cmap='tab10'):
    """
    Plot given columns from `df` as line plots in subplots.
    - df: pandas DataFrame already loaded.
    - columns: list/tuple of column names to plot (must exist in df).
    - figsize: overall figure size.
    - sharex/sharey: pass-through to plt.subplots.
    - style: matplotlib line style (e.g., '-','-o','--').
    - cmap: matplotlib colormap name for line colors.
    """
    cols = list(columns)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"These columns are not in DataFrame: {missing}")

    n = len(cols)
    if n == 0:
        raise ValueError("No columns provided to plot.")

    # grid layout: try near-square layout
    ncols = int(n**0.5)
    ncols = max(1, ncols)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex=sharex, sharey=sharey, squeeze=False)
    axes_flat = axes.flatten()

    colors = plt.get_cmap(cmap).colors
    for i, col in enumerate(cols):
        ax = axes_flat[i]
        series = df[col]
        ax.plot(series.index, series.values, style, label=col, color=colors[i % len(colors)])
        ax.set_title(col)
        ax.grid(True)
        if not sharex:
            ax.set_xlabel('Index')
        if not sharey:
            ax.set_ylabel('Value')
        ax.legend(loc='best', fontsize='small')

    # turn off any unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plt.show()
    return fig, axes

df = pd.read_csv("combined_fuel_price.csv")  
plot_columns_line_subplots(df, ['Diesel'])
# ['Diesel', 'Premium 98', 'Premium 95','Unleaded 91','Premium Diesel','LPG','Low Aromatic Fuel','Ethanol 105 (E85)','Ethanol 94 (E10)','Bio Diesel 20']