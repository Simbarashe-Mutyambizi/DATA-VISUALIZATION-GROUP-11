import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence, Tuple

def plot_columns_moving_average_subplots(
    df: pd.DataFrame,
    columns: Sequence[str],
    window: int = 100,
    figsize: Tuple[int,int] = (12, 8),
    sharex: bool = True,
    sharey: bool = False,
    style: str = '-',
    cmap: str = 'tab10',
    min_periods: int = 1
):
    """
    Plot the moving average (rolling mean) of given columns from `df` as line plots in subplots.
    - df: pandas DataFrame already loaded.
    - columns: list/tuple of column names to plot (must exist in df).
    - window: rolling window size (default 100).
    - min_periods: min periods for rolling (default 1). Use window to require full window.
    - figsize, sharex, sharey, style, cmap: matplotlib params.
    Returns (fig, axes).
    """
    cols = list(columns)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"These columns are not in DataFrame: {missing}")
    if len(cols) == 0:
        raise ValueError("No columns provided to plot.")
    if window < 1:
        raise ValueError("window must be >= 1")

    # compute rolling mean for selected columns
    rolling_df = df[cols].rolling(window=window, min_periods=min_periods).mean()

    n = len(cols)
    ncols = int(n**0.5) or 1
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex=sharex, sharey=sharey, squeeze=False)
    axes_flat = axes.flatten()

    cmap_colors = plt.get_cmap(cmap).colors
    for i, col in enumerate(cols):
        ax = axes_flat[i]
        series = rolling_df[col]
        ax.plot(series.index, series.values, style, label=f"{col} (MA{window})",
                color=cmap_colors[i % len(cmap_colors)])
        ax.set_title(f"{col} — {window}-period MA")
        ax.grid(True)
        if not sharex:
            ax.set_xlabel('Index')
        if not sharey:
            ax.set_ylabel('Value')
        ax.legend(loc='best', fontsize='small')

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plt.show()
    fig.savefig('Fuel_Price_Moving_Average.png')
    return fig, axes

df = pd.read_csv("combined_fuel_price.csv")
plot_columns_moving_average_subplots(df, ['Diesel','Premium 95','Unleaded 91','Premium 98'], window=1000)
