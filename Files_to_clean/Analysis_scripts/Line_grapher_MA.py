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
def plot_col_by_hue(df, y_col, hue_col, cmap='viridis'):
    """
    Plot y_col vs row index as a line; line color follows hue_col values.
    df: pandas.DataFrame
    y_col, hue_col: column names (strings)
    cmap: matplotlib colormap name
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")
    df = df.reset_index(drop=True)
    x = df.index.to_numpy()
    y = df[y_col].to_numpy()
    h = pd.to_numeric(df[hue_col], errors='coerce').to_numpy()

    valid = ~pd.isna(y)
    x, y, h = x[valid], y[valid], h[valid]

    norm = plt.Normalize(vmin=h.min(), vmax=h.max())
    cmap = plt.get_cmap(cmap)

    fig, ax = plt.subplots()
    for i in range(len(x)-1):
        color = cmap(norm((h[i] + h[i+1]) / 2 if not pd.isna(h[i]) and not pd.isna(h[i+1]) else h[i]))
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]], color=color, linewidth=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=hue_col)
    ax.set_xlabel("Row index")
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} (colored by {hue_col})")
    plt.tight_layout()
    fig.savefig
    return fig, ax

df = pd.read_csv('Files_to_clean\Enc_fuel_price.csv')
# df = df.sort_values(by='year', ascending=True)
# df.to_csv('Files_to_clean\cleaned_combined_bike.csv', index=False)
# plot_col_by_hue(df, 'Trips/hour', 'year')
# plt.show()
plot_columns_moving_average_subplots(df, ['Unleaded 91', 'Diesel', 'Premium 98','Premium 95'], window=1000)

