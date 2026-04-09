import pandas as pd
from typing import Iterable, Optional, List
import matplotlib.pyplot as plt
import ast

def describe_by_category(
    df: pd.DataFrame,
    category_col: str,
    value_cols: Optional[Iterable[str]] = None,
    dropna_categories: bool = True
) -> pd.DataFrame:
    """
    Return and print descriptive stats (count, mean, std, min, 25%, 50%, 75%, max, range)
    for numeric value columns grouped by a categorical column.
    """
    if category_col not in df.columns:
        raise KeyError(f"Category column not found: {category_col}")

    # determine numeric value columns if not provided
    if value_cols is None:
        value_cols = df.select_dtypes(include="number").columns.tolist()
    else:
        value_cols = list(value_cols)

    # validate value columns and keep only numeric ones
    missing = [c for c in value_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Value column(s) not found in DataFrame: {missing}")
    # filter to numeric columns only
    numeric_cols = [c for c in value_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric value columns to summarize after validation.")

    grp_df = df.dropna(subset=[category_col]) if dropna_categories else df.copy()

    # compute grouped aggregations via groupby + apply to avoid agg dict issues
    def stats(s: pd.Series) -> pd.Series:
        return pd.Series({
            "count": s.count(),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "25%": s.quantile(0.25),
            "50%": s.median(),
            "75%": s.quantile(0.75),
            "max": s.max(),
            "range": s.max() - s.min()
        })

    grouped = grp_df.groupby(category_col)[numeric_cols].apply(lambda g: g.apply(stats)).unstack(level=1)
    # reformat: produce a MultiIndex (category, value_col) with stats as columns
    # Current shape: index=category, columns=(value_col, stat). We'll stack value_col
    result = grouped.stack(level=0).swaplevel(0,1).sort_index(level=0)

    # print readable output
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    for category, df_cat in result.groupby(level=0):
        print(f"\nCategory: {category}")
        print(df_cat.droplevel(0))

    return result

def plot_fuel_stats_scatter(
    summary: pd.DataFrame,
    stat_names: List[str] = ["count", "mean", "std", "min", "max"],
    categories_order: Optional[List[str]] = None,
    figsize: tuple = (12, 6),
    marker: str = "o",
    fontsize: int = 9
):
    """
    summary: output of describe_by_category with MultiIndex (category, value_col)
             where first level = category (e.g., Region Name), second = value_col (fuel).
    stat_names: list of stats to plot (must be columns in summary)
    Produces one scatter plot per stat with x axis = fuel names and separate series/positions per category.
    """
    # pivot summary so we have index=(category, fuel) -> columns=stats
    # ensure index levels
    if not isinstance(summary.index, pd.MultiIndex) or summary.index.nlevels != 2:
        raise ValueError("summary must have MultiIndex (category, value_col).")

    categories = summary.index.get_level_values(0).unique().tolist()
    fuels = summary.index.get_level_values(1).unique().tolist()
    if categories_order:
        categories = [c for c in categories_order if c in categories] + [c for c in categories if c not in (categories_order or [])]

    x = range(len(fuels))
    x_labels = list(fuels)

    for stat in stat_names:
        if stat not in summary.columns:
            print(f"Skipping missing stat: {stat}")
            continue
        plt.figure(figsize=figsize)
        ax = plt.gca()
        width = 0.8 / max(1, len(categories))  # horizontal spacing for multiple categories
        for i, category in enumerate(categories):
            # get values for each fuel for this category (may be NaN if missing)
            vals = []
            for f in fuels:
                try:
                    vals.append(summary.loc[(category, f), stat])
                except KeyError:
                    vals.append(float("nan"))
            # shift x positions per category
            offsets = [xi + (i - (len(categories)-1)/2) * width for xi in x]
            ax.scatter(offsets, vals, label=str(category), marker=marker)
            # # annotate each point with fuel name above point
            # for xx, yy, fuel in zip(offsets, vals, x_labels):
            #     ax.text(xx, yy, fuel, fontsize=fontsize, rotation=90, va="bottom", ha="center")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylabel(stat)
        ax.set_title(f"{stat} per fuel (by category)")
        ax.legend(title="Category")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.show()
def cols_range_list_str(df, start_idx, end_idx):
    """
    Return a string representing a Python list of column names from start_idx to end_idx inclusive,
    using single quotes. Example: "['col1','col2','col3']".
    Uses 0-based indices.
    """
    cols = list(df.columns)
    n = len(cols)
    if not (0 <= start_idx < n) or not (0 <= end_idx < n):
        raise IndexError(f"indices out of range (0..{n-1})")
    if start_idx <= end_idx:
        selected = cols[start_idx:end_idx+1]
    else:
        selected = cols[end_idx:start_idx+1]
    items = ",".join(f"'{c}'" for c in selected)
    return f"[{items}]"
df = pd.read_csv("Encoded_fuel_price.csv")
s = cols_range_list_str(df, 8,17)
print(s)  # e.g. "Region Name, Diesel"
# summary for numeric columns automatically:
summary = describe_by_category(df, category_col="Region Name",value_cols=ast.literal_eval(s))
plot_fuel_stats_scatter(summary, stat_names=["mean"])
