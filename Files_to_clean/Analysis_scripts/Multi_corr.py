import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def multicollinearity_matrix(df: pd.DataFrame,
                             include_non_numeric: bool = False,
                             method: str = "pearson",
                             figsize=(10, 8),
                             cmap="vlag",
                             annot=True,
                             fmt=".2f",
                             vmin=-1,
                             vmax=1,
                             title="Correlation (Multicollinearity) Matrix"):
    """
    Compute and visualize a correlation (multicollinearity) matrix.

    Parameters:
    - df: input DataFrame
    - include_non_numeric: if True, coerce non-numeric to numeric (errors -> NaN)
    - method: correlation method ('pearson', 'spearman', 'kendall', etc.)
    - figsize, cmap, annot, fmt: heatmap display options
    - vmin, vmax: color scale limits
    - title: plot title

    Returns:
    - corr: pandas.DataFrame correlation matrix
    - fig, ax: matplotlib objects for the heatmap
    """
    # select / coerce numeric data
    if include_non_numeric:
        coerced = df.apply(pd.to_numeric, errors="coerce")
    else:
        coerced = df.select_dtypes(include=[np.number]).copy()

    if coerced.shape[1] == 0:
        raise ValueError("No numeric columns available for correlation matrix.")

    # compute correlation matrix
    corr = coerced.corr(method=method)

    # plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, ax=ax, cmap=cmap, annot=annot, fmt=fmt, vmin=vmin, vmax=vmax,
                square=True, cbar_kws={"shrink": .8}, linewidths=0.5, linecolor="white")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    fig.savefig('Multi_corr_Encoded_bike.png')
    return corr, fig, ax


df = pd.read_csv(r"Files_to_clean\cleaned_combined_bike.csv")
cols_to_drop = list(df.columns[1:21]) + list(df.columns[28:44])
df.drop(columns=cols_to_drop, inplace=True)  
corr, fig, ax = multicollinearity_matrix(df, include_non_numeric=False, method="pearson")
print(corr) 




