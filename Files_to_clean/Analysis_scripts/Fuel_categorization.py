#%%
import pandas as pd
from typing import Iterable, Tuple, Dict

#%%
def filter_regions(df: pd.DataFrame, keep=('Darwin', 'Palmerston')) -> pd.DataFrame:
    """
    Return a DataFrame keeping only rows where 'Region Name' is one of `keep`.
    """
    return df[df['Region Name'].isin(keep)].copy()

# df = pd.read_csv(r"Files_to_clean/filtered_fuel_price.csv")
# filtered = filter_regions(df)
# print(filtered.head())
# filtered.to_csv(r"Files_to_clean\filtered_fuel_price.csv", index=False)

# %%

df = pd.read_csv(r"C:\Personal\Masters\Masters_work\Study\Y1_S2\PRT564\Assignments\Assignment_2\Repo\DATA-VISUALIZATION-GROUP-11\Files_to_clean\filtered_fuel_price_DP.csv")

def encode_columns_categories(
    df: pd.DataFrame,
    columns: Iterable[str]
) -> Tuple[pd.DataFrame, Dict[str, Dict[object, int]]]:
    """
    For each column in `columns`, encode distinct categories to integers.
    Returns (encoded_df, mappings) where `mappings[col]` maps category -> integer.
    Also prints a readable mapping string for each column.
    """
    df_enc = df.copy()
    mappings: Dict[str, Dict[object, int]] = {}

    for col in columns:
        if col not in df_enc.columns:
            raise KeyError(f"Column not found: {col}")
        # get unique values in stable order
        uniques = pd.Series(df_enc[col].dropna().unique())
        # create mapping (NaNs left as NaN in encoded column)
        mapping = {val: i for i, val in enumerate(uniques)}
        mappings[col] = mapping
        # map values, preserve NaN
        df_enc[col + "_encoded"] = df_enc[col].map(mapping)

        # print human-readable mapping
        mapping_str = ", ".join(f"{repr(k)} -> {v}" for k, v in mapping.items())
        print(f"Column '{col}': {mapping_str}")

    return df_enc, mappings

encoded_df, maps = encode_columns_categories(df, ['Region Name','Suburb','Brand Name'])
print(encoded_df)
print(maps)
#%%
encoded_df.to_csv(r"Encoded_fuel_price.csv", index=False)
