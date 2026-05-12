# import pandas as pd

# # df = pd.read_csv("Queensland_cleaned_fuel.csv")
# # print(df['Fuel_Type'].unique())



# def filter_fuel_types(df, col="Fuel_Type"):
#     if not isinstance(df, pd.DataFrame):
#         raise TypeError("Expected a pandas.DataFrame")
#     if col not in df.columns:
#         raise KeyError(f"Column not found: {col}")
#     allowed = {"PULP 95/96 RON", "PULP 98 RON", "Diesel", "Unleaded"}
#     # normalize values (strip whitespace) and filter
#     mask = df[col].astype(str).str.strip().isin(allowed)
#     return df.loc[mask].reset_index(drop=True)

# df = pd.read_csv("Queensland_cleaned_fuel.csv")
# filtered = filter_fuel_types(df)            # uses default "Fuel_type"
# filtered.to_csv("Queensland_filtered_V1.csv", index=False)


# #%%

import pandas as pd
from pathlib import Path

import pandas as pd
from pathlib import Path

def split_by_fuel_type_to_csv(csv_path, fuel_col_any="Fuel_Type", out_dir="fuel_splits", read_kwargs=None, write_kwargs=None):
    read_kwargs = read_kwargs or {}
    write_kwargs = write_kwargs or {}
    df = pd.read_csv(csv_path, low_memory=False, **read_kwargs)
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

    cols_lower = {c.lower(): c for c in df.columns}
    if fuel_col_any in df.columns:
        key = fuel_col_any
    elif fuel_col_any.lower() in cols_lower:
        key = cols_lower[fuel_col_any.lower()]
    else:
        candidates = ["Fuel_Type", "FuelType", "Fuel Type", "fuel_type", "fuel type"]
        key = None
        for cand in candidates:
            if cand in df.columns:
                key = cand; break
            if cand.lower() in cols_lower:
                key = cols_lower[cand.lower()]; break
        if key is None:
            raise KeyError(f"Fuel type column not found. Available: {list(df.columns)}")

    df[key] = df[key].astype(str).str.strip()
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    for val, group in df.groupby(df[key]):
        g = group.reset_index(drop=True)
        # move Price to the end if it exists (case-sensitive match and common variants)
        price_candidates = ["Price", "price", "PRICE"]
        price_col = next((c for c in g.columns if c in price_candidates), None)
        if price_col:
            cols = [c for c in g.columns if c != price_col] + [price_col]
            g = g[cols]
        safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in (val or "empty"))
        filepath = out_p / f"{safe}.csv"
        g.to_csv(filepath, index=False, **write_kwargs)


split_by_fuel_type_to_csv("Queensland_filtered_V1.csv")
