import pandas as pd
from typing import Sequence

def add_price_class_column(input_path: str, output_path: str, price_col_name: str = "Unleaded 91", new_col_name: str = "Price_Class") -> None:
    """
    Read CSV from input_path, compute a new column new_col_name based on comparisons of
    the price_col_name column to its previous row, append it as the last column, and write to output_path.

    Rules per row i (compare value_i to value_{i-1}):
      - if previous value == 0 -> 'ZERO'
      - elif value_i < previous -> 'DECREASE'
      - elif value_i > previous -> 'HIKE'
      - else -> 'STABLE'

    Notes:
      - Assumes price_col_name is the last column in the input DataFrame (but will locate it by name).
      - Non-numeric or missing values are treated as NaN; comparisons with NaN yield 'STABLE' except when previous==0.
    """
    df = pd.read_csv(input_path)

    if price_col_name not in df.columns:
        raise ValueError(f"Price column not found: {price_col_name}")

    # Ensure price column is numeric (coerce errors to NaN)
    prices = pd.to_numeric(df[price_col_name], errors="coerce")

    # Shifted previous values
    prev = prices.shift(1)

    def classify(curr, prev_val):
        # If previous is exactly 0 (including numeric 0.0)
        if pd.notna(prev_val) and prev_val == 0:
            return "ZERO"
        # If either is NaN (except previous==0 handled above), treat as STABLE
        if pd.isna(curr) or pd.isna(prev_val):
            return "STABLE"
        if curr < prev_val:
            return "DECREASE"
        if curr > prev_val:
            return "HIKE"
        return "STABLE"

    df[new_col_name] = [classify(c, p) for c, p in zip(prices, prev)]

    # Ensure the new column is the last column
    cols = [c for c in df.columns if c != new_col_name] + [new_col_name]
    df = df[cols]

    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_csv = "Datasets\\NT\\Final_Fuel_Dataset_NT_V1.csv"
    output_csv = "Datasets\\NT\\Final_Fuel_Dataset_NT_V2.csv"
    add_price_class_column(input_csv, output_csv)
