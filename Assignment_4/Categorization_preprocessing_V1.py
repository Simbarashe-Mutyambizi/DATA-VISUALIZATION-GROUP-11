import pandas as pd
from typing import Sequence

def drop_columns_by_index_and_move(input_path: str, output_path: str, idxs_to_drop: Sequence[int], col_to_move: str = "Unleaded 91") -> None:
    """
    Read CSV from input_path, drop columns at the specified zero-based indices (if present),
    move column named col_to_move to the end (if present), and write result to output_path.

    idxs_to_drop: sequence of column indices (zero-based). Out-of-range indices are ignored.
    col_to_move: column name to relocate to the end.
    """
    df = pd.read_csv(input_path)

    # Drop by index (normalize valid integer indices)
    num_cols = len(df.columns)
    valid_idxs = sorted({i for i in idxs_to_drop if isinstance(i, int) and 0 <= i < num_cols}, reverse=True)
    if valid_idxs:
        cols_to_drop = [df.columns[i] for i in valid_idxs]
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped columns (by index): {valid_idxs} -> names: {cols_to_drop}")
    else:
        print("No valid column indices to drop.")

    # Move specified column to the end if present
    if col_to_move in df.columns:
        cols = [c for c in df.columns if c != col_to_move] + [col_to_move]
        df = df[cols]
        print(f"Moved column to end: {col_to_move}")
    else:
        print(f"Column to move not found: {col_to_move}")   

    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Example: drop first and third columns, then move "Unleaded 91" to the end
    indices_to_remove = [0,2,3,4,5,8,9,10,12,13,14,15,16]  # drops first and third columns if they exist
    input_csv = "Datasets\\NT\\Final_Fuel_Dataset_NT.csv"
    output_csv = "Datasets\\NT\\Final_Fuel_Dataset_NT_V1.csv"

    drop_columns_by_index_and_move(input_csv, output_csv, indices_to_remove, col_to_move="Unleaded 91")


