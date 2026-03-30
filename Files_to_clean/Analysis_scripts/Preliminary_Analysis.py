import pandas as pd

def describe_dataframe(path, **read_kwargs):
    """
    Read a DataFrame from `path` (CSV by default) and print:
      - initial dimen   ons (rows, columns)
      - basic numeric statistics (count, mean, std, min, 25%, 50%, 75%, max)
      - basic non-numeric stats (count, unique, top, freq)
      - memory usage
      - number of missing values per column
    read_kwargs are passed to pd.read_csv (use engine/sep/dtype if needed).
    """
    # load (adjust to pd.read_excel, pd.read_parquet, etc. as needed)
    df = pd.read_csv(path, **read_kwargs)

    # initial dimensions
    rows, cols = df.shape
    print(f"Initial dimensions: rows={rows}, columns={cols}\n")

    # head
    print("First 5 rows:")
    print(df.head(), "\n")

    # overall info summary
    print("Info:")
    df.info(memory_usage="deep")
    print()

    # missing values
    missing = df.isna().sum().sort_values(ascending=False)
    print("Missing values per column (descending):")
    print(missing[missing > 0] if missing.any() else "No missing values")
    print()

    # memory usage
    mem = df.memory_usage(deep=True).sum()
    print(f"Total memory usage: {mem:,} bytes\n")

    # numeric description
    print("Numeric columns summary:")
    print(df.select_dtypes(include="number").describe().T)
    print()

    # non-numeric description
    nonnum = df.select_dtypes(exclude="number")
    if not nonnum.empty:
        print("Non-numeric columns summary:")
        print(nonnum.describe(include="all").T)
    else:
        print("No non-numeric columns.")
    print()

    # optionally show value counts for small-cardinality columns
    small_card = [c for c in df.columns if df[c].nunique(dropna=False) <= 10]
    if small_card:
        print("Value counts for columns with <=10 unique values:")
        for c in small_card:
            print(f"\nColumn: {c}")
            print(df[c].value_counts(dropna=False))
    else:
        print("No columns with <=10 unique values to show value counts.")

    return df

if __name__ == "__main__":
    # example usage: adjust path to your file

    df = describe_dataframe("combined_bike.csv")
