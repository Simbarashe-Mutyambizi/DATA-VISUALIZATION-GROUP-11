import pandas as pd

def df_clean_and_summary(df, date_col=None):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas.DataFrame")
    # default to first column if not specified
    if date_col is None:
        date_col = df.columns[0]
    rows_before, cols_before = df.shape

    # convert date column to datetime (accepts 'YYYY-MM-DD') and format as 'YYYY/MM/DD'
    df = df.copy()
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.strftime('%Y/%m/%d')
    except Exception:
        # if conversion fails, leave column as-is
        pass

    # fill missing/empty with 0
    df_filled = df.fillna(0).replace("", 0)
    df_filled = df.fillna(0).replace(" ", 0)

    # drop rows that are all zeros
    df_cleaned = df_filled[(df_filled != 0).any(axis=1)]

    # drop columns that are all zeros
    df_cleaned = df_cleaned.loc[:, (df_cleaned != 0).any(axis=0)]

    rows_after, cols_after = df_cleaned.shape
    print(f"Before: rows={rows_before}, cols={cols_before}")
    print(f"After:  rows={rows_after}, cols={cols_after}")
    return df_cleaned

# Example
if __name__ == "__main__":
    df = pd.read_csv("Files_to_clean\combined_bike.csv")
    cleaned = df_clean_and_summary(df)
    print(cleaned)
    cleaned.to_csv( "Files_to_clean\cleaned_combined_bike.csv", index=False)
    # print(cleaned.iat[120, 24])
