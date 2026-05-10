import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

def convert_fulldate_columns(csv_path, out_csv_path=None, date_col='FullDate', date_formats=None):
    """
    Parse date_col into Day_of_the_week, Date, Month and label-encode weekdays with Monday=0.
    Returns: (DataFrame, day_mapping) where day_mapping maps weekday name -> label (Mon=0 ... Sun=6).
    """
    df = pd.read_csv(csv_path, dtype=str)
    if date_col not in df.columns:
        raise KeyError(f"{date_col} not found in CSV")

    if date_formats is None:
        date_formats = [
            "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y", "%m/%d/%Y", "%d-%b-%Y", "%Y/%m/%d"
        ]

    def try_parse(s):
        if pd.isna(s):
            return pd.NaT
        s = str(s).strip()
        if not s or all(ch == '#' for ch in s):
            return pd.NaT
        ts = pd.to_datetime(s, errors='coerce', utc=False)
        if not pd.isna(ts):
            return ts
        for fmt in date_formats:
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
        return pd.NaT

    parsed = pd.to_datetime(df[date_col].map(try_parse), errors='coerce')

    df['Day_of_the_week'] = parsed.dt.day_name()
    df['Date'] = parsed.dt.day
    df['Month'] = parsed.dt.month

    # Create weekday number with Monday=0 .. Sunday=6 (NaN preserved)
    df['Weekday_Num'] = parsed.dt.weekday.astype('Int64')  # pandas weekday: Monday=0

    # If you still want a label column (categorical integers 0-6 for present days),
    # map day names to fixed labels Mon=0...Sun=6 so encoding is deterministic.
    weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    day_mapping = {day: i for i, day in enumerate(weekday_order)}
    df['Day_of_the_week_Label'] = df['Day_of_the_week'].map(day_mapping).astype('Int64')
    df['Price_Class'] = df.pop('Price_Class')  # Ensure Price_Class is last column
    df.drop(columns=['Day_of_the_week'], inplace=True)

    if out_csv_path:
        df.to_csv(out_csv_path, index=False)
    return df, day_mapping


df = convert_fulldate_columns(r"Datasets\NT\Final_Fuel_Dataset_NT_V2.csv", out_csv_path=r"Datasets\NT\Final_Fuel_Dataset_NT_V3.csv")

