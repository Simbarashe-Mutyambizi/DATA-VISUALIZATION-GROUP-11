import sys
import pandas as pd

def sort_csv_by_year(infile, outfile=None, year_col='Year'):
    df = pd.read_csv(infile)
    df.sort_values(year_col, inplace=True)
    df.to_csv(outfile or infile, index=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sort_by_year.py input.csv [output.csv]")
        raise SystemExit(1)
    sort_csv_by_year(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

sort_csv_by_year("Files_to_clean\cleaned_combined_bike.csv", "Files_to_clean\cleaned_combined_bike.csv") 