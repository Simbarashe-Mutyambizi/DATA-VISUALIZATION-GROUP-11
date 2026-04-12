import pandas as pd

# loading the previous combined fuel data set

existing = pd.read_csv('combined_fuel_price.csv')
existing = existing.drop(columns=['Unnamed: 0'])
print ("Existing data set loaded.")
print (f"Rows: {len(existing)}")

# Loading the 4 missing Data sets

july = pd.read_excel('treasury-report-1-july-2023-to-31-july-2023.xlsx')
august = pd.read_excel('treasury-report-1-august-2023-to-31-august-2023.xlsx')
september = pd.read_excel('treasury-report-31-august-to-1-october-2023.xlsx')
october = pd.read_excel('treasury-report-1-october-2023-to-31-october-2023.xlsx')

print("RAW ROW COUNTS OF THE MISSING DATA SETS:")
print(f"July: {len(july)} rows")
print(f"August: {len(august)}")
print(f"September: {len(september)} rows (NOTE : FILENAME SAYS Aug 31 to Oct 1)")
print(f"October: {len(october)} rows")

# standardising date formats across all files

existing['FullDate'] = pd.to_datetime(existing['FullDate'], format='%Y-%m-%d')
existing['FullDate'] = existing['FullDate'].dt.normalize()
 
for df in [july, august, september, october]:
    df['FullDate'] = pd.to_datetime(df['FullDate'], format='mixed', dayfirst=True)
    df['FullDate'] = df['FullDate'].dt.normalize()

print("DATE FORMATS STANDARDISED ACROSS ALL FILES.")
print(f"July:      {july['FullDate'].min().date()} to {july['FullDate'].max().date()}")
print(f"August:    {august['FullDate'].min().date()} to {august['FullDate'].max().date()}")
print(f"September: {september['FullDate'].min().date()} to {september['FullDate'].max().date()}")
print(f"October:   {october['FullDate'].min().date()} to {october['FullDate'].max().date()}")
print() 


# removing duplicate boundary rows from september data set (which contains both Aug 31 and Oct 1)   

rows_before = len(september)
september = september[(september['FullDate'].dt.month == 9) & (september['FullDate'].dt.year == 2023)]
rows_after = len(september)
 
print("=== STEP 4: BOUNDARY DUPLICATES REMOVED ===")
print(f"September: {rows_before} rows -> {rows_after} rows")
print(f"Removed {rows_before - rows_after} duplicate boundary rows")
print()


# Verifying column alignment across all data sets

existing_cols = list(existing.columns)
new_cols      = list(july.columns)
 
print("COLUMN ALIGNMENT CHECK")
missing = set(existing_cols) - set(new_cols)
extra   = set(new_cols) - set(existing_cols)
print(f"Missing from new files: {missing if missing else 'None - all good'}")
print(f"Extra in new files:     {extra if extra else 'None - all good'}")
print()

# Combing all data sets together

combined = pd.concat(
    [existing, july, august, september, october],
    ignore_index=True
)
 
print("COMBINED DATA SET CREATED")
print(f"Total rows: {len(combined)}")
print(f"Date range: {combined['FullDate'].min().date()} to {combined['FullDate'].max().date()}")
print()

# Final check for duplicates in the combined data set

rows_before = len(combined)
combined = combined.drop_duplicates()
rows_after = len(combined)
 
print("DEDUPLICATION CHECK")
print(f"Before: {rows_before} | After: {rows_after} | Removed: {rows_before - rows_after}")
print()

# creating new columns for month and year

combined['year']  = combined['FullDate'].dt.year
combined['month'] = combined['FullDate'].dt.month
months_2023 = sorted(combined[combined['year'] == 2023]['month'].unique())
 
print("=== STEP 8: 2023 MONTH COVERAGE ===")
print(f"Months present: {[int(m) for m in months_2023]}")
print(f"Expected:       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]")
if len(months_2023) == 12:
    print("PASS: All 12 months present")
else:
    print("WARNING: Some months still missing!")
print()
 
print("Row counts per year:")
print(combined['year'].value_counts().sort_index().to_string())
print()

# Saving the final combined data set to a new CSV file

combined.to_csv('combined_fuel_CORRECTED.csv', index=False)
 
print("SAVED final combined data set to 'combined_fuel_CORRECTED.csv'")
print("Output: combined_fuel_CORRECTED.csv")
print(f"Final shape: {combined.shape[0]} rows x {combined.shape[1]} columns")
print("Script complete. Ready for analysis.")