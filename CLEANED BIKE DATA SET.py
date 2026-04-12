import pandas as pd

# loading the combined bike data set

bike = pd.read_csv('combined_bike.csv')

print("DATASET LOADED")
print(f"Shape: {bike.shape}")
print(f"Years present: {sorted(bike['year'].unique())}")
print()

# selecting only the columns we need for our analysis
#   Count_Date  - the exact date of each bike count, needed to calculate
#                 the 14-day fuel price lookback window
#   year        - needed for filtering to analysis years and merging
#   electorate  - identifies Darwin vs Palmerston, needed for grouping
#                 in visualisations and subgroup analysis
#   site_id     - uniquely identifies each counting location, needed to
#                 correctly average sites counted multiple times in one year
#   Trips/hour  - our regression Y variable (dependent variable), measures
#                 cycling activity normalised by time

cols_needed = ['Count_Date', 'year', 'electorate', 'site_id', 'Trips/hour']
bike = bike[cols_needed]
 
print("COLUMNS SELECTED")
print(f"Columns kept: {list(bike.columns)}")
print(f"Shape now: {bike.shape}")
print()

# removing artefact rows to improve data quality
# Identify them by their combination of null site_id AND zero Trips/hour.
# Using both conditions avoids accidentally removing genuine zero counts
# that might have a valid site_id.

rows_before = len(bike)
bike = bike[~((bike['site_id'].isnull()) & (bike['Trips/hour'] == 0.0))]
rows_after = len(bike)
 
print("ARTEFACT ROWS REMOVED")
print(f"Rows before: {rows_before} | Rows after: {rows_after}")
print(f"Removed: {rows_before - rows_after} placeholder rows")
print()

# removing 2025 rows (no matching fuel data)

rows_before = len(bike)
bike = bike[bike['year'].isin([2019, 2021, 2023])]
rows_after = len(bike)
 
print("2025 ROWS REMOVED (OUT OF SCOPE FOR ANALYSIS)")
print(f"Rows before: {rows_before} | Rows after: {rows_after}")
print(f"Removed: {rows_before - rows_after} out-of-scope rows")
print(f"Years remaining: {sorted(bike['year'].unique())}")
print()

# converting Count_Date to proper date time  format

bike['Count_Date'] = pd.to_datetime(bike['Count_Date'], format='%Y-%m-%d')
 
print("DATES CONVERTED")
print(f"Count_Date dtype: {bike['Count_Date'].dtype}")
print(f"Date range: {bike['Count_Date'].min().date()} to {bike['Count_Date'].max().date()}")
print()

# handle duplicate sites
# Some sites were counted on multiple Tuesdays in the same year (especially
# 2023 which had counts across July and August)will cause psuedo-replication
# DECISION: For each site_id within each year, we:
#   - Average the Trips/hour values (mean of all count days)
#   - Take the FIRST Count_Date (earliest Tuesday that year) because it accounts for behavioural lag
#   - Take the FIRST electorate value (same site = same electorate always)

bike_averaged = bike.groupby(['site_id', 'year']).agg(
    Count_Date  = ('Count_Date',  'first'),   
    electorate  = ('electorate',  'first'),   
    Trips_hour  = ('Trips/hour',  'mean')     
).reset_index()

# renaming Trips_hour back to Trips/hour for consistency with original data

bike_averaged = bike_averaged.rename(columns={'Trips_hour': 'Trips/hour'})
 
print("DUPLICATE SITES AVERAGED")
print(f"Rows before averaging: {len(bike)}")
print(f"Rows after averaging:  {len(bike_averaged)}")
print()
print("Row counts per year after averaging:")
print(bike_averaged['year'].value_counts().sort_index())
print()

# validating cleaned data set
#   1. No nulls in any column
#   2. Trips/hour range is sensible (no negatives, no extreme outliers)
#   3. site_id is unique within each year (no remaining duplicates)

print("VALIDATION")
 
print("Null counts:")
print(bike_averaged.isnull().sum().to_string())
print()
 
print("Trips/hour statistics:")
print(bike_averaged['Trips/hour'].describe().round(2).to_string())
print()

dupe_check = bike_averaged.groupby(['site_id', 'year']).size()
duplicates = dupe_check[dupe_check > 1]
print(f"Duplicate site-year combinations: {len(duplicates)} (should be 0)")
print()
 
print("Electorate breakdown:")
print(bike_averaged['electorate'].value_counts())
print()

# Saving cleaned bike data set to a new CSV file

bike_averaged.to_csv('bike_cleaned_final.csv', index=False)
 
print("SAVED")
print("Output file: bike_cleaned_final.csv")
print(f"Final shape: {bike_averaged.shape[0]} rows x {bike_averaged.shape[1]} columns")
print()
print("Bike dataset cleaning complete. Ready for merge with fuel data.")
