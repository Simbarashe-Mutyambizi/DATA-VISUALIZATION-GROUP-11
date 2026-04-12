import pandas as pd
import numpy as np

# loading the combined fuel_CORRECTED data set

fuel = pd.read_csv('combined_fuel_CORRECTED.csv', parse_dates=['FullDate'])
 
print("DATASET LOADED")
print(f"Rows: {len(fuel)} | Columns: {len(fuel.columns)}")
print(f"Date range: {fuel['FullDate'].min().date()} to {fuel['FullDate'].max().date()}")
print()

# removing ghost rows

rows_before = len(fuel)
fuel = fuel.dropna(subset=['FullDate'])
rows_after = len(fuel)
 
print("GHOST ROW REMOVED")
print(f"Rows before: {rows_before} | Rows after: {rows_after}")
print(f"Removed: {rows_before - rows_after} empty rows")
print()

# Determining percentage of misssing data for each fuel type.
# We calculate the percentage of valid (non-zero, non-null) records for each fuel type.

fuel_cols = [
    'Diesel', 'Premium 98', 'Premium 95', 'Unleaded 91',
    'Premium Diesel', 'LPG', 'Low Aromatic Fuel',
    'Ethanol 105 (E85)', 'Ethanol 94 (E10)', 'Bio Diesel 20'
]
 
print("=== STEP 2B: FUEL TYPE COMPLETENESS ANALYSIS ===")
print(f"Total rows in dataset: {len(fuel)}")
print()
 
results = []
 
for col in fuel_cols:
    valid = fuel[(fuel[col].notna()) & (fuel[col] > 0)]
    valid_count = len(valid)
    total_count = len(fuel)
    valid_pct = round((valid_count / total_count) * 100, 1)
    results.append({
        'Fuel Type':      col,
        'Valid Records':  valid_count,
        'Valid %':        valid_pct,
        'Missing/Zero %': round(100 - valid_pct, 1)
    })
 
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Valid %', ascending=False)
 
print(results_df.to_string(index=False))
print()
 
commuter_fuels = ['Diesel', 'Unleaded 91', 'Premium 95', 'Premium 98']
commuter_df = results_df[results_df['Fuel Type'].isin(commuter_fuels)]
best_fuel = commuter_df.iloc[0]['Fuel Type']
best_pct  = commuter_df.iloc[0]['Valid %']
 
print(f"Most complete commuter fuel type: {best_fuel} ({best_pct}% valid records)")
print()
 
 # removing outlier fuel prices to all fuel types.

print("IQR CLEANING APPLIED TO ALL FUEL TYPES")
print(f"{'Fuel Type':<22} {'Lower':>8} {'Upper':>8} {'Valid Before':>14} {'Valid After':>12} {'Removed':>9}")
print("-" * 80)
 
for col in fuel_cols:
    valid_prices = fuel[fuel[col] > 0][col]
 
    # Skip IQR filtering if fewer than 100 valid records exist
    # Cannot derive reliable bounds from very sparse data
    if len(valid_prices) < 100:
        valid_before = (fuel[col] > 0).sum()
        fuel[col] = fuel[col].replace(0, np.nan)
        valid_after = fuel[col].notna().sum()
        print(f"{col:<22} {'N/A':>8} {'N/A':>8} {valid_before:>14} {valid_after:>12} {valid_before - valid_after:>9}")
        continue
 
    Q1  = valid_prices.quantile(0.25)
    Q3  = valid_prices.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
 
    valid_before = (fuel[col] > 0).sum()
 
    # np.where(condition, value_if_true, value_if_false)
    # Keep original value if within bounds, replace with NaN if outside
    fuel[col] = np.where(
        (fuel[col] >= lower_bound) & (fuel[col] <= upper_bound),
        fuel[col],
        np.nan
    )
 
    valid_after = fuel[col].notna().sum()
    print(f"{col:<22} {lower_bound:>8.1f} {upper_bound:>8.1f} {valid_before:>14} {valid_after:>12} {valid_before - valid_after:>9}")
 
print()
print("All fuel type columns cleaned using fuel-type-specific IQR bounds.")
print()

 # verifying that fuel prices are now within reasonable ranges / real world ranges after cleaning.
# cross checking with historical fuel price data for Australia to ensure that the cleaned dataset contains realistic values.

# Source 1: Australian Institute of Petroleum - NT State Average  https://aip.com.au/pricing
# Source 2: NT Economy Retail Fuel Market https://nteconomy.nt.gov.au/prices-and-wages/retail-fuel-market

fuel['year'] = fuel['FullDate'].dt.year

print("YEARLY AVERAGE VERIFICATION")
print("Average prices per year for main commuter fuel types:")
print(fuel.groupby('year')[['Diesel', 'Unleaded 91', 'Premium 95', 'Premium 98']].mean().round(1).to_string())
print()

# National pattern confirmed: 2020 lowest (COVID), 2022-2023 highest (post-COVID recovery)
# Programmatic check - verify 2020 is lower than 2022

avg_by_year = fuel.groupby('year')['Diesel'].mean()

if avg_by_year[2020] < avg_by_year[2022]:
    print("PASS: 2020 average diesel lower than 2022 - consistent with known economics")
else:
    print("WARNING: Unexpected price pattern detected - review cleaning steps")
print()
print("Expected pattern: 2020 lowest, 2022-2023 highest")
print()

# checking for duplicates in the cleaned fuel data set

rows_before = len(fuel)
fuel = fuel.drop_duplicates()
rows_after = len(fuel)
 
print("=== STEP 6: DUPLICATE CHECK ===")
print(f"Rows before: {rows_before} | Rows after: {rows_after}")
print(f"Duplicates removed: {rows_before - rows_after}")
print()

# saving the cleaned fuel data set to a new CSV file

fuel.to_csv('combined_fuel_CLEANED.csv', index=False)
 
print("=== STEP 7: SAVED ===")
print("Output file: combined_fuel_CLEANED.csv")
print(f"Final shape: {fuel.shape[0]} rows x {fuel.shape[1]} columns")
print()
print("Fuel dataset cleaning complete.")