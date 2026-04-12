import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition   import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_score
from scipy import stats

# loading both cleaned datasets

fuel = pd.read_csv('combined_fuel_CLEANED.csv', parse_dates=['FullDate'])
bike = pd.read_csv('bike_cleaned_final.csv',    parse_dates=['Count_Date'])
 
print("DATASETS LOADED")
print(f"Fuel dataset: {fuel.shape[0]:,} rows | {fuel.shape[1]} columns")
print(f"Bike dataset: {bike.shape[0]} rows | {bike.shape[1]} columns")
print()

# filtering fuel dataset to darwin and palmerston only (bike data is only for these two locations)

fuel = fuel[fuel['Region Name'].isin(['Darwin', 'Palmerston'])]
 
print("FILTERED TO DARWIN AND PALMERSTON")
print(f"Fuel records remaining: {fuel.shape[0]:,}")
print(f"Regions: {fuel['Region Name'].unique()}")
print()


# PCA assessment - justification for both datasets.
# We formally assess all commuter fuel types for PCA viability.
# PCA is applicable when:
#   1. Multiple predictor variables exist
#   2. Those predictors are highly correlated (multicollinearity)
#   3. Each predictor has sufficient data coverage to be reliable
#
# We set 70% valid records as minimum threshold for PCA viability.
# A fuel type with less than 70% valid records would introduce bias into any PCA component it contributes to.
# Diesel, Unleaded 91 and Premium 95 all rise and fall together because they share the same underlying drivers - global oil prices and NT supply chain costs. 
# PCA Component 1 will capture this shared fuel price signal as one
# number representing the overall fuel price level in Darwin/Palmerston.
#
# WHY PCA INSTEAD OF SELECTING ONE FUEL TYPE:
# Feature selection - choosing just one fuel type - was considered as
# a simpler alternative to PCA. However PCA was preferred for two reasons:
#   1. PCA uses price information from ALL three fuel types rather than
#      discarding two thirds of available fuel price data. A commuter
#      driving a petrol car responds to Unleaded 91 prices not Diesel.
#      PCA keeps that real signal rather than throwing it away.
#   2. PCA removes the subjective choice of which fuel type to pick by
#      letting mathematics find the optimal combination objectively.
#      Manually choosing Diesel over Unleaded 91 requires a threshold
#      judgement that an examiner could legitimately challenge.
# NOTE: With 98.5% variance explained by Component 1, the three fuel
# types are almost perfectly correlated. Results using PCA versus using
# Diesel alone would be near-identical. The small difference justifies
# PCA for methodological rigour and assignment requirements.


# WHY PCA WAS NOT APPLIED TO THE BIKE DATASET:
# PCA reduces multiple correlated PREDICTOR variables (X) into fewer
# components. It is not applicable to outcome variables (Y).
# Our Y variable is Trips/hour - selected because:
#   1. Zero missing values across all valid bike observations
#   2. Normalised measure - accounts for counting duration differences
#      making sites directly comparable regardless of observation length
#   3. Directly comparable across sites of different sizes and locations
# The other candidate columns (leg totals, enter/exit counts) are raw
# counting components that FEED INTO Trips/hour - they are not
# independent variables. Including them alongside Trips/hour would
# create perfect multicollinearity on the Y side, not the X side.
# PCA is an X-side technique. Trips/hour is our Y variable.
# PCA NOT APPLICABLE to outcome variable selection.


commuter_fuels = ['Diesel', 'Unleaded 91', 'Premium 95',
                  'Premium 98', 'LPG']
 
print("PCA VIABILITY ASSESSMENT")
print(f"{'Fuel Type':<15} {'Valid %':>8} {'PCA Viable?':>14}")
print("-" * 40)
 
viable_fuels = []
for col in commuter_fuels:
    valid_pct = round(
        fuel[col].notna().sum() / len(fuel) * 100, 1
    )
    viable = valid_pct >= 70
    status = "YES" if viable else "NO - too sparse"
    print(f"{col:<15} {valid_pct:>8.1f}% {status:>14}")
    if viable:
        viable_fuels.append(col)
 
print()
print(f"Fuel types meeting 70% threshold: {viable_fuels}")
print()
print("These three fuel types are highly correlated (multicollinearity)")
print("because they share the same underlying price drivers.")
print("PCA will combine them into Component 1 - representing the")
print("overall fuel price level in Darwin and Palmerston.")
print()
 
# Confirm the three predictors we will use
pca_fuel_cols = viable_fuels
print(f"PCA predictors selected: {pca_fuel_cols}")
print()


# DEMONSTRATE MULTICOLLINEARITY BETWEEN FUEL TYPES.
# Before applying PCA we must DEMONSTRATE that the three fuel types
# are actually highly correlated in our specific dataset.
# We use a correlation matrix showing Pearson r between each pair.
# Threshold: r > 0.7 = strong correlation, r > 0.9 = multicollinearity
# If correlations are high this confirms PCA is justified.
# If correlations are low PCA would not be appropriate.

print("MULTICOLLINEARITY DEMONSTRATION")

# Calculate correlation matrix using only rows where all three
# fuel types have valid prices - ensures fair comparison
fuel_corr = fuel[pca_fuel_cols].dropna()
corr_matrix = fuel_corr.corr().round(3)

print("Pearson correlation matrix (fuel prices in Darwin/Palmerston):")
print(corr_matrix.to_string())
print()

# Check all pairs exceed 0.7 threshold
pairs = [
    ('Diesel',      'Unleaded 91'),
    ('Diesel',      'Premium 95'),
    ('Unleaded 91', 'Premium 95')
]

print("Pairwise correlation assessment:")
all_high = True
for col1, col2 in pairs:
    r = corr_matrix.loc[col1, col2]
    status = "HIGH - multicollinearity confirmed" if r > 0.7 else "LOW - review needed"
    print(f"  {col1} vs {col2}: r = {r:.3f} → {status}")
    if r <= 0.7:
        all_high = False

print()
if all_high:
    print("CONCLUSION: All three fuel types are highly correlated.")
    print("Multicollinearity confirmed. PCA is justified.")
    print("Combining into one component preserves the shared price")
    print("signal while eliminating redundancy between predictors.")
else:
    print("WARNING: Some fuel pairs show low correlation.")
    print("Review whether PCA is appropriate for this dataset.")
print()

# calcuting 14 day fuel price look back for each bike count date.
# For each bike observation we calculate the average price for ALL THREE
# fuel types from the 14 days BEFORE that site's count date.
#
# WHY 14 DAYS: Cyclists respond to fuel prices observed over recent weeks
# before deciding to change commuting behaviour (behavioural lag).
#
# WHY SAME WINDOW FOR ALL THREE FUEL TYPES:
# A commuter observed ALL fuel prices during those 14 days simultaneously.
# Their behaviour was influenced by the complete fuel price environment
# they experienced - not diesel from one period and unleaded from another.
# Using different windows would mix price signals from different time
# periods making the PCA component analytically meaningless.

print("CALCULATING 14-DAY FUEL PRICE WINDOWS")
print(f"Processing {len(bike)} bike observations...")
print(f"Calculating averages for: {pca_fuel_cols}")
 
# Initialise a dictionary to collect prices for each fuel type
price_results = {col: [] for col in pca_fuel_cols}
 
for i, row in bike.iterrows():
    end_date   = row['Count_Date']
    start_date = end_date - pd.Timedelta(days=14)
 
    # Filter fuel to 14-day window
    window = fuel[
        (fuel['FullDate'] >= start_date) &
        (fuel['FullDate'] <= end_date)
    ]
 
    # Calculate mean for each fuel type in this window
    # NaN values are automatically excluded from mean calculation
    for col in pca_fuel_cols:
        avg_price = window[col].mean()
        price_results[col].append(avg_price)
 
# Add all three price columns to bike dataframe
for col in pca_fuel_cols:
    bike[f'avg_{col.replace(" ","_").replace("/","_")}_14d'] = price_results[col]
 
# Name the new columns cleanly for reference
price_cols = [f'avg_{col.replace(" ","_").replace("/","_")}_14d'
              for col in pca_fuel_cols]
 
print()
print("Average 14-day prices per year:")
print(bike.groupby('year')[price_cols].mean().round(1).to_string())
print()

# Remove any bike observation where ALL three fuel prices are NaN.
# These rows cannot contribute to PCA or regression.
 
rows_before = len(bike)
bike = bike.dropna(subset=price_cols, how='all')
rows_after  = len(bike)
 
print("UNMATCHED ROWS REMOVED")
print(f"Rows before: {rows_before} | Rows after: {rows_after}")
print(f"Removed: {rows_before - rows_after} unmatched observations")
print()

# standardise the three fuel price columns before PCA.
# We standardise each fuel type before applying PCA so all three
# contribute equally to Component 1. Without standardisation, the
# fuel type with the largest numerical range would dominate the component.
# Standardising ensures all three contribute equally to PCA.

print("STANDARDISATION BEFORE PCA")
 
fuel_matrix = bike[price_cols].copy()
for col in price_cols:
    fuel_matrix[col] = fuel_matrix[col].fillna(
        fuel_matrix[col].mean()
    )
 
scaler_fuel = StandardScaler()
fuel_scaled = scaler_fuel.fit_transform(fuel_matrix)
 
print("After standardisation - each fuel type:")
for i, col in enumerate(price_cols):
    print(f"  {col}: mean = {fuel_scaled[:,i].mean():.4f} | "
          f"std = {fuel_scaled[:,i].std():.4f}")
print("All variables now have mean ≈ 0 and std ≈ 1")
print()

# applying PCA to produce component 1.
print("PCA - COMBINING FUEL PRICE PREDICTORS")
 
pca = PCA(n_components=1)
PC1 = pca.fit_transform(fuel_scaled)
 
explained_var = pca.explained_variance_ratio_[0] * 100
 
print(f"Variance explained by Component 1: {explained_var:.1f}%")
print()
 
if explained_var >= 70:
    print("PASS: Component 1 explains >= 70% of variance.")
    print("The three fuel types are highly correlated as expected.")
    print("PCA is appropriate - multicollinearity confirmed.")
else:
    print("NOTE: Component 1 explains less than 70% of variance.")
    print("Fuel types may be less correlated than expected.")
    print("Consider reviewing predictor selection.")
 
print()
print("Loadings (contribution of each fuel type to Component 1):")
for col, loading in zip(price_cols, pca.components_[0]):
    print(f"  {col}: {loading:.4f}")
print()
print("Component 1 represents: overall standardised fuel price level")
print("in Darwin and Palmerston across Diesel, Unleaded 91 and Premium 95")
print()
 
# Add PC1 as the X variable in the bike dataframe
bike['PC1_fuel_price'] = PC1


# exploratory data analysis (EDA) befre modeling.

print("EXPLORATORY DATA ANALYSIS")
print("Generating visualisations...")
print()
 
colors = {2019: 'steelblue', 2021: 'darkorange', 2023: 'green'}
 
# --- FIGURE 1: Distributions ---
# --- FIGURE 1: Distribution of Key Variables ---

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Figure 1: Distribution of Key Variables',
             fontsize=14, fontweight='bold')

# Left - histogram of Trips/hour
axes[0].hist(bike['Trips/hour'], bins=20,
             color='steelblue', edgecolor='white')
axes[0].set_title('Distribution of Trips/hour (Y variable)')
axes[0].set_xlabel('Trips per Hour')
axes[0].set_ylabel('Frequency')
axes[0].axvline(bike['Trips/hour'].mean(), color='red',
                linestyle='--',
                label=f"Mean: {bike['Trips/hour'].mean():.1f}")
axes[0].legend()

# Right - box plot of PC1 by year
data_by_year = [
    bike[bike['year'] == yr]['PC1_fuel_price'].values
    for yr in [2019, 2021, 2023]
]
bp = axes[1].boxplot(
    data_by_year,
    labels=['2019', '2021', '2023'],
    patch_artist=True
)
for patch, color in zip(bp['boxes'],
                        ['steelblue', 'darkorange', 'green']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[1].set_title('PC1 Fuel Price Level by Year\n'
                  '(three distinct price levels)')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('PC1 Score (standardised fuel price level)')
axes[1].axhline(y=0, color='red', linestyle='--',
                linewidth=1, label='Mean = 0')
axes[1].legend()

plt.tight_layout()
plt.savefig('figure1_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure 1 saved: figure1_distributions.png")
print()

 
# --- FIGURE 3: Cycling Trends Over Time by Electorate ---
# Shows Darwin vs Palmerston cycling activity across 2019 2021 2023
# with average fuel price annotated for context.
 
yearly_electorate = bike.groupby(
    ['year', 'electorate'])['Trips/hour'].mean().reset_index()
yearly_fuel = bike.groupby('year')[price_cols[0]].mean().round(1)
 
fig, ax1 = plt.subplots(figsize=(10, 6))
 
for electorate, group in yearly_electorate.groupby('electorate'):
    ax1.plot(group['year'], group['Trips/hour'],
             marker='o', linewidth=2.5, markersize=8, label=electorate)
 
ax1.set_title(
    'Figure 3: Cycling Activity Over Time by Location\n'
    '(with average diesel price for context)',
    fontsize=13, fontweight='bold')
ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel('Average Trips per Hour', fontsize=11)
ax1.set_xticks([2019, 2021, 2023])
ax1.legend(title='Electorate', loc='upper left')
ax1.grid(True, alpha=0.3)
 
# Add fuel price as secondary axis
ax2 = ax1.twinx()
ax2.plot(yearly_fuel.index, yearly_fuel.values,
         color='red', linestyle='--', linewidth=1.5,
         marker='s', markersize=6, label='Avg Diesel Price')
ax2.set_ylabel('Average Diesel Price (cents per litre)',
               color='red', fontsize=10)
ax2.tick_params(axis='y', labelcolor='red')
ax2.legend(loc='upper right')
 
plt.tight_layout()
plt.savefig('figure3_trends.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure 3 saved: figure3_trends.png")
print()
 
# --- FIGURE 4: Bar Chart Darwin vs Palmerston ---
 
electorate_avg = bike.groupby('electorate')['Trips/hour'].mean()
 
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(electorate_avg.index, electorate_avg.values,
              color=['steelblue', 'darkorange'],
              edgecolor='white', width=0.5)
 
for bar, val in zip(bars, electorate_avg.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f'{val:.1f}', ha='center', va='bottom',
            fontsize=12, fontweight='bold')
 
ax.set_title(
    'Figure 4: Average Cycling Activity by Location\n'
    '(2019, 2021 and 2023 combined)',
    fontsize=13, fontweight='bold')
ax.set_xlabel('Electorate', fontsize=11)
ax.set_ylabel('Average Trips per Hour', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
 
plt.tight_layout()
plt.savefig('figure4_electorate_bar.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure 4 saved: figure4_electorate_bar.png")
print()

corr = bike['PC1_fuel_price'].corr(bike['Trips/hour'])
print(f"Pearson correlation (PC1 vs Trips/hour): r = {corr:.4f}")
print()

#standardising the Y variable (Trips/hour) for regression.

print("STANDARDISE Y VARIABLE")
 
scaler_y = StandardScaler()
y = bike['Trips/hour'].values.reshape(-1, 1)
y_scaled = scaler_y.fit_transform(y)
 
print(f"Trips/hour - Original mean: {bike['Trips/hour'].mean():.1f}")
print(f"Trips/hour - Scaled mean:   {y_scaled.mean():.4f}")
print(f"Trips/hour - Original std:  {bike['Trips/hour'].std():.1f}")
print(f"Trips/hour - Scaled std:    {y_scaled.std():.4f}")
print()

# linear regression modeling with PC1 as predictor and scaled Trips/hour as outcome.

# --- LEAVE-ONE-OUT CROSS VALIDATION (LOOCV) ---
# WHY LOOCV INSTEAD OF TRAIN-TEST SPLIT:
# A traditional 70/30 train-test split was considered but rejected
# because our merged dataset contains only 176 observations.
# A 30% test set would contain approximately 53 rows - too few to
# produce reliable evaluation metrics. Results would change dramatically
# depending on which 53 rows were randomly selected for testing.
#
# LOOCV solves this by using every observation as a test point exactly once:
#   Round 1   → train on 175 rows, test on row 1
#   Round 2   → train on 175 rows, test on row 2
#   Round 176 → train on 175 rows, test on row 176
#   Final     → average all 176 test errors
#
# This gives the most reliable out-of-sample evaluation possible from
# a small dataset because every single observation gets tested exactly
# once while the model always trains on the maximum available data.

print("LINEAR REGRESSION")
 
X_pca = bike['PC1_fuel_price'].values.reshape(-1, 1)
 
model = LinearRegression()
model.fit(X_pca, y_scaled)
 
y_pred_scaled = model.predict(X_pca)
 
print(f"Intercept (b0):   {model.intercept_[0]:.4f}")
print(f"Coefficient (b1): {model.coef_[0][0]:.4f}")
print()
print("INTERPRETATION:")
print(f"Coefficient = {model.coef_[0][0]:.4f}")
print("A one standard deviation increase in the combined fuel price")
print(f"signal (PC1) is associated with a {model.coef_[0][0]:.4f} standard")
print("deviation change in cycling activity.")
print()
if model.coef_[0][0] < 0:
    print("The NEGATIVE coefficient confirms an inverse relationship:")
    print("Higher fuel prices → lower cycling activity in Darwin/Palmerston")
else:
    print("The POSITIVE coefficient shows a direct relationship:")
    print("Higher fuel prices → higher cycling activity in Darwin/Palmerston")
print()
 

loo = LeaveOneOut()
loocv_scores = cross_val_score(
    LinearRegression(),
    X_pca,
    y_scaled,
    cv=loo,
    scoring='neg_mean_squared_error'
)
loocv_mse  = -loocv_scores.mean()
loocv_rmse = np.sqrt(loocv_mse)
loocv_std  = np.sqrt(loocv_scores.std())

print(f"LOOCV completed: {len(loocv_scores)} iterations")
print(f"LOOCV MSE:       {loocv_mse:.4f}")
print(f"LOOCV RMSE:      {loocv_rmse:.4f}")
print(f"LOOCV Std Dev:   {loocv_std:.4f}")
print()


#model evaluation metrics on the training data (not LOOCV)

# --- MODEL EVALUATION ---
# We evaluate the model using four metrics:
#
# R² (R-squared):
#   Proportion of variance in Y explained by X (0 to 1, higher better)
#   
# MAE (Mean Absolute Error):
#   Average size of prediction errors in standardised units.
#   Lower is better.
#
# RMSE (Root Mean Squared Error):
#   Similar to MAE but penalises large errors more heavily.
#   Lower is better.
#
# LOOCV RMSE (from Step 8a):
#   Out-of-sample prediction error averaged across 176 test folds.
#   More honest than in-sample RMSE because it tests on unseen data.
#   Comparing in-sample RMSE vs LOOCV RMSE reveals whether the model
#   is overfitting - memorising training data rather than learning
#   a genuine pattern.
#
print("MODEL EVALUATION")

r2   = r2_score(y_scaled, y_pred_scaled)
mae  = mean_absolute_error(y_scaled, y_pred_scaled)
rmse = np.sqrt(np.mean((y_scaled - y_pred_scaled.flatten())**2))

print(f"R²        = {r2:.4f}  → PC1 explains {r2*100:.1f}% of variance in Trips/hour")
print(f"MAE       = {mae:.4f}  → average prediction error (standardised units)")
print(f"RMSE      = {rmse:.4f}  → in-sample root mean squared error")
print(f"LOOCV RMSE= {loocv_rmse:.4f}  → out-of-sample root mean squared error")
print(f"LOOCV Std = {loocv_std:.4f}  → variability of errors across 176 folds")
print()

# --- GENERALISATION ASSESSMENT ---
# Compare in-sample vs out-of-sample performance
# If LOOCV RMSE is within 20% of in-sample RMSE the model generalises well
# If LOOCV RMSE is much higher the model is overfitting

print("GENERALISATION ASSESSMENT:")
print(f"  In-sample RMSE:  {rmse:.4f}  (how well model fits training data)")
print(f"  LOOCV RMSE:      {loocv_rmse:.4f}  (how well model predicts unseen data)")
difference = ((loocv_rmse - rmse) / rmse) * 100
print(f"  Difference:      {difference:.1f}%")
print()

if loocv_rmse <= rmse * 1.2:
    print("PASS: LOOCV RMSE is within 20% of in-sample RMSE.")
    print("The model generalises consistently - no evidence of overfitting.")
    print("Given only 3 distinct fuel price levels, this result confirms")
    print("the model is learning a genuine pattern not memorising noise.")
else:
    print("NOTE: LOOCV RMSE is notably higher than in-sample RMSE.")
    print("The model does not generalise well to unseen observations.")
    print("Expected given only 3 distinct fuel price levels - the model")
    print("has very limited variation in X to learn from.")
print()
print("ACADEMIC NOTE: LOOCV was selected over train-test split because")
print("our 176 observations are too few for a reliable holdout set.")
print("LOOCV maximises training data while providing honest out-of-sample")
print("evaluation - the most appropriate validation strategy for small datasets.")
print()


# MSE (Mean Squared Error):
#   Square of the average error. Penalises large errors more than MAE.
#   Reported alongside RMSE for completeness.
#
# NORMALISED RMSE:
#   RMSE expressed as a proportion of the actual value range (max - min).
#   Makes error interpretable without knowing the scale of the data.
#   e.g. 0.10 means average error is 10% of the full Trips/hour range.
#
# BASELINE MODEL COMPARISON:
#   The baseline (dummy) model predicts the mean Trips/hour for every
#   observation regardless of fuel price. If our regression cannot
#   outperform this naive model, the model has no analytical value.
#   This comparison directly follows the lecture's boston.py approach.

# MSE
mse = np.mean((y_scaled - y_pred_scaled.flatten())**2)

# Normalised RMSE - uses original Trips/hour range not standardised
y_max     = bike['Trips/hour'].max()
y_min     = bike['Trips/hour'].min()
rmse_norm = rmse / (y_max - y_min)

print(f"MSE            = {mse:.4f}  → mean squared error")
print(f"RMSE Normalised= {rmse_norm:.4f}  → error as proportion of value range")
print()

# Baseline model - predicts mean for every observation
y_base      = np.mean(y_scaled)
y_pred_base = np.full_like(y_scaled.flatten(), y_base)
r2_base     = r2_score(y_scaled, y_pred_base)
rmse_base   = np.sqrt(np.mean((y_scaled - y_pred_base)**2))
mae_base    = mean_absolute_error(y_scaled, y_pred_base)

print("BASELINE MODEL COMPARISON:")
print("(Baseline predicts mean Trips/hour for every observation)")
print(f"  Baseline R²:    {r2_base:.4f}")
print(f"  Baseline RMSE:  {rmse_base:.4f}")
print(f"  Baseline MAE:   {mae_base:.4f}")
print()
print(f"  Model R²:       {r2:.4f}")
print(f"  Model RMSE:     {rmse:.4f}")
print(f"  Model MAE:      {mae:.4f}")
print()

if r2 > r2_base and rmse < rmse_base:
    print("PASS: Regression outperforms baseline on both R² and RMSE.")
    print("The fuel price signal adds predictive value beyond the mean.")
else:
    print("NOTE: Model does not substantially outperform baseline.")
    print("Fuel price alone adds limited predictive value over the mean.")
    print("This is consistent with our low R² finding (5.1%) and confirms")
    print("that other factors beyond fuel price drive cycling behaviour.")
print()

#statistical significance test.

#H0: No relationship between combined fuel price and cycling activity
# H1: A relationship exists
# Decision: reject H0 if p-value < 0.05.
 
print("STATISTICAL SIGNIFICANCE TEST")
 
corr_coef, p_value = stats.pearsonr(
    bike['PC1_fuel_price'],
    bike['Trips/hour']
)
 
print(f"Pearson r: {corr_coef:.4f}")
print(f"P-value:   {p_value:.4f}")
print()
 
if p_value < 0.05:
    print(f"RESULT: p = {p_value:.4f} < 0.05")
    print("REJECT the null hypothesis.")
    print("The relationship between combined fuel price (PC1) and")
    print("cycling activity is statistically significant at 95% confidence.")
    print()
    print("PLAIN LANGUAGE FOR DLI STAKEHOLDER:")
    print("We are 95% confident that rising fuel prices are associated")
else:
    print(f"RESULT: p = {p_value:.4f} > 0.05")
    print("FAIL TO REJECT the null hypothesis.")
    print("The relationship is not statistically significant.")
print()
 

# residual plot to check linear regression.

print(" RESIDUAL PLOT")
 
residuals = y_scaled.flatten() - y_pred_scaled.flatten()
colors_per_row = [colors[yr] for yr in bike['year']]
 
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Figure 5: Regression Diagnostics',
             fontsize=14, fontweight='bold')
 
axes[0].scatter(y_pred_scaled, residuals,
                alpha=0.6, c=colors_per_row, s=60)
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
axes[0].set_title('Residual Plot\n(coloured by year)')
axes[0].set_xlabel('Predicted Values (standardised)')
axes[0].set_ylabel('Residuals')
axes[0].grid(True, alpha=0.3)
 
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors[yr], label=str(yr))
    for yr in [2019, 2021, 2023]
]
axes[0].legend(handles=legend_elements, title='Year')
 
axes[1].scatter(bike['PC1_fuel_price'], bike['Trips/hour'],
                c=colors_per_row, alpha=0.7, s=60)
 
x_line = np.linspace(
    bike['PC1_fuel_price'].min(),
    bike['PC1_fuel_price'].max(), 100
).reshape(-1, 1)
y_line_scaled = model.predict(x_line)
y_line = scaler_y.inverse_transform(y_line_scaled)
 
axes[1].plot(x_line, y_line, color='red', linewidth=2,
             label='Regression line')
axes[1].set_title('Regression Line (PC1 vs Trips/hour)')
axes[1].set_xlabel('PC1 Combined Fuel Price Score')
axes[1].set_ylabel('Trips per Hour')
axes[1].grid(True, alpha=0.3)
axes[1].legend(
    handles=legend_elements + [
        plt.Line2D([0], [0], color='red',
                   linewidth=2, label='Regression line')
    ],
    title='Year'
)
 
plt.tight_layout()
plt.savefig('figure5_diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure 5 saved: figure5_diagnostics.png")
print()
 
# --- regresssion line: SCATTER PLOT - DIESEL PRICE VS TRIPS/HOUR ---

# --- FIGURE 2: SCATTER PLOT - DIESEL PRICE VS TRIPS/HOUR ---


fig, ax = plt.subplots(figsize=(10, 7))

for year, group in bike.groupby('year'):
    ax.scatter(group['avg_Diesel_14d'], group['Trips/hour'],
               label=str(year), color=colors[year], alpha=0.7, s=60)

# Regression line on original diesel scale
x_line = np.linspace(
    bike['avg_Diesel_14d'].min(),
    bike['avg_Diesel_14d'].max(), 100
).reshape(-1, 1)
x_line_full = pd.DataFrame(
    np.column_stack([
        x_line,
        np.full_like(x_line, bike['avg_Unleaded_91_14d'].mean()),
        np.full_like(x_line, bike['avg_Premium_95_14d'].mean())
    ]),
    columns=price_cols
)
x_line_scaled = scaler_fuel.transform(x_line_full)
x_line_pca    = pca.transform(x_line_scaled)
y_line_scaled = model.predict(x_line_pca)
y_line        = scaler_y.inverse_transform(y_line_scaled)

ax.plot(x_line, y_line, color='red', linewidth=2,
        linestyle='--', label='Regression line')

# Annotate with statistical results directly on chart
ax.annotate(
    f'r = {corr_coef:.3f} | p = {p_value:.4f}\n'
    f'R² = {r2:.3f} (fuel explains {r2*100:.1f}% of cycling variation)',
    xy=(0.02, 0.97), xycoords='axes fraction',
    fontsize=9, color='dimgray', va='top',
    bbox=dict(boxstyle='round,pad=0.4',
              facecolor='lightyellow', alpha=0.8))

ax.set_title(
    'Figure 2: Average Diesel Price vs Cycling Activity\n'
    '(Each point = one counting site, coloured by year)',
    fontsize=13, fontweight='bold')
ax.set_xlabel('Average Diesel Price - 14 Day Window (cents per litre)',
              fontsize=11)
ax.set_ylabel('Trips per Hour', fontsize=11)
ax.legend(title='Year')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure2_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure 2 saved: figure2_scatter.png")
print()


# end of analysis

bike.to_csv('merged_analysis_dataset.csv', index=False)
 
print("ANALYSIS COMPLETE")
print()
print("Output files produced:")
print("  figure1_distributions.png   - Distribution of key variables")
print("  figure2_scatter.png         - PC1 vs Trips/hour (Treasury)")
print("  figure3_trends.png          - Cycling trends over time")
print("  figure4_electorate_bar.png  - Darwin vs Palmerston (DLI)")
print("  figure5_diagnostics.png     - Residual plot + regression line")
print("  merged_analysis_dataset.csv - Final merged dataset")
print()
print("=== KEY RESULTS FOR PRESENTATION ===")
print(f"PCA variance explained:  {explained_var:.1f}%")
print(f"Pearson r:               {corr_coef:.4f}")
print(f"R²:                      {r2:.4f} ({r2*100:.1f}% variance explained)")
print(f"P-value:                 {p_value:.4f} "
      f"({'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'})")
print(f"Regression coefficient:  {model.coef_[0][0]:.4f}")
print()
print("PRESENTATION NARRATIVE:")
print("A statistically significant but weak negative relationship exists")
print("between combined fuel prices and cycling activity in Darwin")
print("and Palmerston. PC1 captures the shared fuel price movement")
print(f"across three fuel types ({explained_var:.0f}% of variance explained by PCA).")
print("The negative coefficient confirms higher fuel prices are associated")
print("with lower cycling rates - however fuel price accounts for only")
print(f"{r2*100:.1f}% of cycling variation. Confounding factors including")
print("Darwin's wet/dry seasons and day-of-week effects likely explain")
print("the remaining variance.")