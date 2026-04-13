#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import metrics


# Formatted and setup for the regression analysis by formatting the main dataset for each of the fuel types

#%%
# Read dataset into a DataFrame
df = pd.read_csv(r"Files_to_clean\Final_Fuel_Dataset.csv")
# #%%

# For Unleaded 91

U_91 = df.drop(df.columns[[0,1,2,3,4,5,8,9,10,12,13,14,15,16,17,18,19]], axis=1)
# Move cols to the end
cols_at_end = [ 'Unleaded 91']
U_91 = U_91[[c for c in U_91 if c not in cols_at_end] + cols_at_end]
U_91.to_csv(r"Regression_scripts\Unleaded_91.csv", index=False)

#%%
# Diesel
D = df.drop(df.columns[[0,1,2,3,4,5,9,10,11,12,13,14,15,16,17,18,19]], axis=1)
# Move cols to the end
cols_at_end = [ 'Diesel']
D = D[[c for c in D if c not in cols_at_end] + cols_at_end]
D.to_csv(r"Regression_scripts\Diesel.csv", index=False)

#%%
# Premium 98
P_98 = df.drop(df.columns[[0,1,2,3,4,5,8,10,11,12,13,14,15,16,17,18,19]], axis=1)
# Move cols to the end
cols_at_end = [ 'Premium 98']
P_98 = P_98[[c for c in P_98 if c not in cols_at_end] + cols_at_end]
P_98.to_csv(r"Regression_scripts\Premium_98.csv", index=False)


#%%
# Premium 95
P_95 = df.drop(df.columns[[0,1,2,3,4,5,8,9,11,12,13,14,15,16,17,18,19]], axis=1)
# Move cols to the end
cols_at_end = [ 'Premium 95']
P_95 = P_95[[c for c in P_95 if c not in cols_at_end] + cols_at_end]
P_95.to_csv(r"Regression_scripts\Premium_95.csv", index=False)



#%%
# df = pd.read_csv(r"C:\Personal\Masters\Masters_work\Study\Y1_S2\PRT564\Assignments\Assignment_2\Repo\DATA-VISUALIZATION-GROUP-11\Regression_scripts\Regression_dataframes\Unleaded_91.csv")

# Separate explanatory variables (x) from the response variable (y)
# x = df.iloc[:,:-1].values
# y = df.iloc[:,-1].values

# # Split dataset into 60% training and 40% test sets 
# # Note: other % split can be used.
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)





# # compare the cumulative explained variance versus number of PCA components
# pca = PCA().fit(X_train)

# # Plot the cumulative explained variance versus number of PCA components
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xticks(range(1,13))
# plt.xlabel('Number of components')
# plt.ylabel('Cumulative explained variance')
# plt.grid()
# plt.show()

# # Train a linear regression on PCA-transformed training data (top-5 components)
# pca = PCA(n_components=6)
# X_train_p = pca.fit_transform(X_train)

# # Compare the dimensionality of the original data vs. its dimensionality reduced version
# print("Dimension of original data:", X_train.shape)
# print("Dimension of PCA-reduced data:", X_train_p.shape)

# # Build a linear regression model
# model = LinearRegression()
# model.fit(X_train_p, y_train)

# # Use linear regression to predict the values of (y) in the training set
# y_pred = model.predict(X_train_p)

# # Get R-Squared score
# r_2 = metrics.r2_score(y_train, y_pred)
# print("Trained with 5-component PCA:")
# print("R^2: ", r_2)


# %%
