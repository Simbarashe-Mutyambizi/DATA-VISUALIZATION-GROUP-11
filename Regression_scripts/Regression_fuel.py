import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import metrics


# Read dataset into a DataFrame
df = pd.read_csv("boston.csv")

# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)


# compare the cumulative explained variance versus number of PCA components
pca = PCA().fit(X_train)

# Plot the cumulative explained variance versus number of PCA components
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xticks(range(1,13))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.grid()
plt.show()

# Train a linear regression on PCA-transformed training data (top-5 components)
pca = PCA(n_components=5)
X_train_p = pca.fit_transform(X_train)

# Compare the dimensionality of the original data vs. its dimensionality reduced version
print("Dimension of original data:", X_train.shape)
print("Dimension of PCA-reduced data:", X_train_p.shape)

# Build a linear regression model
model = LinearRegression()
model.fit(X_train_p, y_train)

# Use linear regression to predict the values of (y) in the training set
y_pred = model.predict(X_train_p)

# Get R-Squared score
r_2 = metrics.r2_score(y_train, y_pred)
print("Trained with 5-component PCA:")
print("R^2: ", r_2)

