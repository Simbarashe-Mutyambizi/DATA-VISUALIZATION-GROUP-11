import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# #########################################
# Load and prepare dataset as a dataframe #

# Get the breast cancer dataset loader
cancer = load_breast_cancer()

# Get the explanatory variables (30 variables, 569 rows)
explanatory = cancer.data

# Get the response variable (1 variable, 569 rows)
response = cancer.target

# Concatenate the explanatory and response variables into a single numpy array
response = np.reshape(response, (len(response),1))
data = np.concatenate([explanatory, response], axis=1)

# Transform numpy array into a dataframe
data = pd.DataFrame(data)

# Get the feature names and use them as column names
features = cancer.feature_names
features_classes = np.append(features, "classes")
data.columns = features_classes

# Recode the values of the response variable
data["classes"].replace(0, "Benign", inplace=True)
data["classes"].replace(1, "Malignant", inplace=True)


# #############################################
# Apply standardisation to the exp. variables #

# Get the values of the explanatory variables
x = data.loc[:, features].values

# Apply standardisation
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Create another dataframe to host the standardised dataset
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
standardised_data = pd.DataFrame(x,columns=feat_cols)


# ###################################################
# Fit a 2-component PCA on the standardised dataset #

# Build and apply PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(x)

# Create another dataframe to host dimensionality reduced dataset
pca_df = pd.DataFrame(data = pca_data, columns = ['principal component 1', 'principal component 2'])

# Find out the ratio of explained variance by the 2 PCA components 
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


# ##########################################
# Plot dataset along PCA components 1 and 2 #

# Setting up figure settings
plt.figure(figsize=(8,8))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)

# Plot data points
targets = ['Benign', 'Malignant']
colors = ['r', 'g']
plt.legend(targets,prop={'size': 15})

for target, color in zip(targets,colors):
    indicesToKeep = data["classes"] == target
    plt.scatter(pca_df.loc[indicesToKeep, 'principal component 1']
               , pca_df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.show()