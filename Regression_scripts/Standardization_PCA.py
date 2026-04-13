import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def multilinear_regression_report_from_split(X_train, X_test, y_train, y_test,feature_names=None, p_decimals=3, outfile=None):
    """
    Inputs:
      X_train, X_test: 2D arrays or DataFrames
      y_train, y_test: 1D arrays or Series
      feature_names: list of column names (optional). If None, features are named 'x0'...
      p_decimals: decimals for formatted p-values
      outfile: path to write text summary (optional)
    Returns:
      report dict with model, coefficients (with names), p_values (formatted), train_metrics, test_metrics
    """
    # ensure pandas DataFrame/Series for names and statsmodels
    X_train = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train
    X_test  = pd.DataFrame(X_test)  if not isinstance(X_test, pd.DataFrame)  else X_test
    y_train = pd.Series(y_train)    if not isinstance(y_train, pd.Series)   else y_train
    y_test  = pd.Series(y_test)     if not isinstance(y_test, pd.Series)    else y_test

    if feature_names:
        X_train.columns = feature_names
        X_test.columns = feature_names
    else:
        feature_names = [str(c) for c in X_train.columns]

    # fit sklearn linear model
    lr = LinearRegression().fit(X_train, y_train)

    def metrics(y_true, y_pred, p):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        n = len(y_true)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else float("nan")
        return {"mse": float(mse), "rmse": float(rmse), "mae": float(mae),
                "r2": float(r2), "adj_r2": float(adj_r2)}

    train_metrics = metrics(y_train, lr.predict(X_train), X_train.shape[1])
    test_metrics  = metrics(y_test,  lr.predict(X_test),  X_test.shape[1])

    # statsmodels OLS for p-values on full data (use concatenated X and y)
    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)
    X_sm = sm.add_constant(X_full)
    ols = sm.OLS(y_full, X_sm).fit()

    coeffs = {"intercept": float(lr.intercept_)}
    coeffs.update(dict(zip(feature_names, lr.coef_.tolist())))

    p_values = {k: float(v) for k, v in ols.pvalues.items()}              # numeric
    p_values_fmt = {k: f"{v:.{p_decimals}f}" for k, v in p_values.items()} # formatted strings

    report = {
        "model_sklearn": lr,
        "model_ols": ols,
        "coefficients": coeffs,
        "p_values": p_values_fmt,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    if outfile:
        with open(outfile, "w") as f:
            f.write(f"Coefficients: {report['coefficients']}\n")
            f.write(f"P-values: {report['p_values']}\n\n")
            f.write(f"Train metrics: {report['train_metrics']}\n")
            f.write(f"Test metrics: {report['test_metrics']}\n")

    return report


df = pd.read_csv(r"Regression_scripts\Regression_dataframes_fuel\Diesel.csv")

# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

# training and test splits on original data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)

# apply standardisation to explanatory variables
std_scaler = preprocessing.StandardScaler()
x_train_std = std_scaler.fit_transform(x_train)
x_test_std = std_scaler.transform(x_test)

# after your scaling code
x_train_std = std_scaler.fit_transform(x_train)
x_test_std = std_scaler.transform(x_test)

# convert to DataFrame with original feature names
feature_names = list(df.columns[:-1])
X_train_std_df = pd.DataFrame(x_train_std, columns=feature_names)
X_test_std_df  = pd.DataFrame(x_test_std,  columns=feature_names)

print(X_train_std_df.shape)
print(X_train_std_df.head())

# optional: quick summaries
print(X_train_std_df.describe().T)
# # initialise PCA
# pca = PCA(n_components=5)
# # PCA on standardised explanatory variables
# x_train_std = pca.fit_transform(x_train_std)
# x_test_std = pca.transform(x_test_std)

# feature_names = list(df.columns[:-1])
# report = multilinear_regression_report_from_split(x_train_std, x_test_std, y_train, y_test,
#                                                  feature_names=feature_names, outfile="Test.txt")

