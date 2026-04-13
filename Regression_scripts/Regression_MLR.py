#%%
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def multilinear_regression_report(df, test_size=0.2, random_state=0, p_decimals=3, outfile=None):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    def metrics(y_true, y_pred, n_features):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        n = len(y_true)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1) if n > n_features + 1 else np.nan
        return {"mse": float(mse), "rmse": float(rmse), "mae": float(mae),
                "r2": float(r2), "adj_r2": float(adj_r2) if not np.isnan(adj_r2) else adj_r2}

    y_pred_train = lr.predict(X_train)
    y_pred_test = lr.predict(X_test)

    train_metrics = metrics(y_train, y_pred_train, X_train.shape[1])
    test_metrics = metrics(y_test, y_pred_test, X_test.shape[1])

    X_sm = sm.add_constant(X)
    ols = sm.OLS(y, X_sm).fit()

    coeffs = {"intercept": float(lr.intercept_)}
    coeffs.update(dict(zip(X.columns, lr.coef_.tolist())))

    p_values = {k: float(v) for k, v in ols.pvalues.items()}
    p_values_formatted = {k: f"{v:.{p_decimals}f}" for k, v in p_values.items()}

    report = {
        "model_sklearn": lr,
        "model_ols": ols,
        "coefficients": coeffs,
        "p_values": p_values_formatted,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    if outfile:
        lines = []
        lines.append(f"Coefficients: {report['coefficients']}\n")
        lines.append(f"P-values: {report['p_values']}\n\n")
        lines.append(f"Train metrics: {report['train_metrics']}\n")
        lines.append(f"Test metrics: {report['test_metrics']}\n")
        with open(outfile, "w") as f:
            f.writelines(lines)

    return report

def plot_coeffs_and_pvalues(report, features=None, figsize=(10,5), title="Unleaded_91_Coefficients_and_Pvalues"):
    coeffs = report["coefficients"]
    pvals = report["p_values"]
    pvals_numeric = {k: float(v) for k, v in pvals.items()}

    keys = [k for k in coeffs.keys() if k != "intercept"]
    if features:
        keys = [k for k in features if k in coeffs]

    coef_vals = [coeffs[k] for k in keys]
    pval_vals = [pvals_numeric.get(k, float("nan")) for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    bars1 = ax1.bar(keys, coef_vals, color="C0")
    ax1.set_title("Coefficients")
    ax1.set_ylabel("Coefficient")
    ax1.set_xticklabels(keys, rotation=45)

    bars2 = ax2.bar(keys, pval_vals, color="C1")
    ax2.set_title("P-values")
    ax2.set_ylabel("P-value")
    ax2.set_xticklabels(keys, rotation=45)

    # annotate bars with values
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.3g}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{title.replace(' ', '_')}.png", dpi=1000)
    return fig, (ax1, ax2)


# Example usage:
df = pd.read_csv(r"Regression_scripts\Regression_dataframes\Unleaded_91.csv")
# run regression and get report
# report = multilinear_regression_report(df, outfile="Unleaded_91_report.txt", p_decimals=3)

# print coefficients and p-values
# print("Coefficients:", report["coefficients"])
# print("P-values:", report["p_values"])

# print performance metrics
# print("Train metrics:", report["train_metrics"])
# print("Test metrics:", report["test_metrics"])

# access statsmodels summary (detaile
# %%
# report = multilinear_regression_report(df)
# fig, axes = plot_coeffs_and_pvalues(report, features=None)

# plt.show()


