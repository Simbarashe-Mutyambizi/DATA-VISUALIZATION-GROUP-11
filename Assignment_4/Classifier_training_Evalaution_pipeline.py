import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score
)

start = time.time()

df = pd.read_csv(r"Datasets\NT\Final_Fuel_Dataset_NT_V3.csv")

y = df["Price_Class"]
X = df.drop(columns=["Price_Class"])

drop_cols = [c for c in ["Unnamed: 0", "FullDate"] if c in X.columns]
X = X.drop(columns=drop_cols)

encoded_cols = [c for c in X.columns if c.endswith("_encoded")]
if encoded_cols:
    X = X.drop(columns=encoded_cols)

categorical_features = [c for c in X.columns if X[c].dtype == "object"]
numeric_features = [c for c in X.columns if c not in categorical_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ],
    remainder="drop"
)

models = {
    "LGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42),
    "CatBoost": CatBoostClassifier(iterations=300, learning_rate=0.05, depth=8, verbose=0, random_state=42),
    "Random_Forest": RandomForestClassifier(
        n_estimators=500, max_depth=20  , min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt", bootstrap=True,
        random_state=42, n_jobs=-1
    )
}

output_dir = "Error Metric Plots"
os.makedirs(output_dir, exist_ok=True)

results = []
cv_results = []
classes = np.sort(y.unique())
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = len(classes)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_results.append({
        "Model": name,
        "CV_Mean_Accuracy": cv_scores.mean(),
        "CV_Std_Accuracy": cv_scores.std()
    })

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1
    })

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{name}_confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    if hasattr(pipe.named_steps["model"], "predict_proba"):
        y_score = pipe.predict_proba(X_test)
    else:
        y_score = pipe.decision_function(X_test)
        if y_score.ndim == 1:
            y_score = np.column_stack([-y_score, y_score])

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test_bin[:, 1], y_score[:, 1])
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{name} - ROC Curve")
        ax.legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{name}_roc_curve.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        try:
            roc_auc = roc_auc_score(y_test_bin, y_score, multi_class="ovr", average="macro")
        except Exception:
            roc_auc = np.nan

results_df = pd.DataFrame(results)
cv_df = pd.DataFrame(cv_results)

results_df.to_csv(os.path.join(output_dir, "model_metrics.csv"), index=False)
cv_df.to_csv(os.path.join(output_dir, "cross_validation_metrics.csv"), index=False)

print("Test-set metrics:")
print(results_df)
print("\n5-fold CV metrics:")
print(cv_df)

end = time.time()
print(f"Elapsed: {end - start:.2f} seconds")