import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

# =========================
# LOAD DATA
# =========================

df_original = pd.read_csv(
    r"Datasets\NT\Final_Fuel_Dataset_NT_V3.csv"
)

# Create a working copy
df = df_original.copy()

# =========================
# DATE PROCESSING
# =========================

df["FullDate"] = pd.to_datetime(
    df["FullDate"],
    format="mixed",
    dayfirst=True,
    errors="coerce"
)

# Remove invalid dates
df = df.dropna(subset=["FullDate"])

# Extract date features
df["Year"] = df["FullDate"].dt.year
df["Month"] = df["FullDate"].dt.month
df["Day"] = df["FullDate"].dt.day

# Drop original date column
df.drop("FullDate", axis=1, inplace=True)

# Remove label column for clustering
if "Price_Class" in df.columns:
    df.drop("Price_Class", axis=1, inplace=True)

# =========================
# FEATURE GROUPS
# =========================

numeric_features = [
    "Lat",
    "Long",
    "Unleaded 91",
    "Year",
    "Month",
    "Day"
]

categorical_features = [
    "Region Name_encoded",
    "Suburb_encoded",
    "Brand Name_encoded"
]

# =========================
# NUMERIC PIPELINE
# =========================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# =========================
# CATEGORICAL PIPELINE
# =========================

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# =========================
# COLUMN TRANSFORMER
# =========================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# =========================
# ELBOW METHOD
# =========================

inertia_values = []

cluster_range = range(1, 10)

for k in cluster_range:

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("kmeans", KMeans(
            n_clusters=k,
            init="k-means++",
            random_state=42,
            n_init=10
        ))
    ])

    model.fit(df)

    inertia = model.named_steps["kmeans"].inertia_
    inertia_values.append(inertia)

    print(f"k = {k}, Inertia = {inertia}")

# =========================
# CALCULATE VARIANCE REDUCTION
# =========================

initial_inertia = inertia_values[0]

variance_reduction = [
    (1 - (i / initial_inertia)) * 100
    for i in inertia_values
]

# =========================
# PLOT ELBOW GRAPH
# =========================

plt.figure(figsize=(10, 6))

plt.plot(
    cluster_range,
    variance_reduction,
    marker='o',
    linewidth=2
)

plt.xticks(cluster_range)

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Reduction in Variance (%)")
plt.title("Elbow Plot Using Reduction in Variance")
plt.grid(True)

plt.savefig(r"Clustering_Error_Metric\NT_Fuel_Elbow_plot.png")

plt.show()

# =========================
# FINAL K-MEANS MODEL
# =========================

# Choose optimal k from elbow plot
optimal_k = 10

final_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("kmeans", KMeans(
        n_clusters=optimal_k,
        init="k-means++",
        random_state=42,
        n_init=10
    ))
])

# Fit final model
final_model.fit(df)

# =========================
# CREATE CLUSTER COLUMN
# =========================

df["Cluster"] = (
    final_model.named_steps["kmeans"].labels_
)

# =========================
# ADD CLUSTER TO ORIGINAL DATASET
# =========================

# Match rows after dropped invalid dates
df_original = df_original.loc[df.index]

# Add cluster column
df_original["Cluster"] = df["Cluster"].values

# =========================
# SAVE FINAL DATASET
# =========================

df_original.to_csv(
    r"Datasets\NT\Final_Fuel_Dataset_NT_V3_With_Clusters.csv",
    index=False
)

print("\nCluster column added successfully.")
print(df_original.head())

# =========================
# OPTIONAL: VISUALIZE CLUSTERS
# =========================

# plt.figure(figsize=(12, 8))

# scatter = plt.scatter(
#     df_original["Long"],
#     df_original["Lat"],
#     c=df_original["Cluster"],
#     cmap="viridis",
#     alpha=0.7
# )

# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.title("Fuel Station Clusters")

# plt.colorbar(scatter, label="Cluster")

# plt.savefig(r"Clustering_Error_Metric\NT_Fuel_Clusters_Map.png")

plt.show()