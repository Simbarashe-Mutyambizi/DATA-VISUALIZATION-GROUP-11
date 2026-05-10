import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from sklearn.decomposition import PCA

# =====================================================
# LOAD DATASET WITH CLUSTERS
# =====================================================

df = pd.read_csv(
    r"Datasets\NT\Final_Fuel_Dataset_NT_V3_With_Clusters.csv"
)

print("\nDataset Loaded Successfully")
print(df.head())

# =====================================================
# BASIC INFORMATION
# =====================================================

print("\n==============================")
print("DATASET INFORMATION")
print("==============================")

print(df.info())

# =====================================================
# CLUSTER DISTRIBUTION
# =====================================================

print("\n==============================")
print("CLUSTER COUNTS")
print("==============================")

cluster_counts = df["Cluster"].value_counts().sort_index()

print(cluster_counts)

# Plot cluster distribution
plt.figure(figsize=(10,6))

cluster_counts.plot(kind="bar")

plt.xlabel("Cluster")
plt.ylabel("Number of Records")
plt.title("Cluster Distribution")

plt.grid(True)

plt.savefig("Clustering_Error_Metric\Cluster_Distribution.png")

plt.show()

# =====================================================
# NUMERIC FEATURE SUMMARY
# =====================================================

print("\n==============================")
print("CLUSTER SUMMARY STATISTICS")
print("==============================")

cluster_summary = df.groupby("Cluster").mean(
    numeric_only=True
)

print(cluster_summary)

# Save summary
cluster_summary.to_csv(
    "Cluster_Summary_Statistics.csv"
)

# =====================================================
# FUEL PRICE DISTRIBUTION
# =====================================================

if "Unleaded 91" in df.columns:

    plt.figure(figsize=(12,6))

    df.boxplot(
        column="Unleaded 91",
        by="Cluster"
    )

    plt.title("Fuel Price Distribution by Cluster")
    plt.suptitle("")

    plt.xlabel("Cluster")
    plt.ylabel("Fuel Price")

    plt.savefig("Clustering_Error_Metric\Fuel_Price_By_Cluster.png")

    plt.show()

# =====================================================
# GEOGRAPHIC CLUSTER VISUALIZATION
# =====================================================

if "Lat" in df.columns and "Long" in df.columns:

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Long"], df["Lat"]),
        crs="EPSG:4326"
    )

    gdf = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(14, 10))

    gdf.plot(
        ax=ax,
        column="Cluster",
        cmap="tab20",          # more contrasting than viridis
        categorical=True,
        legend=True,
        markersize=35,         # slightly larger points
        alpha=0.95,            # less transparent
        edgecolor="black",     # outline to separate from map
        linewidth=0.4
    )

    ctx.add_basemap(
        ax,
        source=ctx.providers.Esri.WorldImagery
    )

    ax.set_axis_off()

    plt.title("Geographic Cluster Visualization with Satellite Overlay")

    plt.savefig(
        r"Clustering_Error_Metric\Geographic_Clusters_Satellite.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.show()

# =====================================================
# BRAND ANALYSIS
# =====================================================

if "Brand Name_encoded" in df.columns:

    brand_cluster = pd.crosstab(
        df["Brand Name_encoded"],
        df["Cluster"]
    )

    print("\n==============================")
    print("BRAND VS CLUSTER")
    print("==============================")

    print(brand_cluster)

    brand_cluster.plot(
        kind="bar",
        figsize=(14,7)
    )

    plt.title("Brand Distribution Across Clusters")

    plt.xlabel("Brand")
    plt.ylabel("Count")

    plt.savefig(
        r"Clustering_Error_Metric\Brand_Cluster_Distribution.png"
    )

    plt.show()

# =====================================================
# PCA VISUALIZATION
# =====================================================

print("\n==============================")
print("PCA VISUALIZATION")
print("==============================")

# Remove non-numeric columns
X = df.select_dtypes(include=np.number)

# Remove cluster column from PCA features
X_features = X.drop("Cluster", axis=1)

# PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_features)

# Plot PCA
plt.figure(figsize=(12,8))

scatter = plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df["Cluster"],
    cmap="viridis",
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("PCA Projection of Clusters")

plt.colorbar(scatter, label="Cluster")

plt.savefig("Clustering_Error_Metric\PCA_Cluster_Visualization.png")

plt.show()

# =====================================================
# CLUSTER INTERPRETATION
# =====================================================

print("\n==============================")
print("CLUSTER INTERPRETATION")
print("==============================")

for cluster in sorted(df["Cluster"].unique()):

    subset = df[df["Cluster"] == cluster]

    print(f"\nCluster {cluster}")
    print("-" * 40)

    if "Unleaded 91" in subset.columns:
        print(
            f"Average Fuel Price: "
            f"{subset['Unleaded 91'].mean():.2f}"
        )

    print(
        f"Total Records: {len(subset)}"
    )

    if "Lat" in subset.columns:
        print(
            f"Average Latitude: "
            f"{subset['Lat'].mean():.4f}"
        )

    if "Long" in subset.columns:
        print(
            f"Average Longitude: "
            f"{subset['Long'].mean():.4f}"
        )

print("\nCluster analysis complete.")