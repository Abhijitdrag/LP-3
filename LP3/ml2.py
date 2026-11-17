
# ==============================================================
# MACHINE LEARNING PRACTICAL
# Experiment: K-Means and Hierarchical Clustering
"""
Question :
Implement K-Means clustering/ hierarchical clustering on 
sales_data_sample.csv dataset. Determine the number of clusters 
using the elbow method.
"""
# Objective:
# Implement K-Means and Hierarchical Clustering on
# sales_data_sample.csv and determine the optimal number
# of clusters using the Elbow Method.
#
# Algorithms:
# 1. K-Means Clustering
# 2. Hierarchical Clustering
#
# Tasks:
# 1. Load and preprocess the dataset.
# 2. Select numerical features.
# 3. Standardize the data.
# 4. Find optimal clusters using Elbow Method.
# 5. Implement K-Means Clustering.
# 6. Implement Hierarchical Clustering.
# 7. Visualize and compare the clusters.
#
# HOW TO RUN:
# Run this Python file CELL BY CELL in Jupyter Notebook,
# Google Colab or VS Code using Python cells.
#
# In VS Code, use Shift + Enter to execute each cell.
#
# Dataset:
# sales_data_sample.csv
#
# INSTALL REQUIRED LIBRARIES:
# !pip install pandas numpy matplotlib seaborn scikit-learn scipy
# ==============================================================


# ==============================================================
# CELL 1: IMPORT REQUIRED LIBRARIES
#
# Purpose:
# Import libraries required for data preprocessing,
# visualization and clustering algorithms.
#
# ==============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage


# ==============================================================
# CELL 2: LOAD THE DATASET
#
# Purpose:
# Load sales_data_sample.csv using Pandas.
# Display the first five records, dataset shape
# and information about the columns.
#
# ==============================================================

df = pd.read_csv("sales_data_sample.csv")

print("First Five Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# ==============================================================
# CELL 3: DATA PREPROCESSING
#
# Purpose:
# Clean and prepare the dataset.
#
# 1. Remove duplicate records.
# 2. Standardize column names.
# 3. Select useful numerical features.
# 4. Handle missing values using median.
#
# Why numerical features?
# K-Means calculates distances between data points,
# so it requires numerical input.
#
# ==============================================================

# Remove unnecessary spaces and convert column names to uppercase
df.columns = df.columns.str.strip().str.upper()

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Select important numerical sales features
# These columns are commonly available in this dataset.
preferred_features = [
    "QUANTITYORDERED",
    "PRICEEACH",
    "SALES",
    "MSRP"
]

# Select only features available in the dataset
features = [
    col for col in preferred_features if col in df.columns
]

# If preferred columns are unavailable,
# automatically select other suitable numerical columns.
if len(features) < 2:

    excluded_columns = [
        "ORDERNUMBER",
        "ORDERLINENUMBER",
        "CUSTOMERNUMBER",
        "YEAR_ID",
        "MONTH_ID",
        "QTR_ID",
        "POSTALCODE"
    ]

    features = [
        col for col in df.select_dtypes(include=np.number).columns
        if col not in excluded_columns
    ]

print("\nSelected Features:")
print(features)

# Convert selected columns into numerical format
X = df[features].apply(pd.to_numeric, errors="coerce")

# Replace infinite values with missing values
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove columns containing only missing values
X.dropna(axis=1, how="all", inplace=True)

# Fill missing values with median
X = X.fillna(X.median())

# Remove constant columns because they do not help clustering
X = X.loc[:, X.nunique() > 1]

if X.shape[1] < 2:
    raise ValueError("At least two useful numerical features are required.")

# Keep only rows having valid feature data
df = df.loc[X.index].copy()

print("\nDataset After Preprocessing:")
print(X.head())

print("\nFinal Dataset Shape:", X.shape)


# ==============================================================
# CELL 4: STANDARDIZE THE DATA
#
# Purpose:
# Standardize numerical features using StandardScaler.
#
# Why Scaling?
# Different features may have different ranges.
# For example:
# SALES may have large values.
# QUANTITYORDERED may have smaller values.
#
# Standardization converts features to comparable scales.
#
# StandardScaler:
# Mean = 0
# Standard Deviation = 1
#
# ==============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nStandardized Data:")
print(X_scaled[:5])


# ==============================================================
# CELL 5: FIND OPTIMAL CLUSTERS USING ELBOW METHOD
#
# Purpose:
# Determine the suitable number of clusters (K).
#
# Elbow Method:
# 1. Apply K-Means for different K values.
# 2. Calculate WCSS / Inertia for each K.
# 3. Plot K against WCSS.
# 4. Find the elbow point where the decrease in WCSS
#    starts becoming slower.
#
# WCSS = Within Cluster Sum of Squares.
#
# Lower WCSS means data points are closer to
# their respective cluster centers.
#
# ==============================================================

# Maximum number of clusters to test
max_k = min(10, len(X) - 1)

k_values = range(1, max_k + 1)

wcss = []

# Calculate WCSS for different K values
for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)

    print("K =", k, "| WCSS =", round(model.inertia_, 2))


# ==============================================================
# CELL 6: PLOT THE ELBOW GRAPH
#
# Purpose:
# Visualize WCSS values to identify the optimal K.
#
# The elbow point is selected where the curve
# starts flattening.
#
# ==============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    wcss,
    marker="o",
    linestyle="-"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS / Inertia")

plt.title("Elbow Method for Optimal K")

plt.xticks(list(k_values))
plt.grid(True)

plt.show()


# ==============================================================
# CELL 7: SUGGEST OPTIMAL K
#
# Purpose:
# Estimate the elbow point automatically using
# the maximum perpendicular distance from the line
# joining the first and last points of the WCSS curve.
#
# IMPORTANT:
# This is only a mathematical heuristic.
# Always inspect the elbow graph before final selection.
#
# The user can enter a different K based on the graph.
#
# ==============================================================

if len(wcss) > 2:

    # Normalize K and WCSS values for comparison
    x_normalized = np.array(list(k_values), dtype=float)

    y_normalized = np.array(wcss, dtype=float)

    x_normalized = (
        (x_normalized - x_normalized.min()) /
        (x_normalized.max() - x_normalized.min())
    )

    if y_normalized.max() != y_normalized.min():

        y_normalized = (
            (y_normalized - y_normalized.min()) /
            (y_normalized.max() - y_normalized.min())
        )

    # Calculate distance from the straight line
    # connecting the first and last points
    points = np.column_stack((x_normalized, y_normalized))

    line = points[-1] - points[0]

    difference = points[1:-1] - points[0]

    distances = np.abs(
        line[0] * difference[:, 1] -
        line[1] * difference[:, 0]
    ) / np.linalg.norm(line)

    suggested_k = list(k_values)[
        1 + np.argmax(distances)
    ]

else:
    suggested_k = 2

print("\nSuggested Optimal Number of Clusters:", suggested_k)

# Allow user to confirm or change K
choice = input(
    "Enter number of clusters K (Press Enter to use suggested K): "
)

if choice.strip() == "":
    optimal_k = suggested_k

else:
    optimal_k = int(choice)

# Validate cluster count
if optimal_k < 2 or optimal_k >= len(X):
    raise ValueError("K must be between 2 and number of records - 1.")

print("\nFinal Number of Clusters:", optimal_k)


# ==============================================================
# CELL 8: IMPLEMENT K-MEANS CLUSTERING
#
# Purpose:
# Divide the sales dataset into K clusters using
# the K-Means clustering algorithm.
#
# Working:
# 1. Select K initial centroids.
# 2. Assign each data point to its nearest centroid.
# 3. Calculate new centroids.
# 4. Repeat until centroids stabilize.
#
# ==============================================================

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

# Train the K-Means model and assign cluster labels
kmeans_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataset
df["KMEANS_CLUSTER"] = kmeans_labels

print("\nK-Means Clustering Results:")
print(df[["KMEANS_CLUSTER"]].head(10))

print("\nNumber of Records in Each K-Means Cluster:")
print(df["KMEANS_CLUSTER"].value_counts().sort_index())


# ==============================================================
# CELL 9: VISUALIZE K-MEANS CLUSTERS
#
# Purpose:
# Visualize the clusters using PCA.
#
# PCA = Principal Component Analysis.
#
# It reduces multiple numerical features into
# two dimensions so that clusters can be plotted.
#
# ==============================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    cmap="viridis",
    s=25
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("K-Means Clustering Visualization")

plt.colorbar(label="Cluster")

plt.show()


# ==============================================================
# CELL 10: IMPLEMENT HIERARCHICAL CLUSTERING
#
# Purpose:
# Implement Agglomerative Hierarchical Clustering.
#
# Working:
# 1. Initially, each data point is considered a separate cluster.
# 2. Find the two closest clusters.
# 3. Merge them into one cluster.
# 4. Repeat until the required number of clusters remains.
#
# Linkage Method:
# Ward's method minimizes the increase in
# within-cluster variance when merging clusters.
#
# ==============================================================

hierarchical = AgglomerativeClustering(
    n_clusters=optimal_k,
    linkage="ward"
)

# Perform hierarchical clustering
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# Add cluster labels to the dataset
df["HIERARCHICAL_CLUSTER"] = hierarchical_labels

print("\nHierarchical Clustering Results:")
print(df[["HIERARCHICAL_CLUSTER"]].head(10))

print("\nNumber of Records in Each Hierarchical Cluster:")
print(
    df["HIERARCHICAL_CLUSTER"].value_counts().sort_index()
)


# ==============================================================
# CELL 11: GENERATE DENDROGRAM
#
# Purpose:
# Display a Dendrogram to visualize how clusters
# are merged during hierarchical clustering.
#
# Dendrogram:
# A tree-like diagram showing the merging of clusters.
#
# ==============================================================

# Generate linkage matrix using Ward's method
linkage_matrix = linkage(
    X_scaled,
    method="ward"
)

plt.figure(figsize=(12, 6))

dendrogram(
    linkage_matrix,
    truncate_mode="lastp",
    p=30,
    show_leaf_counts=True
)

plt.title("Hierarchical Clustering Dendrogram")

plt.xlabel("Data Points / Clusters")

plt.ylabel("Distance")

plt.show()


# ==============================================================
# CELL 12: VISUALIZE HIERARCHICAL CLUSTERS
#
# Purpose:
# Display hierarchical clusters in two dimensions
# using the PCA-transformed dataset.
#
# ==============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=hierarchical_labels,
    cmap="viridis",
    s=25
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("Hierarchical Clustering Visualization")

plt.colorbar(label="Cluster")

plt.show()


# ==============================================================
# CELL 13: EVALUATE CLUSTERING MODELS
#
# Purpose:
# Evaluate clustering quality using Silhouette Score.
#
# Silhouette Score:
# Measures how similar a data point is to its
# own cluster compared with other clusters.
#
# Range:
# +1 = Excellent clustering.
#  0 = Clusters overlap.
# -1 = Incorrect clustering.
#
# Higher score generally indicates better clustering.
#
# ==============================================================

# Use a maximum sample of 1000 records to keep
# silhouette calculation reasonably efficient on large datasets.

sample_size = min(1000, len(X_scaled))

kmeans_score = silhouette_score(
    X_scaled,
    kmeans_labels,
    sample_size=sample_size,
    random_state=42
)

hierarchical_score = silhouette_score(
    X_scaled,
    hierarchical_labels,
    sample_size=sample_size,
    random_state=42
)

print("\nK-Means Silhouette Score:", round(kmeans_score, 4))

print(
    "Hierarchical Silhouette Score:",
    round(hierarchical_score, 4)
)


# ==============================================================
# CELL 14: COMPARE BOTH CLUSTERING METHODS
#
# Purpose:
# Compare K-Means and Hierarchical Clustering
# using their Silhouette Scores.
#
# Higher Silhouette Score indicates better separation
# and cohesion for the selected number of clusters.
#
# ==============================================================

comparison = pd.DataFrame({

    "Clustering Algorithm": [
        "K-Means",
        "Hierarchical"
    ],

    "Number of Clusters": [
        optimal_k,
        optimal_k
    ],

    "Silhouette Score": [
        kmeans_score,
        hierarchical_score
    ]

})

print("\nCLUSTERING MODEL COMPARISON:")

print(comparison.to_string(index=False))

# Identify better clustering result based on Silhouette Score
if kmeans_score > hierarchical_score:

    print("\nBetter Clustering Method: K-Means")

elif hierarchical_score > kmeans_score:

    print("\nBetter Clustering Method: Hierarchical Clustering")

else:

    print("\nBoth methods have equal Silhouette Scores.")


# ==============================================================
# CELL 15: DISPLAY CLUSTER-WISE SALES SUMMARY
#
# Purpose:
# Calculate average sales, quantity and price
# for each K-Means cluster.
#
# This helps understand the characteristics
# of different sales groups.
#
# ==============================================================

print("\nK-MEANS CLUSTER SUMMARY:")

summary = (
    df.groupby("KMEANS_CLUSTER")[features]
    .mean()
    .round(2)
)

print(summary)

print("\nHIERARCHICAL CLUSTER SUMMARY:")

hierarchical_summary = (
    df.groupby("HIERARCHICAL_CLUSTER")[features]
    .mean()
    .round(2)
)

print(hierarchical_summary)


# ==============================================================
# CELL 16: SAVE FINAL CLUSTERED DATASET
#
# Purpose:
# Save the original dataset with both clustering labels.
#
# Output File:
# sales_data_clustered.csv
#
# ==============================================================

df.to_csv("sales_data_clustered.csv", index=False)

print("\nClustered dataset saved successfully!")

print("Output File: sales_data_clustered.csv")


# ==============================================================
# IMPORTANT THEORY AND EXTERNAL VIVA QUESTIONS
# ==============================================================

# 1. What is Clustering?
# Clustering is an unsupervised Machine Learning technique
# that groups similar data points into clusters.

# 2. What is Unsupervised Learning?
# It is a Machine Learning technique in which the model
# learns patterns from data without predefined labels.

# 3. What is K-Means Clustering?
# K-Means is an unsupervised learning algorithm that
# divides data into K clusters based on similarity.

# 4. Why is it called K-Means?
# K represents the number of clusters.
# Means represents the average position (centroid)
# of all points in a cluster.

# 5. What are the steps in K-Means?
# Step 1: Choose the number of clusters K.
# Step 2: Initialize K centroids.
# Step 3: Assign points to the nearest centroid.
# Step 4: Calculate new centroids.
# Step 5: Repeat until assignments or centroids stabilize.

# 6. What is a Centroid?
# A centroid is the mean position of all data points
# belonging to a particular cluster.

# 7. What is the Elbow Method?
# It is a technique used to estimate the optimal
# number of clusters in K-Means.
# It plots K against WCSS and identifies the elbow point.

# 8. What is WCSS?
# Within Cluster Sum of Squares.
# It calculates the sum of squared distances
# between data points and their respective centroids.

# 9. What is Inertia in K-Means?
# Inertia is the sum of squared distances of points
# from their nearest cluster centers.
# It is the WCSS value used by Scikit-Learn.

# 10. Why does WCSS decrease when K increases?
# Because more clusters allow data points to be
# assigned closer to their respective centroids.
# WCSS becomes zero when every data point has
# its own individual cluster.

# 11. What is the optimal K?
# The K value at which the WCSS curve begins
# to flatten significantly is considered the elbow.
# It is an estimate, not always a unique answer.

# 12. What is Hierarchical Clustering?
# It is an unsupervised learning algorithm that creates
# a hierarchy of clusters by repeatedly merging
# or splitting groups.

# 13. What are the two types of Hierarchical Clustering?
#
# Agglomerative:
# Bottom-up approach.
# Each point starts as a separate cluster.
# Clusters are repeatedly merged.
#
# Divisive:
# Top-down approach.
# All points start in one cluster and are split.

# 14. Which type is implemented in this practical?
# Agglomerative Hierarchical Clustering.

# 15. What is a Dendrogram?
# A tree-like diagram that displays the sequence
# of cluster merging in hierarchical clustering.

# 16. What is Linkage?
# Linkage defines how the distance between two clusters
# is calculated.

# 17. What are the common linkage methods?
# Single Linkage: Minimum distance between clusters.
# Complete Linkage: Maximum distance between clusters.
# Average Linkage: Average distance between clusters.
# Ward Linkage: Minimizes increase in within-cluster variance.

# 18. What is Ward's Method?
# It merges clusters in a way that minimizes the increase
# in within-cluster sum of squares.

# 19. What is Standardization?
# It transforms numerical features so that they have
# approximately zero mean and unit standard deviation.

# 20. Why is StandardScaler used?
# Because clustering algorithms based on distance
# can be affected by features with larger numerical ranges.

# 21. What is Euclidean Distance?
# It measures the straight-line distance between two points.
# K-Means commonly uses Euclidean distance.

# 22. What is PCA?
# Principal Component Analysis.
# It reduces the number of dimensions while preserving
# as much variance as possible.
# Here, PCA is used for two-dimensional visualization.

# 23. What is Silhouette Score?
# It measures how well-separated the clusters are.
# Range is -1 to +1.
# Higher values generally indicate better clustering.

# 24. What is the difference between K-Means
# and Hierarchical Clustering?
#
# K-Means:
# - Requires K in advance.
# - Uses centroids.
# - Reassigns points during iterations.
# - Generally faster on large datasets.
#
# Hierarchical:
# - Builds a hierarchy of clusters.
# - Uses a dendrogram.
# - Does not need K to construct its hierarchy.
# - Agglomerative clustering is computationally expensive
#   for large datasets.

# 25. Is K-Means supervised or unsupervised?
# Unsupervised Learning.

# 26. Is Hierarchical Clustering supervised or unsupervised?
# Unsupervised Learning.

# 27. What are the advantages of K-Means?
# - Simple to understand and implement.
# - Efficient for many numerical datasets.
# - Works well with compact, separated clusters.

# 28. What are the disadvantages of K-Means?
# - K must be selected beforehand.
# - Sensitive to outliers.
# - Sensitive to initial centroid positions.
# - Can perform poorly on irregularly shaped clusters.

# 29. What are the advantages of Hierarchical Clustering?
# - Produces a hierarchy of clusters.
# - Dendrogram helps understand cluster relationships.
# - Does not require K to build the hierarchy.

# 30. What are the disadvantages of Hierarchical Clustering?
# - High memory and computational requirements.
# - Sensitive to distance and linkage choices.
# - Early merging decisions cannot be undone.

# 31. What is the role of random_state=42?
# It ensures reproducible random initialization
# in K-Means.

# 32. What is n_init=10?
# K-Means is run 10 times using different initial
# centroid configurations, and the best solution
# based on inertia is selected.

# 33. What is fit_predict()?
# It trains the clustering model and returns
# the assigned cluster label for each data point.

# 34. What is a Cluster Label?
# A numerical identifier assigned to every data point
# based on the cluster to which it belongs.
# Labels 0, 1, 2, etc. do not indicate ranking.

# 35. Why do we use sales_data_sample.csv?
# It contains sales-related information such as
# quantity ordered, price, sales and product details.
# Clustering helps identify groups with similar
# sales characteristics.

# 36. What is the purpose of clustering sales data?
# To identify similar sales records and understand
# different sales patterns or customer/order groups.

# 37. Why are identifiers excluded from clustering?
# Columns such as ORDERNUMBER are identifiers,
# not meaningful numerical measurements of similarity.

# 38. Why is the Elbow Method not always exact?
# Sometimes the WCSS graph does not show a clear elbow.
# In such cases, Silhouette Score and domain knowledge
# can help select a suitable cluster count.

# 39. Does clustering predict a target variable?
# No. Clustering groups data based on similarity.
# It does not predict a predefined target like regression.

# 40. What is the difference between Classification
# and Clustering?
#
# Classification:
# Supervised learning that predicts known categories.
#
# Clustering:
# Unsupervised learning that discovers groups
# without predefined class labels.

# 41. What is the purpose of correlation in sales data?
# It helps identify relationships between numerical
# features, although correlation is not required
# for the K-Means algorithm itself.

# 42. Important practical assumptions:
# - Dataset should contain numerical features.
# - Missing values must be handled.
# - Features should be standardized.
# - K-Means requires a selected K.
# - Hierarchical clustering requires a distance
#   and linkage strategy.
# - Clusters are based on similarity, not actual labels.

# ==============================================================
# COMPLEXITY ANALYSIS
# ==============================================================

# Let:
# n = Number of data records.
# k = Number of clusters.
# d = Number of features.
# i = Number of K-Means iterations.

# 1. K-MEANS CLUSTERING
#
# Time Complexity:
# O(n * k * i * d)
#
# Space Complexity:
# O(n * d + k * d)
#
# The algorithm calculates distances between
# data points and centroids repeatedly.

# 2. HIERARCHICAL CLUSTERING
#
# Time Complexity:
# Typically O(n^2 log n) for common optimized
# agglomerative linkage implementations.
# Exact performance depends on linkage and implementation.
#
# Space Complexity:
# O(n^2) for common full hierarchical implementations.
#
# Ward linkage can be computationally expensive
# for large datasets.

# 3. ELBOW METHOD
#
# Time Complexity:
# Approximately O(K_max * n * k * i * d)
# across all tested K values, where k varies per run.
#
# Space Complexity:
# Depends on the stored data and model.

# 4. STANDARDIZATION
#
# Time Complexity:
# O(n * d)
#
# Space Complexity:
# O(n * d) for the transformed dataset.

# 5. PCA
#
# Time Complexity:
# Depends on the number of records and features.
# Full SVD is approximately O(min(n*d^2, n^2*d)).
#
# Space Complexity:
# O(n*d) for storing transformed data and related arrays.

# 6. SILHOUETTE SCORE
#
# Exact calculation can require O(n^2 * d) time.
# This practical uses at most 1000 sampled records
# for evaluation to reduce computation.

# ==============================================================
# FINAL PRACTICAL SUMMARY
# ==============================================================

# Experiment:
# Sales Data Clustering

# Dataset:
# sales_data_sample.csv

# Learning Type:
# Unsupervised Machine Learning

# Algorithms:
# 1. K-Means Clustering
# 2. Agglomerative Hierarchical Clustering

# Preprocessing:
# Duplicate Removal, Feature Selection,
# Missing Value Handling and Standardization.

# Optimal Cluster Selection:
# Elbow Method

# Hierarchical Visualization:
# Dendrogram

# Cluster Visualization:
# PCA 2D Scatter Plot

# Evaluation:
# Silhouette Score

# Output:
# sales_data_clustered.csv

# IMPORTANT EXTERNAL VIVA POINT:
# K-Means uses centroids and minimizes within-cluster
# squared distances, whereas Hierarchical Clustering
# builds a tree-like structure using cluster merging.
#
# The Elbow Method is used to estimate K for K-Means.
# A dendrogram is used to inspect hierarchical merging.
#
# ==============================================================
# END OF PRACTICAL
# ==============================================================
