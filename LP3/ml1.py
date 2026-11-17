# ==============================================================
# MACHINE LEARNING PRACTICAL
# Experiment: Uber Ride Fare Price Prediction
"""
Question 
Predict the price of the Uber ride from a given pickup point to the 
agreed drop-off location. 
Perform the following tasks: 
1. Pre-process the dataset. 
2. Identify outliers. 
3. Check the correlation. 
4. Implement linear regression and random forest regression models. 
5. Evaluate the models and compare their respective scores like R2, 
RMSE, etc. 
"""
# Objective:
# Predict Uber ride fare prices based on pickup location,
# drop-off location, distance, passenger count and date/time.
#
# Algorithms:
# 1. Linear Regression
# 2. Random Forest Regression
#
# Tasks:
# 1. Dataset preprocessing
# 2. Outlier identification and removal
# 3. Correlation analysis
# 4. Model implementation
# 5. Model evaluation and comparison
#
# HOW TO RUN:
# Run this Python file CELL BY CELL in Jupyter Notebook
# or Google Colab.
#
# Dataset:
# uber.csv
#
# Expected columns:
# fare_amount, pickup_longitude, pickup_latitude,
# dropoff_longitude, dropoff_latitude,
# passenger_count, pickup_datetime
# ==============================================================


# ==============================================================
# CELL 1: IMPORT REQUIRED LIBRARIES
#
# Purpose:
# Import libraries required for data preprocessing,
# visualization, model building and model evaluation.
#
# ==============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


# ==============================================================
# CELL 2: LOAD THE DATASET
#
# Purpose:
# Load the Uber dataset using Pandas.
# Display the first five records and dataset information.
#
# Before running, upload uber.csv into the same folder
# as this Python file.
#
# ==============================================================

df = pd.read_csv("uber.csv")

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
# Clean the dataset by:
# 1. Removing unnecessary columns.
# 2. Removing duplicate records.
# 3. Handling missing values.
# 4. Converting pickup_datetime into datetime format.
# 5. Extracting useful date and time features.
#
# ==============================================================

# Remove unnecessary index column if present
df.drop(columns=["Unnamed: 0", "key"],
        errors="ignore", inplace=True)

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Convert pickup_datetime into datetime format
df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"], errors="coerce", utc=True
)

# Remove rows containing missing values
df.dropna(inplace=True)

# Extract useful date and time features
df["hour"] = df["pickup_datetime"].dt.hour
df["day"] = df["pickup_datetime"].dt.day
df["month"] = df["pickup_datetime"].dt.month
df["year"] = df["pickup_datetime"].dt.year

# Remove original datetime column
df.drop(columns=["pickup_datetime"], inplace=True)

print("\nDataset After Preprocessing:")
print(df.head())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# ==============================================================
# CELL 4: REMOVE INVALID RECORDS
#
# Purpose:
# Remove records containing invalid fare amounts,
# passenger counts and geographic coordinates.
#
# Longitude range: -180 to 180
# Latitude range: -90 to 90
#
# ==============================================================

df = df[
    (df["fare_amount"] > 0) &
    (df["passenger_count"] > 0) &
    (df["pickup_longitude"].between(-180, 180)) &
    (df["dropoff_longitude"].between(-180, 180)) &
    (df["pickup_latitude"].between(-90, 90)) &
    (df["dropoff_latitude"].between(-90, 90))
].copy()

print("\nDataset After Removing Invalid Records:")
print(df.shape)


# ==============================================================
# CELL 5: IDENTIFY AND REMOVE OUTLIERS
#
# Purpose:
# Identify outliers using the IQR (Interquartile Range) method.
#
# Q1 = First Quartile (25%)
# Q3 = Third Quartile (75%)
# IQR = Q3 - Q1
#
# Lower Limit = Q1 - 1.5 * IQR
# Upper Limit = Q3 + 1.5 * IQR
#
# Values outside these limits are identified as outliers.
#
# ==============================================================

# Check outliers in fare amount
Q1 = df["fare_amount"].quantile(0.25)
Q3 = df["fare_amount"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print("\nOutlier Limits:")
print("Lower Limit:", lower)
print("Upper Limit:", upper)

# Count outliers before removal
outliers = df[
    (df["fare_amount"] < lower) |
    (df["fare_amount"] > upper)
]

print("\nNumber of Outliers:", len(outliers))

# Remove fare amount outliers
df = df[
    (df["fare_amount"] >= lower) &
    (df["fare_amount"] <= upper)
].copy()

print("\nDataset Shape After Outlier Removal:")
print(df.shape)


# ==============================================================
# CELL 6: VISUALIZE OUTLIERS USING BOXPLOT
#
# Purpose:
# Display a boxplot to visualize the distribution
# of fare amounts and identify potential outliers.
#
# ==============================================================

plt.figure(figsize=(8, 4))

sns.boxplot(x=df["fare_amount"])

plt.title("Uber Fare Amount - Outlier Detection")
plt.xlabel("Fare Amount")

plt.show()


# ==============================================================
# CELL 7: CALCULATE DISTANCE
#
# Purpose:
# Calculate the geographical distance between pickup
# and drop-off locations using the Haversine Formula.
#
# Haversine Formula calculates distance between
# two geographical coordinates on Earth.
#
# ==============================================================

def calculate_distance(lat1, lon1, lat2, lon2):

    # Convert degrees into radians
    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2 +
        np.cos(lat1) * np.cos(lat2) *
        np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    # Earth's radius in kilometers
    return 6371 * c


# Calculate ride distance
df["distance"] = calculate_distance(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"]
)

# Remove zero-distance and unrealistic long-distance rides
df = df[
    (df["distance"] > 0) &
    (df["distance"] < 200)
].copy()

print("\nDataset With Distance:")
print(df[["distance", "fare_amount"]].head())


# ==============================================================
# CELL 8: CORRELATION ANALYSIS
#
# Purpose:
# Calculate correlation between numerical variables.
# Visualize the relationship using a heatmap.
#
# Correlation:
# +1 = Perfect Positive Correlation
# -1 = Perfect Negative Correlation
#  0 = No Linear Correlation
#
# ==============================================================

plt.figure(figsize=(10, 7))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()

print("\nCorrelation With Fare Amount:")
print(correlation["fare_amount"].sort_values(ascending=False))


# ==============================================================
# CELL 9: SELECT FEATURES AND TARGET
#
# Purpose:
# Select input features (X) and target variable (Y).
#
# Features:
# Pickup coordinates, drop-off coordinates,
# passenger count, time features and distance.
#
# Target:
# fare_amount
#
# ==============================================================

X = df[
    [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "hour",
        "day",
        "month",
        "year",
        "distance"
    ]
]

y = df["fare_amount"]

print("\nInput Features:")
print(X.head())

print("\nTarget Variable:")
print(y.head())


# ==============================================================
# CELL 10: SPLIT THE DATASET
#
# Purpose:
# Divide dataset into training and testing sets.
#
# 80% = Training Data
# 20% = Testing Data
#
# Training data is used to train the model.
# Testing data is used to evaluate model performance.
#
# ==============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ==============================================================
# CELL 11: IMPLEMENT LINEAR REGRESSION
#
# Purpose:
# Train a Linear Regression model to predict Uber fare prices.
#
# Linear Regression finds a linear relationship
# between input features and the target variable.
#
# ==============================================================

lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Predict fare prices
lr_predictions = lr_model.predict(X_test)

print("\nLinear Regression Model Trained Successfully!")


# ==============================================================
# CELL 12: IMPLEMENT RANDOM FOREST REGRESSION
#
# Purpose:
# Train a Random Forest Regression model.
#
# Random Forest combines multiple Decision Trees
# and averages their predictions.
#
# ==============================================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_model.fit(X_train, y_train)

# Predict fare prices
rf_predictions = rf_model.predict(X_test)

print("\nRandom Forest Model Trained Successfully!")


# ==============================================================
# CELL 13: EVALUATE BOTH MODELS
#
# Purpose:
# Compare the performance of both regression models.
#
# R2 Score:
# Measures how well the model explains variation in fares.
# Closer to 1 is generally better.
#
# RMSE:
# Root Mean Squared Error.
# Lower value indicates smaller prediction errors.
#
# MAE:
# Mean Absolute Error.
# Lower value indicates better predictions.
#
# ==============================================================

def evaluate_model(name, actual, predicted):

    r2 = r2_score(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mae = mean_absolute_error(actual, predicted)

    print("\n", name)
    print("R2 Score:", round(r2, 4))
    print("RMSE:", round(rmse, 4))
    print("MAE:", round(mae, 4))


# Evaluate Linear Regression
evaluate_model(
    "Linear Regression",
    y_test,
    lr_predictions
)

# Evaluate Random Forest
evaluate_model(
    "Random Forest Regression",
    y_test,
    rf_predictions
)


# ==============================================================
# CELL 14: COMPARE MODEL PERFORMANCE
#
# Purpose:
# Create a comparison table to identify which model
# performs better based on R2, RMSE and MAE.
#
# Higher R2 is better.
# Lower RMSE and MAE are better.
#
# ==============================================================

results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest"
    ],

    "R2 Score": [
        r2_score(y_test, lr_predictions),
        r2_score(y_test, rf_predictions)
    ],

    "RMSE": [
        np.sqrt(mean_squared_error(y_test, lr_predictions)),
        np.sqrt(mean_squared_error(y_test, rf_predictions))
    ],

    "MAE": [
        mean_absolute_error(y_test, lr_predictions),
        mean_absolute_error(y_test, rf_predictions)
    ]

})

print("\nMODEL COMPARISON:")
print(results.to_string(index=False))

# Select the model with the highest R2 score
best_model = results.loc[
    results["R2 Score"].idxmax(), "Model"
]

print("\nBest Performing Model:", best_model)


# ==============================================================
# CELL 15: VISUALIZE ACTUAL VS PREDICTED FARES
#
# Purpose:
# Compare actual fare prices with predicted fare prices
# using the Random Forest model.
#
# Points closer to the diagonal line indicate
# more accurate predictions.
#
# ==============================================================

plt.figure(figsize=(7, 5))

plt.scatter(
    y_test,
    rf_predictions,
    alpha=0.4
)

plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")

plt.title("Actual vs Predicted Uber Fare")

plt.show()


# ==============================================================
# CELL 16: PREDICT A NEW UBER RIDE FARE
#
# Purpose:
# Accept new ride details from the user and predict
# its fare using the best performing trained model.
#
# Coordinates should be entered in decimal degrees.
# ==============================================================

print("\nEnter New Ride Details:")

pickup_lon = float(input("Pickup Longitude: "))
pickup_lat = float(input("Pickup Latitude: "))

dropoff_lon = float(input("Drop-off Longitude: "))
dropoff_lat = float(input("Drop-off Latitude: "))

passengers = int(input("Number of Passengers: "))

hour = int(input("Pickup Hour (0-23): "))
day = int(input("Pickup Day (1-31): "))
month = int(input("Pickup Month (1-12): "))
year = int(input("Pickup Year: "))

# Calculate distance for the new ride
distance = calculate_distance(
    pickup_lat,
    pickup_lon,
    dropoff_lat,
    dropoff_lon
)

# Prepare new ride data in the same feature order
new_ride = pd.DataFrame([[
    pickup_lon,
    pickup_lat,
    dropoff_lon,
    dropoff_lat,
    passengers,
    hour,
    day,
    month,
    year,
    distance
]], columns=X.columns)

# Use the best performing model
if best_model == "Random Forest":
    predicted_fare = rf_model.predict(new_ride)[0]
else:
    predicted_fare = lr_model.predict(new_ride)[0]

print("\nPredicted Uber Ride Fare:", round(predicted_fare, 2))


# ==============================================================
# IMPORTANT THEORY AND EXTERNAL VIVA QUESTIONS
# ==============================================================

# 1. What is Machine Learning?
# Machine Learning enables computers to learn patterns
# from data and make predictions without explicitly
# programming every rule.

# 2. What is Regression?
# Regression is a supervised learning technique
# used to predict continuous numerical values.
# Here, the continuous value is Uber ride fare.

# 3. What is Linear Regression?
# Linear Regression models the relationship between
# input variables and a continuous target variable
# using a linear equation.

# 4. What is Random Forest Regression?
# It is an ensemble learning algorithm that combines
# multiple Decision Trees to predict numerical values.
# It averages the predictions of individual trees.

# 5. What is Supervised Learning?
# Learning using labelled data.
# Here, ride details are inputs and fare amount
# is the known output.

# 6. What is Dataset Preprocessing?
# It is the process of cleaning and transforming data
# before supplying it to a Machine Learning model.

# 7. What is Pandas?
# A Python library used for data manipulation and analysis.

# 8. What is NumPy?
# A Python library used for numerical computations.

# 9. What is Matplotlib?
# A Python library used for data visualization.

# 10. What is Seaborn?
# A Python visualization library built on Matplotlib.
# It is useful for statistical plots and heatmaps.

# 11. What are Outliers?
# Outliers are observations that differ significantly
# from the majority of the dataset.

# 12. What is IQR?
# Interquartile Range measures the spread of the middle
# 50% of data.
# IQR = Q3 - Q1.

# 13. What is Correlation?
# Correlation measures the strength and direction
# of a linear relationship between two numerical variables.

# 14. What is a Correlation Heatmap?
# A graphical representation of correlation values
# between numerical variables.

# 15. What is Feature Selection?
# Selecting relevant input variables for model training.

# 16. What is a Target Variable?
# The variable that the model tries to predict.
# Here, fare_amount is the target.

# 17. What is Train-Test Split?
# Dividing data into training and testing sets
# to evaluate how well a model performs on unseen data.

# 18. What is the purpose of random_state=42?
# It makes the random splitting and model initialization
# reproducible.

# 19. What is R2 Score?
# It measures the proportion of target variance
# explained by the model.
# Best possible value = 1.
# It can also be negative for a poor model.

# 20. What is RMSE?
# Root Mean Squared Error measures the square root
# of average squared prediction errors.
# Lower RMSE indicates better predictions.

# 21. What is MAE?
# Mean Absolute Error measures the average absolute
# difference between actual and predicted values.
# Lower MAE indicates better predictions.

# 22. Difference between Linear Regression and Random Forest:
#
# Linear Regression:
# - Fits a linear relationship.
# - Simple and easy to interpret.
# - May not capture complex patterns.
#
# Random Forest:
# - Uses multiple Decision Trees.
# - Can capture nonlinear patterns.
# - Usually requires more computation.
# - Less directly interpretable.

# 23. Which model is better?
# The model with higher R2 and lower RMSE/MAE
# generally performs better on the same test dataset.
# Actual results depend on dataset quality.

# 24. What is Haversine Formula?
# It calculates the great-circle distance between
# two points on Earth's surface using latitude
# and longitude coordinates.

# 25. What is Overfitting?
# When a model learns training data too closely
# and performs poorly on unseen data.

# 26. What is Underfitting?
# When a model fails to learn important patterns
# from the training data.

# 27. What is an Ensemble Learning Algorithm?
# A technique that combines multiple models
# to improve prediction performance.
# Random Forest is an ensemble algorithm.

# 28. What is n_estimators=100?
# It means the Random Forest contains 100 Decision Trees.

# 29. Why is distance important in Uber fare prediction?
# Ride distance is an important factor influencing
# the fare amount.

# 30. What is the Haversine distance measured in?
# The formula in this program returns distance in kilometers.

# 31. What is the purpose of dropna()?
# It removes rows containing missing values.

# 32. Why remove duplicate records?
# Duplicate records can cause repeated observations
# and may distort model training.

# 33. What is Data Leakage?
# It occurs when information from the test dataset
# or future information improperly influences training.
# Preprocessing rules should ideally be learned from
# training data only.

# 34. What is the limitation of this practical?
# - Fare prices can depend on traffic and demand.
# - Weather and surge pricing are not included.
# - Geographic coordinates may contain errors.
# - Historical data may not represent current prices.
# - The model predicts learned historical fares,
#   not guaranteed live Uber prices.

# 35. What is the difference between Classification
# and Regression?
#
# Classification predicts categories.
# Regression predicts continuous numerical values.

# 36. Why do we compare multiple models?
# To identify which model provides better predictions
# on unseen testing data.

# ==============================================================
# COMPLEXITY AND IMPORTANT POINTS
# ==============================================================

# Algorithm 1: Linear Regression
# Training complexity: Approximately O(n * p^2 + p^3)
# for the standard least-squares solver.
# Prediction complexity: O(p) per record.
#
# Algorithm 2: Random Forest Regression
# Training complexity: Depends on the number of trees,
# samples, features and tree depth.
# Approximate training complexity:
# O(T * n * p * log(n)) for typical balanced trees.
#
# n = Number of training records.
# p = Number of input features.
# T = Number of trees.
#
# Dataset preprocessing:
# Depends on the number of records and features.
#
# IQR outlier detection:
# Quantile computation depends on sorting/interpolation.
#
# Correlation:
# Approximately O(n * p^2).
#
# IMPORTANT:
# These are approximate algorithmic complexities.
# Actual runtime depends on implementation and data.

# ==============================================================
# FINAL PRACTICAL SUMMARY
# ==============================================================

# Experiment Name:
# Uber Ride Fare Price Prediction

# Type:
# Supervised Machine Learning - Regression

# Dataset:
# Uber Ride Dataset (uber.csv)

# Target Variable:
# fare_amount

# Preprocessing:
# Missing Values, Duplicates, Invalid Values,
# Date-Time Extraction and Outlier Removal.

# Outlier Detection:
# IQR Method

# Distance Calculation:
# Haversine Formula

# Correlation:
# Correlation Matrix and Heatmap

# Model 1:
# Linear Regression

# Model 2:
# Random Forest Regression

# Evaluation Metrics:
# R2 Score, RMSE and MAE

# Visualization:
# Boxplot, Correlation Heatmap,
# Actual vs Predicted Scatterplot

# Final Output:
# Model Comparison and New Uber Ride Fare Prediction

# ==============================================================
# END OF PRACTICAL
# ==============================================================