
# ==============================================================
# MACHINE LEARNING PRACTICAL
# Experiment: Diabetes Prediction Using K-Nearest Neighbors
"""
Question : 
Implement the K-Nearest Neighbors algorithm on the diabetes.csv 
dataset. Compute confusion matrix, accuracy, error rate, precision 
and recall on the given dataset.
"""
# Objective:
# Implement K-Nearest Neighbors (KNN) classification
# on diabetes.csv dataset and calculate:
#
# 1. Confusion Matrix
# 2. Accuracy
# 3. Error Rate
# 4. Precision
# 5. Recall
#
# Algorithm:
# K-Nearest Neighbors (KNN)
#
# Type:
# Supervised Machine Learning - Classification
#
# HOW TO RUN:
# Run this Python file CELL BY CELL in Jupyter Notebook
# or Google Colab.
#
# In VS Code, use Shift + Enter to execute each cell.
#
# Dataset:
# diabetes.csv
#
# INSTALL REQUIRED LIBRARIES:
# !pip install pandas numpy matplotlib seaborn scikit-learn
# ==============================================================


# ==============================================================
# CELL 1: IMPORT REQUIRED LIBRARIES
#
# Purpose:
# Import libraries required for data preprocessing,
# visualization, classification and model evaluation.
#
# ==============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ==============================================================
# CELL 2: LOAD THE DATASET
#
# Purpose:
# Load diabetes.csv using Pandas.
# Display the first five records, dataset shape
# and information about the columns.
#
# Expected columns in the common Pima diabetes dataset:
# Pregnancies, Glucose, BloodPressure, SkinThickness,
# Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome.
#
# Outcome:
# 0 = Non-diabetic
# 1 = Diabetic
#
# ==============================================================

df = pd.read_csv("diabetes.csv")

# Remove unnecessary spaces from column names
df.columns = df.columns.str.strip()

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
# Clean the dataset before applying KNN.
#
# 1. Remove duplicate records.
# 2. Identify the target column.
# 3. Separate input features and target.
# 4. Handle invalid zero values.
# 5. Handle missing values.
#
# Important:
# In the common Pima diabetes dataset, zero values in
# Glucose, BloodPressure, SkinThickness, Insulin and BMI
# generally represent missing measurements.
#
# These zero values are replaced with NaN and later
# imputed using the training dataset median.
#
# Zero in Pregnancies is valid and is NOT replaced.
#
# ==============================================================

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Automatically identify the target column
target_candidates = [
    "outcome",
    "diabetes",
    "diabetes_binary",
    "target",
    "class"
]

target_column = None

for col in df.columns:

    if col.lower() in target_candidates:
        target_column = col
        break

# If target column is not identified automatically,
# ask the user to enter its name.
if target_column is None:

    print("\nAvailable Columns:", list(df.columns))

    target_column = input("Enter target column name: ").strip()

    if target_column not in df.columns:
        raise ValueError("Target column not found in dataset.")

print("\nTarget Column:", target_column)

# Remove rows having missing target values
df.dropna(subset=[target_column], inplace=True)

# Separate target variable
y = df[target_column]

# Select numerical input features
X = df.drop(columns=[target_column])

X = X.select_dtypes(include=np.number)

# Remove unnecessary ID columns if available
id_columns = [
    "id",
    "patientid",
    "patient_id"
]

X = X.drop(
    columns=[
        col for col in X.columns
        if col.lower() in id_columns
    ],
    errors="ignore"
)

# Convert invalid values into NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Handle zero values in common Pima diabetes columns
missing_zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in X.columns:

    if col.lower() in [c.lower() for c in missing_zero_columns]:
        X.loc[X[col] == 0, col] = np.nan

# Remove columns containing only missing values
X.dropna(axis=1, how="all", inplace=True)

# Remove constant features
X = X.loc[:, X.nunique(dropna=True) > 1]

if X.shape[1] == 0:
    raise ValueError("No useful numerical features found.")

print("\nSelected Features:")
print(list(X.columns))

print("\nFeatures After Preprocessing:")
print(X.head())

print("\nTarget Distribution:")
print(y.value_counts())


# ==============================================================
# CELL 4: ENCODE THE TARGET VARIABLE
#
# Purpose:
# Convert target labels into numerical values.
#
# Label Encoding:
# Converts categorical class labels into numbers.
#
# For the common diabetes dataset:
# 0 = Non-diabetic
# 1 = Diabetic
#
# The positive class (1) is used for calculating
# precision, recall and other binary classification metrics.
#
# ==============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

print("\nTarget Class Mapping:")

for i, label in enumerate(label_encoder.classes_):
    print(i, "=", label)

# This practical expects binary diabetes classification
if len(label_encoder.classes_) != 2:
    raise ValueError(
        "This practical expects exactly two target classes."
    )

print("\nTarget Encoding Completed Successfully!")


# ==============================================================
# CELL 5: SPLIT DATASET INTO TRAINING AND TESTING DATA
#
# Purpose:
# Divide the dataset into training and testing sets.
#
# Training Data = 80%
# Testing Data = 20%
#
# Training data is used to build the model.
# Testing data is used to evaluate predictions.
#
# stratify=y maintains the class distribution in both sets.
#
# ==============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nTraining Class Distribution:")
print(pd.Series(y_train).value_counts())


# ==============================================================
# CELL 6: FEATURE SCALING AND MISSING VALUE HANDLING
#
# Purpose:
# Prepare features for the KNN algorithm.
#
# KNN uses distance calculations.
# Features with larger numerical ranges can dominate
# the distance calculation.
#
# StandardScaler:
# Converts features to approximately:
# Mean = 0
# Standard Deviation = 1
#
# SimpleImputer:
# Replaces missing values using the median.
#
# IMPORTANT:
# The Pipeline fits the imputer and scaler only on
# training data to prevent data leakage.
#
# ==============================================================

# Create preprocessing and KNN pipeline
# The KNN model will be added in the next cell.

preprocessor = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

# Fit preprocessing on training data only
X_train_scaled = preprocessor.fit_transform(X_train)

# Transform testing data using the same training parameters
X_test_scaled = preprocessor.transform(X_test)

print("\nFeature Scaling Completed Successfully!")

print("\nFirst Five Scaled Training Records:")
print(X_train_scaled[:5])


# ==============================================================
# CELL 7: IMPLEMENT K-NEAREST NEIGHBORS (KNN)
#
# Purpose:
# Implement the KNN classification algorithm.
#
# Working:
# 1. Select the value of K.
# 2. Calculate distances between the test point
#    and training data points.
# 3. Identify the K nearest neighbors.
# 4. Find the majority class among those neighbors.
# 5. Assign the majority class as the prediction.
#
# Default K = 5
#
# ==============================================================

# Take K value from the user
k_input = input("\nEnter value of K (Press Enter for 5): ")

if k_input.strip() == "":
    k = 5
else:
    k = int(k_input)

# K cannot exceed the number of training records
if k < 1 or k > len(X_train):
    raise ValueError("K must be between 1 and number of training records.")

# Create KNN classifier
knn = KNeighborsClassifier(
    n_neighbors=k,
    metric="euclidean",
    weights="uniform"
)

# Train the KNN model
knn.fit(X_train_scaled, y_train)

print("\nKNN Model Trained Successfully!")

# Predict testing data
y_pred = knn.predict(X_test_scaled)

print("\nFirst 10 Actual Values:")
print(y_test[:10])

print("\nFirst 10 Predicted Values:")
print(y_pred[:10])


# ==============================================================
# CELL 8: CALCULATE CONFUSION MATRIX
#
# Purpose:
# Compare actual values with predicted values.
#
# Confusion Matrix:
#
#                 Predicted
#                 0       1
# Actual  0      TN      FP
#         1      FN      TP
#
# TN = True Negative
# FP = False Positive
# FN = False Negative
# TP = True Positive
#
# ==============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)

# Extract individual values
tn, fp, fn, tp = cm.ravel()

print("\nTrue Negative (TN):", tn)
print("False Positive (FP):", fp)
print("False Negative (FN):", fn)
print("True Positive (TP):", tp)


# ==============================================================
# CELL 9: CALCULATE ACCURACY AND ERROR RATE
#
# Purpose:
# Calculate the overall correctness of the model.
#
# Accuracy:
# Number of correct predictions / Total predictions.
#
# Error Rate:
# Number of incorrect predictions / Total predictions.
#
# ==============================================================

accuracy = accuracy_score(y_test, y_pred)

error_rate = 1 - accuracy

print("\nAccuracy:", round(accuracy, 4))

print("Accuracy Percentage:", round(accuracy * 100, 2), "%")

print("Error Rate:", round(error_rate, 4))

print("Error Percentage:", round(error_rate * 100, 2), "%")


# ==============================================================
# CELL 10: CALCULATE PRECISION AND RECALL
#
# Purpose:
# Calculate Precision and Recall for the positive class.
#
# Precision:
# Out of all cases predicted as diabetic,
# how many are actually diabetic?
#
# Recall:
# Out of all actual diabetic cases,
# how many are correctly identified?
#
# Positive Class = 1 (Diabetic)
#
# ==============================================================

precision = precision_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

print("\nPrecision:", round(precision, 4))

print("Recall:", round(recall, 4))

print("F1 Score:", round(f1, 4))


# ==============================================================
# CELL 11: VISUALIZE CONFUSION MATRIX
#
# Purpose:
# Display the confusion matrix using a heatmap.
#
# This visualization helps identify:
# Correct predictions and incorrect predictions.
#
# ==============================================================

class_names = [
    str(label) for label in label_encoder.classes_
]

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Values")

plt.ylabel("Actual Values")

plt.title("KNN Confusion Matrix")

plt.show()


# ==============================================================
# CELL 12: DISPLAY CLASSIFICATION REPORT
#
# Purpose:
# Display precision, recall, F1-score and support
# for both target classes.
#
# Support:
# Number of actual records belonging to each class.
#
# ==============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=class_names,
        zero_division=0
    )
)


# ==============================================================
# CELL 13: DISPLAY FINAL PERFORMANCE SUMMARY
#
# Purpose:
# Display all important evaluation metrics together
# in a structured table.
#
# ==============================================================

results = pd.DataFrame({

    "Evaluation Metric": [
        "Accuracy",
        "Error Rate",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Score": [
        accuracy,
        error_rate,
        precision,
        recall,
        f1
    ]

})

print("\nFINAL KNN MODEL PERFORMANCE:")

print(results.to_string(index=False))

print("\nK Value Used:", k)

print("Algorithm: K-Nearest Neighbors")

print("Model Evaluation Completed Successfully!")


# ==============================================================
# IMPORTANT THEORY AND EXTERNAL VIVA QUESTIONS
# ==============================================================

# 1. What is KNN?
# K-Nearest Neighbors is a supervised Machine Learning
# algorithm used for classification and regression.
# It predicts the class of a new data point based on
# the majority class of its K nearest neighbors.

# 2. Is KNN supervised or unsupervised?
# Supervised Learning because it uses labelled training data.

# 3. What is the full form of KNN?
# K-Nearest Neighbors.

# 4. Why is it called K-Nearest Neighbors?
# K represents the number of nearest data points considered
# to determine the class of a new data point.

# 5. What are the steps of KNN?
# Step 1: Load the dataset.
# Step 2: Preprocess the data.
# Step 3: Split into training and testing data.
# Step 4: Scale the features.
# Step 5: Select the value of K.
# Step 6: Calculate distances.
# Step 7: Find K nearest neighbors.
# Step 8: Apply majority voting.
# Step 9: Predict the output class.
# Step 10: Evaluate the model.

# 6. What is the role of K?
# K specifies the number of nearest neighbors
# considered while predicting the class.

# 7. What happens when K = 1?
# The class of the nearest single neighbor
# determines the prediction.
# It can be sensitive to noise.

# 8. What happens when K is very large?
# The model may consider too many neighbors,
# potentially causing underfitting.

# 9. What happens when K is very small?
# The model may become sensitive to noise
# and overfit the training data.

# 10. Why is an odd value of K commonly selected?
# In binary classification, an odd K reduces
# the possibility of equal votes between two classes.
# However, ties can still occur in multiclass problems.

# 11. What is Distance Measurement?
# It measures the distance between two data points
# to identify the nearest neighbors.

# 12. What is Euclidean Distance?
# It is the straight-line distance between two points.
# KNN commonly uses Euclidean Distance.
#
# For two points:
# A = (x1, y1)
# B = (x2, y2)
#
# Distance = sqrt((x2-x1)^2 + (y2-y1)^2)

# 13. What other distance measures are available?
# - Manhattan Distance
# - Minkowski Distance
# - Hamming Distance
# - Cosine Distance

# 14. Why is StandardScaler used?
# KNN is distance-based.
# Features having larger ranges can dominate
# the distance calculation.
# StandardScaler brings features to comparable scales.

# 15. What is Majority Voting?
# The class that receives the maximum number
# of votes among K neighbors is selected
# as the predicted class.

# 16. What is Confusion Matrix?
# It is a table used to evaluate a classification model
# by comparing actual and predicted classes.

# 17. What are the four components of a Confusion Matrix?
#
# True Positive (TP):
# Actual positive and predicted positive.
#
# True Negative (TN):
# Actual negative and predicted negative.
#
# False Positive (FP):
# Actual negative but predicted positive.
#
# False Negative (FN):
# Actual positive but predicted negative.

# 18. What is Accuracy?
# It measures the proportion of correct predictions.
#
# Formula:
# Accuracy = (TP + TN) / (TP + TN + FP + FN)

# 19. What is Error Rate?
# It measures the proportion of incorrect predictions.
#
# Formula:
# Error Rate = (FP + FN) / Total Predictions
#
# Error Rate = 1 - Accuracy

# 20. What is Precision?
# Precision measures how many predicted positive
# cases are actually positive.
#
# Formula:
# Precision = TP / (TP + FP)

# 21. What is Recall?
# Recall measures how many actual positive cases
# are correctly identified.
#
# Formula:
# Recall = TP / (TP + FN)

# 22. What is F1 Score?
# It is the harmonic mean of Precision and Recall.
#
# Formula:
# F1 = 2 * Precision * Recall / (Precision + Recall)

# 23. What is the difference between Precision and Recall?
#
# Precision:
# Focuses on avoiding False Positives.
#
# Recall:
# Focuses on avoiding False Negatives.
#
# In medical screening, missing actual diabetic cases
# can be important, so Recall deserves attention.

# 24. What is Feature Scaling?
# It is the process of bringing features
# to a comparable numerical scale.

# 25. What is Standardization?
# It transforms data so that features have
# approximately mean 0 and standard deviation 1.

# 26. What is Train-Test Split?
# It divides the dataset into training and testing sets.
# Training data builds the model.
# Testing data evaluates its performance.

# 27. What is the purpose of random_state=42?
# It ensures reproducibility of the random train-test split.

# 28. What is stratify=y?
# It maintains approximately the same class distribution
# in training and testing sets.

# 29. What is a Lazy Learning Algorithm?
# KNN is called a lazy learner because it does not
# build an explicit general model during training.
# It stores training data and performs distance calculations
# when a prediction is required.

# 30. Is KNN a parametric or non-parametric algorithm?
# KNN is non-parametric because it does not assume
# a fixed mathematical distribution for the data.

# 31. What is the advantage of KNN?
# - Simple to understand and implement.
# - Supports classification and regression.
# - No complex training process.
# - Can classify nonlinear decision boundaries.

# 32. What are the disadvantages of KNN?
# - Prediction can be slow for large datasets.
# - Requires feature scaling.
# - Sensitive to irrelevant features.
# - Sensitive to noise and outliers.
# - Performance can decrease in high-dimensional data.

# 33. What is the Curse of Dimensionality?
# As the number of features increases,
# distance-based methods may become less effective
# because data points become sparse in high-dimensional space.

# 34. Why is KNN used for Diabetes Prediction?
# It identifies similar patient records based on
# their medical measurements and predicts the
# diabetes class from neighboring patients.

# 35. What is the target variable?
# Outcome in the common Pima diabetes dataset.
# 0 = Non-diabetic.
# 1 = Diabetic.

# 36. What are the input features?
# Common features include:
# - Pregnancies
# - Glucose
# - BloodPressure
# - SkinThickness
# - Insulin
# - BMI
# - DiabetesPedigreeFunction
# - Age

# 37. Why are zero values handled in some columns?
# In the common Pima dataset, zero values in
# certain medical measurement columns represent
# missing measurements rather than real values.
# These are treated as missing and imputed.

# 38. What is SimpleImputer?
# It replaces missing values with a selected statistic.
# Here, median imputation is used.

# 39. Why is median used instead of mean?
# Median is less affected by extreme values
# and can be more suitable for skewed medical data.

# 40. What is the difference between Classification
# and Regression?
#
# Classification:
# Predicts categorical outputs.
# Example: Diabetic or Non-diabetic.
#
# Regression:
# Predicts continuous numerical outputs.
# Example: House price prediction.

# 41. What is Data Leakage?
# It occurs when information from testing data
# improperly influences model training.
# This practical fits the imputer and scaler
# only on training data using preprocessing.

# 42. What is Model Evaluation?
# It is the process of measuring how well
# a trained model performs on unseen data.

# 43. Which metric is most important for this problem?
# It depends on the application.
# In medical screening, Recall can be particularly
# important because False Negatives may be harmful.
# Accuracy alone may be misleading for imbalanced data.

# 44. What is Class Imbalance?
# It occurs when one class has significantly
# more observations than another class.
# In such cases, accuracy alone may not be sufficient.

# 45. What is the difference between KNN and Decision Tree?
#
# KNN:
# Predicts using nearest data points and distance.
#
# Decision Tree:
# Predicts using a sequence of feature-based decisions.

# 46. What is the role of fit()?
# It stores the training data and prepares the KNN model.

# 47. What is the role of predict()?
# It predicts the class labels of new data points.

# 48. What is the role of confusion_matrix()?
# It compares actual and predicted labels
# and returns TN, FP, FN and TP counts.

# 49. What is the role of classification_report()?
# It displays precision, recall, F1-score and support
# for each class.

# 50. Does KNN require a separate training phase?
# It has no conventional parameter-learning phase.
# It stores training observations for later prediction.

# ==============================================================
# COMPLEXITY ANALYSIS
# ==============================================================

# Let:
# n = Number of training records.
# d = Number of input features.
# k = Number of nearest neighbors.

# 1. KNN TRAINING COMPLEXITY:
# Approximately O(1) model computation,
# excluding the time required to store the data.
#
# 2. KNN PREDICTION COMPLEXITY:
# O(n * d) per query using a brute-force search.
#
# Each query calculates distances to training records.

# 3. KNN SPACE COMPLEXITY:
# O(n * d)
# Training data must be stored for predictions.

# 4. STANDARDIZATION:
# O(n * d) for transforming the dataset.

# 5. CONFUSION MATRIX:
# O(m), where m = number of testing records.

# 6. ACCURACY, PRECISION AND RECALL:
# O(m), where m = number of testing records.

# IMPORTANT:
# Actual prediction performance depends on the
# number of records, features, distance metric
# and neighbor-search implementation.

# ==============================================================
# FINAL PRACTICAL SUMMARY
# ==============================================================

# Experiment:
# Diabetes Prediction Using KNN

# Dataset:
# diabetes.csv

# Learning Type:
# Supervised Machine Learning

# Algorithm:
# K-Nearest Neighbors Classification

# Target:
# Outcome (0 = Non-diabetic, 1 = Diabetic)
# in the common Pima diabetes dataset.

# Preprocessing:
# Duplicate Removal, Missing Value Handling,
# Target Encoding and Feature Scaling.

# Distance Metric:
# Euclidean Distance

# Default K:
# 5

# Training Data:
# 80%

# Testing Data:
# 20%

# Evaluation Metrics:
# 1. Confusion Matrix
# 2. Accuracy
# 3. Error Rate
# 4. Precision
# 5. Recall
# 6. F1 Score

# Visualization:
# Confusion Matrix Heatmap

# IMPORTANT EXTERNAL VIVA POINT:
# KNN is a supervised, non-parametric and lazy learning
# algorithm. It classifies a new record based on the
# majority class of its K nearest neighbors.
#
# Feature scaling is important because KNN uses
# distance calculations.
#
# ==============================================================
# END OF PRACTICAL
# ==============================================================
