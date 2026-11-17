
# ==============================================================
# MACHINE LEARNING PRACTICAL
# Experiment: Bank Customer Churn Prediction Using ANN
"""
Question: 
Given a bank customer, build a neural network-based classifier that 
can determine whether they will leave or not in the next 6 months. 
Dataset Description: The case study is from an open-source dataset 
from Kaggle. The dataset contains 10,000 sample points with 14 
distinct features such as Customer ID, Credit Score, Geography, 
Gender, Age, Tenure, Balance, etc. 
Perform the following steps: 
1. Read the dataset. 
2. Distinguish the feature and target set and divide the data set into 
training and test sets. 
3. Normalize the train and test data. 
4. Initialize and build the model. Identify the points of improvement 
and implement the same. 
5. Print the accuracy score and confusion matrix (5 points).
"""
# Objective:
# Build a Neural Network-based classification model
# to predict whether a bank customer will leave
# (Churn) or stay with the bank.
#
# Algorithm:
# Artificial Neural Network (ANN)
#
# Type:
# Supervised Machine Learning - Binary Classification
#
# Dataset:
# Churn_Modelling.csv
#
# Dataset Description:
# - Approximately 10,000 customer records.
# - Customer ID, Credit Score, Geography, Gender,
#   Age, Tenure, Balance, NumOfProducts, etc.
# - Target column: Exited
#
# Target:
# 0 = Customer stays
# 1 = Customer leaves (Churn)
#
# HOW TO RUN:
# Run this Python file CELL BY CELL in Jupyter Notebook
# or Google Colab.
#
# In VS Code, use Shift + Enter to execute each cell.
#
# INSTALL REQUIRED LIBRARIES:
# !pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
#
# ==============================================================


# ==============================================================
# CELL 1: IMPORT REQUIRED LIBRARIES
#
# Purpose:
# Import libraries required for data preprocessing,
# neural network creation, visualization and evaluation.
#
# ==============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Input
)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Set random seeds for reproducible results
np.random.seed(42)
tf.random.set_seed(42)


# ==============================================================
# CELL 2: LOAD THE DATASET
#
# Purpose:
# Read the bank customer dataset using Pandas.
# Display first five records, shape and dataset information.
#
# ==============================================================

df = pd.read_csv("Churn_Modelling.csv")

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
# Prepare the dataset before training the neural network.
#
# 1. Remove duplicate records.
# 2. Identify the target column.
# 3. Remove unnecessary ID columns.
# 4. Separate input features and target variable.
#
# Important:
# CustomerId, RowNumber and Surname are not useful
# predictive measurements, so they are removed.
#
# Geography and Gender are categorical features.
# These will be converted into numerical values later.
#
# ==============================================================

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Identify the target column automatically
target_candidates = [
    "exited",
    "churn",
    "churned",
    "attrition",
    "target"
]

target_column = None

for col in df.columns:

    if col.strip().lower() in target_candidates:
        target_column = col
        break

if target_column is None:

    print("\nAvailable Columns:", list(df.columns))

    target_column = input("Enter target column name: ").strip()

    if target_column not in df.columns:
        raise ValueError("Target column not found!")

print("\nTarget Column:", target_column)

# Remove records having missing target values
df.dropna(subset=[target_column], inplace=True)

# Separate target variable
y = df[target_column]

# Remove target from input features
X = df.drop(columns=[target_column])

# Remove unnecessary identification columns
# These columns do not represent useful customer behavior.
id_columns = [
    "RowNumber",
    "CustomerId",
    "CustomerID",
    "Surname",
    "ID"
]

X = X.drop(
    columns=[
        col for col in X.columns
        if col.strip().lower() in [c.lower() for c in id_columns]
    ],
    errors="ignore"
)

# Replace infinite values with missing values
X.replace([np.inf, -np.inf], np.nan, inplace=True)

print("\nSelected Input Features:")
print(X.head())

print("\nTarget Distribution:")
print(y.value_counts())


# ==============================================================
# CELL 4: ENCODE THE TARGET VARIABLE
#
# Purpose:
# Convert the target labels into numerical values.
#
# Label Encoding:
# Converts categorical class labels into numerical labels.
#
# In the standard Kaggle dataset:
# 0 = Customer stayed.
# 1 = Customer exited.
#
# ==============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

print("\nTarget Class Mapping:")

for i, label in enumerate(label_encoder.classes_):
    print(i, "=", label)

# This practical requires binary classification
if len(label_encoder.classes_) != 2:
    raise ValueError("Target must contain exactly two classes.")

print("\nTarget Encoding Completed!")


# ==============================================================
# CELL 5: IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
#
# Purpose:
# Separate numerical and categorical columns.
#
# Numerical Features:
# CreditScore, Age, Tenure, Balance, etc.
#
# Categorical Features:
# Geography, Gender, etc.
#
# ==============================================================

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# ==============================================================
# CELL 6: SPLIT DATASET INTO TRAINING AND TESTING DATA
#
# Purpose:
# Divide the dataset into training and testing sets.
#
# Training Data = 80%
# Testing Data = 20%
#
# Training data is used to train the ANN model.
# Testing data is used to evaluate its performance.
#
# stratify=y maintains the class distribution.
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
# CELL 7: NORMALIZE AND ENCODE THE DATA
#
# Purpose:
# Normalize numerical features and encode categorical features.
#
# Numerical Features:
# - Missing values are replaced using median.
# - StandardScaler converts features to comparable scales.
#
# Categorical Features:
# - Missing values are replaced using most frequent value.
# - OneHotEncoder converts categories into binary columns.
#
# IMPORTANT:
# Preprocessing is fitted only on training data.
# The same transformation is applied to testing data.
# This prevents data leakage.
#
# ==============================================================

# Create numerical preprocessing pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Create categorical preprocessing pipeline
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

# Combine both pipelines
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Fit on training data and transform
X_train_scaled = preprocessor.fit_transform(X_train)

# Transform testing data
X_test_scaled = preprocessor.transform(X_test)

# Ensure numerical NumPy arrays
X_train_scaled = np.asarray(X_train_scaled, dtype=np.float32)
X_test_scaled = np.asarray(X_test_scaled, dtype=np.float32)

print("\nTraining Data After Normalization:")
print(X_train_scaled[:5])

print("\nTesting Data After Normalization:")
print(X_test_scaled[:5])

print("\nNormalized Training Shape:", X_train_scaled.shape)
print("Normalized Testing Shape:", X_test_scaled.shape)


# ==============================================================
# CELL 8: BUILD THE INITIAL NEURAL NETWORK MODEL
#
# Purpose:
# Create a basic Artificial Neural Network (ANN).
#
# Architecture:
# Input Layer
# Hidden Layer 1 = 16 Neurons
# Hidden Layer 2 = 8 Neurons
# Output Layer = 1 Neuron
#
# Activation Functions:
# ReLU = Hidden layers.
# Sigmoid = Output layer.
#
# Sigmoid returns a probability between 0 and 1.
#
# ==============================================================

input_features = X_train_scaled.shape[1]

# Create the initial ANN model
model = Sequential([

    Input(shape=(input_features,)),

    Dense(16, activation="relu"),

    Dense(8, activation="relu"),

    Dense(1, activation="sigmoid")

])

# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nInitial Neural Network Model:")
model.summary()


# ==============================================================
# CELL 9: TRAIN THE INITIAL MODEL
#
# Purpose:
# Train the neural network using training data.
#
# Epochs = 30
# Batch Size = 32
#
# Validation Split = 20% of training data.
#
# ==============================================================

initial_history = model.fit(
    X_train_scaled,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.20,
    verbose=1
)

print("\nInitial Model Training Completed!")


# ==============================================================
# CELL 10: EVALUATE THE INITIAL MODEL
#
# Purpose:
# Evaluate the initial model using testing data.
# Calculate its initial accuracy.
#
# ==============================================================

initial_loss, initial_accuracy = model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

print("\nInitial Model Results:")

print("Initial Loss:", round(initial_loss, 4))

print(
    "Initial Accuracy:",
    round(initial_accuracy * 100, 2),
    "%"
)


# ==============================================================
# CELL 11: IDENTIFY POINTS OF IMPROVEMENT
#
# Purpose:
# Improve the initial neural network model.
#
# Improvements:
#
# 1. Increase Hidden Neurons:
#    Use 64 and 32 neurons instead of 16 and 8.
#    This increases the learning capacity.
#
# 2. Batch Normalization:
#    Helps stabilize and speed up training.
#
# 3. Dropout:
#    Randomly disables some neurons during training
#    to reduce overfitting.
#
# 4. L2 Regularization:
#    Penalizes excessively large weights.
#
# 5. Early Stopping:
#    Stops training when validation loss stops improving.
#
# 6. Reduce Learning Rate:
#    Reduces learning rate when validation loss plateaus.
#
# 7. Class Weights:
#    Gives more importance to minority class records.
#    Useful when customer churn classes are imbalanced.
#
# ==============================================================

# Calculate balanced class weights
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}

print("\nClass Weights:")
print(class_weights)


# ==============================================================
# CELL 12: BUILD THE IMPROVED NEURAL NETWORK
#
# Purpose:
# Build an improved ANN using additional layers,
# Dropout, Batch Normalization and L2 Regularization.
#
# Improved Architecture:
#
# Input Layer
# Hidden Layer 1 = 64 Neurons + ReLU
# Batch Normalization
# Dropout = 30%
#
# Hidden Layer 2 = 32 Neurons + ReLU
# Batch Normalization
# Dropout = 20%
#
# Output Layer = 1 Neuron + Sigmoid
#
# ==============================================================

improved_model = Sequential([

    Input(shape=(input_features,)),

    Dense(
        64,
        activation="relu",
        kernel_regularizer=l2(0.001)
    ),

    BatchNormalization(),

    Dropout(0.30),

    Dense(
        32,
        activation="relu",
        kernel_regularizer=l2(0.001)
    ),

    BatchNormalization(),

    Dropout(0.20),

    Dense(1, activation="sigmoid")

])

# Compile improved model
improved_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nImproved Neural Network Model:")
improved_model.summary()


# ==============================================================
# CELL 13: IMPLEMENT EARLY STOPPING AND LEARNING RATE REDUCTION
#
# Purpose:
# Improve training performance and reduce overfitting.
#
# EarlyStopping:
# Stops training if validation loss does not improve
# for 5 consecutive epochs.
#
# ReduceLROnPlateau:
# Reduces the learning rate if validation loss
# stops improving.
#
# ==============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=0.00001
)

print("\nCallbacks Created Successfully!")


# ==============================================================
# CELL 14: TRAIN THE IMPROVED NEURAL NETWORK
#
# Purpose:
# Train the improved ANN using training data.
#
# Epochs = 50
# Batch Size = 32
# Validation Split = 20%
#
# Class Weights are applied to handle class imbalance.
#
# ==============================================================

improved_history = improved_model.fit(

    X_train_scaled,
    y_train,

    epochs=50,

    batch_size=32,

    validation_split=0.20,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)

print("\nImproved Model Training Completed!")


# ==============================================================
# CELL 15: PLOT TRAINING AND VALIDATION ACCURACY
#
# Purpose:
# Visualize model accuracy during training.
#
# Training Accuracy:
# Accuracy calculated on training data.
#
# Validation Accuracy:
# Accuracy calculated on validation data.
#
# ==============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    improved_history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    improved_history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Improved ANN - Accuracy")

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()


# ==============================================================
# CELL 16: PLOT TRAINING AND VALIDATION LOSS
#
# Purpose:
# Visualize training loss and validation loss.
#
# Loss measures the difference between predicted
# and actual values.
#
# Lower loss generally indicates better learning.
#
# ==============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    improved_history.history["loss"],
    label="Training Loss"
)

plt.plot(
    improved_history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Improved ANN - Loss")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()


# ==============================================================
# CELL 17: EVALUATE THE IMPROVED MODEL
#
# Purpose:
# Evaluate the improved ANN using testing data.
#
# ==============================================================

test_loss, test_accuracy = improved_model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

print("\nImproved Model Evaluation:")

print("Test Loss:", round(test_loss, 4))

print(
    "Test Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)


# ==============================================================
# CELL 18: MAKE PREDICTIONS
#
# Purpose:
# Predict whether customers will leave or stay.
#
# Sigmoid Output:
# Probability >= 0.5 means customer will leave.
# Probability < 0.5 means customer will stay.
#
# ==============================================================

# Predict churn probabilities
y_probability = improved_model.predict(
    X_test_scaled,
    verbose=0
).ravel()

# Convert probabilities into binary class labels
y_pred = (y_probability >= 0.5).astype(int)

print("\nFirst 10 Actual Values:")
print(y_test[:10])

print("\nFirst 10 Predicted Values:")
print(y_pred[:10])


# ==============================================================
# CELL 19: CALCULATE ACCURACY SCORE
#
# Purpose:
# Calculate the percentage of correct predictions.
#
# Accuracy:
# Correct Predictions / Total Predictions.
#
# ==============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nFinal Accuracy Score:", round(accuracy, 4))

print(
    "Accuracy Percentage:",
    round(accuracy * 100, 2),
    "%"
)


# ==============================================================
# CELL 20: GENERATE CONFUSION MATRIX
#
# Purpose:
# Compare actual and predicted customer churn values.
#
# Confusion Matrix:
#
#                  Predicted
#                  0        1
# Actual  0       TN       FP
#         1       FN       TP
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

# Extract matrix values
tn, fp, fn, tp = cm.ravel()

print("\nTrue Negative:", tn)

print("False Positive:", fp)

print("False Negative:", fn)

print("True Positive:", tp)


# ==============================================================
# CELL 21: VISUALIZE CONFUSION MATRIX
#
# Purpose:
# Display the confusion matrix using a Seaborn heatmap.
#
# ==============================================================

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stayed", "Exited"],
    yticklabels=["Stayed", "Exited"]
)

plt.xlabel("Predicted Values")

plt.ylabel("Actual Values")

plt.title("Bank Customer Churn - Confusion Matrix")

plt.show()


# ==============================================================
# CELL 22: CALCULATE OTHER PERFORMANCE METRICS
#
# Purpose:
# Calculate Precision, Recall and F1 Score.
#
# Precision:
# How many predicted churn customers actually left.
#
# Recall:
# How many actual churn customers were correctly detected.
#
# F1 Score:
# Harmonic mean of Precision and Recall.
#
# ==============================================================

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

error_rate = 1 - accuracy

print("\nPrecision:", round(precision, 4))

print("Recall:", round(recall, 4))

print("F1 Score:", round(f1, 4))

print("Error Rate:", round(error_rate, 4))


# ==============================================================
# CELL 23: COMPARE INITIAL AND IMPROVED MODEL
#
# Purpose:
# Compare the accuracy of the initial neural network
# and improved neural network.
#
# ==============================================================

comparison = pd.DataFrame({

    "Model": [
        "Initial ANN",
        "Improved ANN"
    ],

    "Accuracy": [
        initial_accuracy,
        accuracy_score(
            y_test,
            (model.predict(X_test_scaled, verbose=0).ravel() >= 0.5).astype(int)
        )
    ],

    "Test Loss": [
        initial_loss,
        test_loss
    ]

})

print("\nMODEL PERFORMANCE COMPARISON:")

print(comparison.to_string(index=False))


# ==============================================================
# CELL 24: DISPLAY CLASSIFICATION REPORT
#
# Purpose:
# Display Precision, Recall, F1 Score and Support
# for both customer classes.
#
# ==============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Stayed (0)", "Exited (1)"],
        zero_division=0
    )
)


# ==============================================================
# IMPORTANT THEORY AND EXTERNAL VIVA QUESTIONS
# ==============================================================

# 1. What is Artificial Intelligence?
# Artificial Intelligence enables machines to perform
# tasks that generally require human intelligence.

# 2. What is Machine Learning?
# Machine Learning enables computers to learn patterns
# from data and make predictions.

# 3. What is Artificial Neural Network (ANN)?
# ANN is a Machine Learning model inspired by the
# structure and functioning of biological neural networks.
# It contains interconnected artificial neurons.

# 4. What is Customer Churn?
# Customer churn means a customer stops using
# the services of a company or bank.

# 5. What is the objective of this practical?
# Predict whether a bank customer will leave
# or remain with the bank.

# 6. Is this supervised or unsupervised learning?
# Supervised Learning because the dataset contains
# labelled customer churn information.

# 7. Is this classification or regression?
# Binary Classification.
# The output contains two classes:
# 0 = Stayed
# 1 = Exited.

# 8. What is an Input Layer?
# It receives the input features from the dataset.

# 9. What is a Hidden Layer?
# It processes input information using neurons
# and learns patterns from the data.

# 10. What is an Output Layer?
# It produces the final prediction of the neural network.

# 11. What is a Neuron?
# A neuron is the basic processing unit of a neural network.
# It receives inputs, applies weights and bias,
# and produces an output using an activation function.

# 12. What is an Epoch?
# One complete pass through the entire training dataset.

# 13. What is Batch Size?
# Number of training samples processed before
# the model updates its weights.

# 14. What is an Activation Function?
# It introduces non-linearity into a neural network
# so that it can learn complex relationships.

# 15. What is ReLU?
# Rectified Linear Unit.
# It returns zero for negative inputs
# and the input itself for positive inputs.

# 16. What is Sigmoid?
# It converts an input into a value between 0 and 1.
# It is commonly used in binary classification output layers.

# 17. Why is Sigmoid used in the output layer?
# Because the problem has two possible classes.
# The output represents the probability of customer churn.

# 18. What is the purpose of the Adam optimizer?
# Adam updates model weights using adaptive learning rates.
# It combines ideas from momentum and RMSProp.

# 19. What is Loss Function?
# A loss function measures the difference between
# actual values and model predictions.

# 20. What is Binary Cross-Entropy?
# It is a loss function used in binary classification.
# It penalizes incorrect probability predictions.

# 21. What is Backpropagation?
# It calculates gradients of the loss with respect
# to model weights and propagates the error backwards
# through the neural network.

# 22. What is Forward Propagation?
# Information moves from the input layer through
# hidden layers to produce the final output.

# 23. What is StandardScaler?
# It standardizes numerical features to approximately
# zero mean and unit variance.

# 24. Why is normalization important?
# Features with large numerical ranges can influence
# neural network training disproportionately.
# Scaling helps make training more stable.

# 25. What is One-Hot Encoding?
# It converts categorical values into separate
# binary columns.
# Example:
# Gender = Male/Female becomes binary feature columns.

# 26. Why is Geography encoded?
# Neural networks require numerical inputs.
# One-Hot Encoding converts geographical categories
# into numerical representations.

# 27. Why do we remove Customer ID?
# Customer ID is an identifier, not a meaningful
# customer behavior feature.
# It may introduce irrelevant patterns.

# 28. What is Overfitting?
# When a model learns training data too closely
# and performs poorly on unseen data.

# 29. What is Underfitting?
# When a model fails to learn important patterns
# from the training dataset.

# 30. What is Dropout?
# Dropout randomly disables a percentage of neurons
# during training to reduce overfitting.

# 31. What is Batch Normalization?
# It normalizes layer activations during training
# and can improve training stability.

# 32. What is L2 Regularization?
# It adds a penalty based on the squared magnitude
# of model weights to the loss function.
# It discourages excessively large weights.

# 33. What is Early Stopping?
# It stops training when the monitored validation
# metric stops improving for a specified number of epochs.

# 34. What is ReduceLROnPlateau?
# It reduces the learning rate when model performance
# stops improving on the monitored validation metric.

# 35. What is Class Imbalance?
# It occurs when one class has more records
# than another class.

# 36. What is Class Weight?
# It assigns different importance to classes
# during model training.
# It helps the model pay more attention
# to minority class observations.

# 37. What is Validation Data?
# Data used during training to monitor model performance
# and help identify overfitting.

# 38. What is Testing Data?
# Unseen data used to evaluate the final model.

# 39. What is Accuracy?
# Proportion of correctly classified records.
#
# Accuracy = (TP + TN) / Total Records

# 40. What is Confusion Matrix?
# A table comparing actual and predicted classes.
#
# TN = True Negative
# FP = False Positive
# FN = False Negative
# TP = True Positive

# 41. What is Precision?
# Precision = TP / (TP + FP)
# It measures correctness among predicted positives.

# 42. What is Recall?
# Recall = TP / (TP + FN)
# It measures how many actual positives are identified.

# 43. What is F1 Score?
# It is the harmonic mean of Precision and Recall.

# 44. What is the difference between Accuracy and Recall?
# Accuracy measures overall correct predictions.
# Recall measures correct identification of actual
# positive cases.

# 45. What is the purpose of random_state=42?
# It ensures reproducibility of the train-test split.

# 46. What is the purpose of validation_split=0.20?
# It reserves 20% of training data for validation.
# This data is not used for gradient updates.

# 47. What is the purpose of model.compile()?
# It configures the optimizer, loss function
# and evaluation metrics of the neural network.

# 48. What is the purpose of model.fit()?
# It trains the neural network using training data.

# 49. What is the purpose of model.predict()?
# It generates predictions for new data.

# 50. Why are two models implemented?
# The initial model provides a baseline.
# The improved model uses regularization,
# additional neurons, class weights and callbacks
# to improve generalization.

# 51. What is the difference between ANN and CNN?
# ANN is a general neural network architecture.
# CNN uses convolutional layers and is commonly used
# for image and spatial pattern processing.

# 52. What is the difference between ANN and RNN?
# ANN generally processes independent input features.
# RNN is designed to process sequential data.

# 53. What is a Learning Rate?
# It determines how much model weights are updated
# during each optimization step.

# 54. What happens if the Learning Rate is too high?
# Training may become unstable and fail to converge.

# 55. What happens if the Learning Rate is too low?
# Training may become very slow.

# 56. What is a Weight?
# A trainable parameter controlling the importance
# of a connection between neurons.

# 57. What is Bias?
# A trainable parameter added to the weighted input
# before applying the activation function.

# 58. What is Model Generalization?
# The ability of a trained model to perform well
# on unseen customer records.

# 59. Can this model guarantee that a customer will leave?
# No. It estimates a probability based on historical data.
# Actual customer behavior cannot be guaranteed.

# 60. Important practical point:
# The dataset's Exited column gives the churn label.
# A specific six-month prediction horizon is valid only
# if the dataset's target definition represents churn
# within the next six months. Do not assume the horizon
# from the column name alone.

# ==============================================================
# COMPLEXITY ANALYSIS
# ==============================================================

# Let:
# n = Number of training records.
# d = Number of input features after encoding.
# h1 = Number of neurons in hidden layer 1.
# h2 = Number of neurons in hidden layer 2.
# E = Number of training epochs.

# 1. NEURAL NETWORK TRAINING:
#
# Approximate time complexity:
# O(E * n * (d*h1 + h1*h2 + h2))
#
# This includes forward propagation and backpropagation
# up to constant factors.

# 2. NEURAL NETWORK PREDICTION:
#
# Approximate time complexity per record:
# O(d*h1 + h1*h2 + h2)

# 3. MODEL SPACE COMPLEXITY:
#
# Depends on the number of weights, biases,
# optimizer state and stored activations.
#
# Model parameters are approximately:
# (d + 1)*h1 + (h1 + 1)*h2 + (h2 + 1)
#
# Training also requires additional memory
# for optimizer states and batches.

# 4. DATA PREPROCESSING:
#
# Standardization:
# O(n*d)
#
# One-Hot Encoding:
# Depends on the number of categorical features
# and their unique categories.

# 5. CONFUSION MATRIX:
#
# O(m), where m = number of testing records.

# IMPORTANT:
# Actual training time depends on dataset size,
# number of layers, neurons, epochs, batch size
# and available hardware.

# ==============================================================
# FINAL PRACTICAL SUMMARY
# ==============================================================

# Experiment:
# Bank Customer Churn Prediction Using ANN

# Dataset:
# Churn_Modelling.csv

# Learning Type:
# Supervised Machine Learning

# Problem Type:
# Binary Classification

# Target Variable:
# Exited
#
# 0 = Customer Stayed
# 1 = Customer Exited

# Preprocessing:
# Duplicate Removal, ID Column Removal,
# Missing Value Handling, One-Hot Encoding
# and Standardization.

# Data Splitting:
# 80% Training
# 20% Testing

# Initial Model:
# 16 and 8 Hidden Neurons.

# Improved Model:
# 64 and 32 Hidden Neurons,
# Batch Normalization, Dropout and L2 Regularization.

# Optimizer:
# Adam

# Loss Function:
# Binary Cross-Entropy

# Activation Functions:
# ReLU and Sigmoid

# Improvements:
# Class Weights, Early Stopping and
# Reduce Learning Rate on Plateau.

# Evaluation:
# Accuracy, Confusion Matrix, Precision,
# Recall and F1 Score.

# Visualizations:
# Training Accuracy, Training Loss
# and Confusion Matrix Heatmap.

# IMPORTANT EXTERNAL VIVA POINT:
# ANN learns customer churn patterns through
# interconnected neurons, weights, biases,
# activation functions and backpropagation.
#
# The initial model is compared with an improved model
# to understand the effects of regularization,
# dropout, batch normalization and early stopping.
#
# ==============================================================
# END OF PRACTICAL
# ==============================================================
