
# ==============================================================
# MACHINE LEARNING PRACTICAL
# Experiment: Email Spam Detection Using KNN and SVM
"""
Question : 
Classify the email using the binary classification method. Email 
Spam detection has two states: 
a) Normal State - Not Spam 
b) Abnormal State - Spam. 
Use K-Nearest Neighbors and Support Vector Machine for 
classification. Analyze their performance.
"""
# Objective:
# Classify emails into two categories using Machine Learning:
#
# 1. Normal State (Not Spam) = 0
# 2. Abnormal State (Spam) = 1
#
# Algorithms:
# 1. K-Nearest Neighbors (KNN)
# 2. Support Vector Machine (SVM)
#
# Tasks:
# 1. Read the email spam dataset.
# 2. Preprocess email text.
# 3. Convert text into numerical features using TF-IDF.
# 4. Split data into training and testing sets.
# 5. Implement KNN classification.
# 6. Implement SVM classification.
# 7. Calculate accuracy, precision, recall and F1-score.
# 8. Generate confusion matrices.
# 9. Compare both algorithms.
#
# HOW TO RUN:
# Run this Python file CELL BY CELL in Jupyter Notebook
# or Google Colab.
#
# In VS Code, use Shift + Enter to execute each cell.
#
# Dataset:
# spam.csv
#
# Common dataset column formats:
# v1, v2
# Category, Message
# label, text
# spam, email
#
# INSTALL REQUIRED LIBRARIES:
# !pip install pandas numpy matplotlib seaborn scikit-learn
# ==============================================================


# ==============================================================
# CELL 1: IMPORT REQUIRED LIBRARIES
#
# Purpose:
# Import libraries for data preprocessing,
# text feature extraction, classification and evaluation.
#
# ==============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Set random seed for reproducibility
np.random.seed(42)


# ==============================================================
# CELL 2: LOAD THE DATASET
#
# Purpose:
# Read the spam email dataset using Pandas.
# Display the first five records, shape and information.
#
# ==============================================================

# Read dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Remove completely empty columns
df.dropna(axis=1, how="all", inplace=True)

print("First Five Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nAvailable Columns:")
print(list(df.columns))


# ==============================================================
# CELL 3: IDENTIFY TEXT AND TARGET COLUMNS
#
# Purpose:
# Automatically identify:
# 1. Email text column.
# 2. Spam classification column.
#
# Common Target Columns:
# label, category, v1, spam, target.
#
# Common Text Columns:
# message, text, v2, email, content, body.
#
# ==============================================================

# Automatically identify target column
target_candidates = [
    "label",
    "category",
    "v1",
    "spam",
    "target",
    "class",
    "is_spam"
]

target_column = None

for col in df.columns:

    if col.strip().lower() in target_candidates:
        target_column = col
        break

# Ask user if target column is not identified
if target_column is None:

    print("\nAvailable Columns:", list(df.columns))

    target_column = input("Enter target column name: ").strip()

    if target_column not in df.columns:
        raise ValueError("Target column not found!")

# Automatically identify email text column
text_candidates = [
    "message",
    "v2",
    "text",
    "emailtext",
    "email_text",
    "email",
    "content",
    "body",
    "mail"
]

text_column = None

for col in df.columns:

    if col.strip().lower() in text_candidates:
        if col != target_column:
            text_column = col
            break

# Ask user if text column is not identified
if text_column is None:

    print("\nAvailable Columns:", list(df.columns))

    text_column = input("Enter email text column name: ").strip()

    if text_column not in df.columns or text_column == target_column:
        raise ValueError("Invalid email text column!")

print("\nSelected Target Column:", target_column)
print("Selected Email Text Column:", text_column)


# ==============================================================
# CELL 4: DATA PREPROCESSING
#
# Purpose:
# Clean the dataset before classification.
#
# 1. Select only email text and target columns.
# 2. Remove missing records.
# 3. Remove duplicate records.
# 4. Convert email text into string format.
# 5. Remove empty email messages.
#
# ==============================================================

# Select required columns
df = df[[text_column, target_column]].copy()

# Rename columns for simplicity
df.columns = ["email", "label"]

# Remove missing values
df.dropna(inplace=True)

# Convert email text into string
df["email"] = df["email"].astype(str)

# Remove unnecessary spaces
df["email"] = df["email"].str.strip()

# Remove empty email messages
df = df[df["email"] != ""]

# Remove duplicate records
df.drop_duplicates(inplace=True)

print("\nDataset After Preprocessing:")
print(df.head())

print("\nDataset Shape After Cleaning:")
print(df.shape)


# ==============================================================
# CELL 5: CONVERT TARGET LABELS INTO BINARY VALUES
#
# Purpose:
# Convert email categories into numerical labels.
#
# Normal State = Not Spam = 0
# Abnormal State = Spam = 1
#
# Common Dataset Labels:
# ham  = Not Spam
# spam = Spam
#
# ==============================================================

def convert_label(value):

    value = str(value).strip().lower()

    value = value.replace("_", " ").replace("-", " ")

    # Normal / Not Spam
    if value in [
        "ham",
        "not spam",
        "normal",
        "legitimate",
        "0",
        "0.0",
        "false",
        "no"
    ]:
        return 0

    # Abnormal / Spam
    elif value in [
        "spam",
        "abnormal",
        "junk",
        "1",
        "1.0",
        "true",
        "yes"
    ]:
        return 1

    else:
        return np.nan


# Apply binary label conversion
df["label"] = df["label"].apply(convert_label)

# Remove records with unknown labels
df.dropna(subset=["label"], inplace=True)

df["label"] = df["label"].astype(int)

if df["label"].nunique() != 2:
    raise ValueError(
        "Dataset must contain both Spam and Not Spam classes."
    )

print("\nBinary Label Conversion Completed!")

print("\nLabel Mapping:")
print("0 = Normal (Not Spam)")
print("1 = Abnormal (Spam)")

print("\nClass Distribution:")
print(df["label"].value_counts())

print("\nClass Distribution Percentage:")
print(df["label"].value_counts(normalize=True) * 100)


# ==============================================================
# CELL 6: SPLIT THE DATASET
#
# Purpose:
# Divide the dataset into training and testing sets.
#
# Training Data = 80%
# Testing Data = 20%
#
# Training data is used to train the models.
# Testing data is used to evaluate predictions.
#
# stratify=y maintains the class distribution.
#
# ==============================================================

X = df["email"]

y = df["label"]

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
print(y_train.value_counts())


# ==============================================================
# CELL 7: TEXT FEATURE EXTRACTION USING TF-IDF
#
# Purpose:
# Convert email text into numerical features.
#
# Machine Learning algorithms cannot directly understand
# raw email text.
#
# TF-IDF:
# Term Frequency - Inverse Document Frequency.
#
# It assigns importance to words based on:
# 1. Frequency of a word in an email.
# 2. How common or rare the word is across all emails.
#
# ==============================================================

# Create TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

# Fit on training text only
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform testing text using the same vocabulary
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF Transformation Completed!")

print("\nTraining TF-IDF Shape:")
print(X_train_tfidf.shape)

print("\nTesting TF-IDF Shape:")
print(X_test_tfidf.shape)


# ==============================================================
# CELL 8: IMPLEMENT K-NEAREST NEIGHBORS (KNN)
#
# Purpose:
# Implement KNN classification for email spam detection.
#
# Working:
# 1. Select the value of K.
# 2. Calculate distances between email feature vectors.
# 3. Find K nearest emails.
# 4. Identify the majority class.
# 5. Predict Spam or Not Spam.
#
# K = 5
# Distance Metric = Cosine Distance
#
# Cosine distance is useful for comparing text vectors.
#
# ==============================================================

# Select K value
k_input = input("\nEnter K value for KNN (Press Enter for 5): ")

if k_input.strip() == "":
    k = 5
else:
    k = int(k_input)

if k < 1 or k > len(X_train):
    raise ValueError("Invalid K value!")

# Create KNN classifier
knn = KNeighborsClassifier(
    n_neighbors=k,
    metric="cosine",
    algorithm="brute",
    weights="distance"
)

# Train KNN model
knn.fit(X_train_tfidf, y_train)

print("\nKNN Model Trained Successfully!")

# Predict testing data
knn_predictions = knn.predict(X_test_tfidf)

print("\nFirst 10 KNN Predictions:")
print(knn_predictions[:10])


# ==============================================================
# CELL 9: IMPLEMENT SUPPORT VECTOR MACHINE (SVM)
#
# Purpose:
# Implement SVM classification for spam detection.
#
# SVM:
# Finds an optimal decision boundary (hyperplane)
# that separates Spam and Not Spam emails.
#
# Linear SVM is suitable for high-dimensional
# TF-IDF text features.
#
# ==============================================================

# Create Linear SVM classifier
svm = LinearSVC(
    C=1.0,
    max_iter=5000,
    random_state=42
)

# Train SVM model
svm.fit(X_train_tfidf, y_train)

print("\nSVM Model Trained Successfully!")

# Predict testing data
svm_predictions = svm.predict(X_test_tfidf)

print("\nFirst 10 SVM Predictions:")
print(svm_predictions[:10])


# ==============================================================
# CELL 10: EVALUATE KNN MODEL
#
# Purpose:
# Calculate the performance metrics of KNN.
#
# Metrics:
# 1. Accuracy
# 2. Error Rate
# 3. Precision
# 4. Recall
# 5. F1 Score
#
# Positive Class = 1 (Spam)
#
# ==============================================================

knn_accuracy = accuracy_score(
    y_test,
    knn_predictions
)

knn_error = 1 - knn_accuracy

knn_precision = precision_score(
    y_test,
    knn_predictions,
    pos_label=1,
    zero_division=0
)

knn_recall = recall_score(
    y_test,
    knn_predictions,
    pos_label=1,
    zero_division=0
)

knn_f1 = f1_score(
    y_test,
    knn_predictions,
    pos_label=1,
    zero_division=0
)

print("\nKNN PERFORMANCE:")

print("Accuracy:", round(knn_accuracy, 4))

print("Error Rate:", round(knn_error, 4))

print("Precision:", round(knn_precision, 4))

print("Recall:", round(knn_recall, 4))

print("F1 Score:", round(knn_f1, 4))


# ==============================================================
# CELL 11: GENERATE KNN CONFUSION MATRIX
#
# Purpose:
# Compare actual and predicted KNN classifications.
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

knn_cm = confusion_matrix(
    y_test,
    knn_predictions,
    labels=[0, 1]
)

print("\nKNN Confusion Matrix:")
print(knn_cm)

tn, fp, fn, tp = knn_cm.ravel()

print("\nTrue Negative:", tn)

print("False Positive:", fp)

print("False Negative:", fn)

print("True Positive:", tp)


# ==============================================================
# CELL 12: EVALUATE SVM MODEL
#
# Purpose:
# Calculate the performance metrics of SVM.
#
# Higher Accuracy, Precision, Recall and F1 Score
# generally indicate better classification performance.
#
# ==============================================================

svm_accuracy = accuracy_score(
    y_test,
    svm_predictions
)

svm_error = 1 - svm_accuracy

svm_precision = precision_score(
    y_test,
    svm_predictions,
    pos_label=1,
    zero_division=0
)

svm_recall = recall_score(
    y_test,
    svm_predictions,
    pos_label=1,
    zero_division=0
)

svm_f1 = f1_score(
    y_test,
    svm_predictions,
    pos_label=1,
    zero_division=0
)

print("\nSVM PERFORMANCE:")

print("Accuracy:", round(svm_accuracy, 4))

print("Error Rate:", round(svm_error, 4))

print("Precision:", round(svm_precision, 4))

print("Recall:", round(svm_recall, 4))

print("F1 Score:", round(svm_f1, 4))


# ==============================================================
# CELL 13: GENERATE SVM CONFUSION MATRIX
#
# Purpose:
# Compare actual and predicted SVM classifications.
#
# ==============================================================

svm_cm = confusion_matrix(
    y_test,
    svm_predictions,
    labels=[0, 1]
)

print("\nSVM Confusion Matrix:")
print(svm_cm)

tn, fp, fn, tp = svm_cm.ravel()

print("\nTrue Negative:", tn)

print("False Positive:", fp)

print("False Negative:", fn)

print("True Positive:", tp)


# ==============================================================
# CELL 14: VISUALIZE BOTH CONFUSION MATRICES
#
# Purpose:
# Display KNN and SVM confusion matrices using heatmaps.
#
# This helps compare the correct and incorrect predictions
# of both classification algorithms.
#
# ==============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

sns.heatmap(
    knn_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Spam", "Spam"],
    yticklabels=["Not Spam", "Spam"]
)

plt.title("KNN Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")


plt.subplot(1, 2, 2)

sns.heatmap(
    svm_cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Not Spam", "Spam"],
    yticklabels=["Not Spam", "Spam"]
)

plt.title("SVM Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.show()


# ==============================================================
# CELL 15: DISPLAY CLASSIFICATION REPORTS
#
# Purpose:
# Generate a detailed classification report
# for KNN and SVM.
#
# Report contains:
# Precision, Recall, F1 Score and Support.
#
# ==============================================================

print("\nKNN CLASSIFICATION REPORT:")

print(
    classification_report(
        y_test,
        knn_predictions,
        labels=[0, 1],
        target_names=["Not Spam", "Spam"],
        zero_division=0
    )
)


print("\nSVM CLASSIFICATION REPORT:")

print(
    classification_report(
        y_test,
        svm_predictions,
        labels=[0, 1],
        target_names=["Not Spam", "Spam"],
        zero_division=0
    )
)


# ==============================================================
# CELL 16: COMPARE KNN AND SVM PERFORMANCE
#
# Purpose:
# Compare both models using their evaluation scores.
#
# Higher Accuracy, Precision, Recall and F1 Score
# are generally better.
#
# Lower Error Rate is better.
#
# F1 Score is especially useful when classes are imbalanced.
#
# ==============================================================

comparison = pd.DataFrame({

    "Evaluation Metric": [
        "Accuracy",
        "Error Rate",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "KNN": [
        knn_accuracy,
        knn_error,
        knn_precision,
        knn_recall,
        knn_f1
    ],

    "SVM": [
        svm_accuracy,
        svm_error,
        svm_precision,
        svm_recall,
        svm_f1
    ]

})

print("\nMODEL PERFORMANCE COMPARISON:")

print(comparison.to_string(index=False))

# Identify the better model using F1 Score
if knn_f1 > svm_f1:

    print("\nBetter Performing Model: KNN")

elif svm_f1 > knn_f1:

    print("\nBetter Performing Model: SVM")

else:

    print("\nBoth models have equal F1 Scores.")


# ==============================================================
# CELL 17: VISUALIZE MODEL PERFORMANCE
#
# Purpose:
# Compare the Accuracy and F1 Score of both algorithms
# using a bar graph.
#
# ==============================================================

models = ["KNN", "SVM"]

accuracies = [knn_accuracy, svm_accuracy]

f1_scores = [knn_f1, svm_f1]

x = np.arange(len(models))

width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    x - width / 2,
    accuracies,
    width,
    label="Accuracy"
)

plt.bar(
    x + width / 2,
    f1_scores,
    width,
    label="F1 Score"
)

plt.xticks(x, models)

plt.ylabel("Score")

plt.title("KNN vs SVM Performance")

plt.legend()

plt.show()


# ==============================================================
# CELL 18: CLASSIFY A NEW EMAIL
#
# Purpose:
# Accept a new email message from the user.
# Predict whether it is Spam or Not Spam
# using both trained models.
#
# ==============================================================

new_email = input("\nEnter an email message to classify: ")

if new_email.strip() != "":

    # Convert new email into TF-IDF features
    new_email_tfidf = tfidf.transform([new_email])

    # KNN prediction
    knn_result = knn.predict(new_email_tfidf)[0]

    # SVM prediction
    svm_result = svm.predict(new_email_tfidf)[0]

    # Display KNN result
    print("\nKNN Prediction:")

    if knn_result == 1:
        print("Abnormal State: SPAM")
    else:
        print("Normal State: NOT SPAM")

    # Display SVM result
    print("\nSVM Prediction:")

    if svm_result == 1:
        print("Abnormal State: SPAM")
    else:
        print("Normal State: NOT SPAM")

else:

    print("\nNo new email entered. Prediction skipped.")


# ==============================================================
# IMPORTANT THEORY AND EXTERNAL VIVA QUESTIONS
# ==============================================================

# 1. What is Email Spam Detection?
# It is a classification task that identifies whether
# an email is unwanted (Spam) or legitimate (Not Spam).

# 2. What are the two states of email classification?
#
# Normal State:
# Not Spam (Ham) = 0
#
# Abnormal State:
# Spam = 1

# 3. What is Binary Classification?
# A Machine Learning task in which data is classified
# into exactly two categories.

# 4. What is Machine Learning?
# It enables computers to learn patterns from data
# and make predictions without explicitly programming
# every rule.

# 5. What is Supervised Learning?
# Learning using labelled data.
# In this practical, emails are labelled Spam or Not Spam.

# 6. What is KNN?
# K-Nearest Neighbors is a supervised learning algorithm
# that classifies data based on the majority class
# of its K nearest neighbors.

# 7. What is SVM?
# Support Vector Machine is a supervised learning
# algorithm that finds an optimal decision boundary
# to separate different classes.

# 8. What is a Hyperplane?
# A decision boundary that separates classes
# in a feature space.

# 9. What are Support Vectors?
# Data points closest to the decision boundary.
# They help determine the position of the hyperplane.

# 10. What is the Margin in SVM?
# The distance between the decision boundary
# and the nearest data points from each class.
# SVM tries to maximize this margin.

# 11. What is Linear SVM?
# An SVM model that uses a linear decision boundary
# to separate different classes.

# 12. Why is SVM useful for spam detection?
# Email text can be represented as high-dimensional
# numerical vectors. SVM can classify these vectors
# using an effective separating boundary.

# 13. What is the difference between KNN and SVM?
#
# KNN:
# - Distance-based algorithm.
# - Uses nearest neighbors.
# - Lazy learning algorithm.
# - Prediction can be slow for large datasets.
#
# SVM:
# - Margin-based algorithm.
# - Learns a separating decision boundary.
# - Uses support vectors.
# - Can work effectively with high-dimensional text data.

# 14. What is TF-IDF?
# Term Frequency - Inverse Document Frequency.
# It converts text into numerical vectors by measuring
# the importance of words in documents.

# 15. Why is TF-IDF used?
# Machine Learning algorithms cannot directly process
# ordinary text.
# TF-IDF converts email messages into numerical features.

# 16. What is Term Frequency (TF)?
# It measures how frequently a word appears
# in a particular document.

# 17. What is Inverse Document Frequency (IDF)?
# It reduces the importance of words that appear
# in many documents and gives more importance
# to relatively rare words.

# 18. What are Stop Words?
# Common words such as "the", "is", "and", "a".
# These words are often removed because they
# may provide little useful classification information.

# 19. What is Tokenization?
# Dividing text into smaller units such as words or tokens.

# 20. What is Text Preprocessing?
# Cleaning and transforming text before applying
# Machine Learning algorithms.

# 21. What is Feature Extraction?
# Converting raw email text into numerical
# features that can be used by an ML model.

# 22. What is Cosine Distance?
# It measures the difference in direction between
# two vectors.
# It is useful for comparing text documents represented
# using TF-IDF features.

# 23. Why is Cosine Distance used for KNN?
# Text documents may differ in length.
# Cosine distance focuses on vector direction
# rather than only the magnitude of the vectors.

# 24. What is the role of K?
# K determines the number of nearest data points
# considered for classification.

# 25. What happens if K is too small?
# The model may become sensitive to noise and overfit.

# 26. What happens if K is too large?
# The model may include unrelated neighbors
# and underfit the data.

# 27. What is Majority Voting?
# The class receiving the maximum number of votes
# among the K nearest neighbors is selected.

# 28. What is the role of C in SVM?
# C is the regularization parameter.
# A larger C penalizes classification errors more strongly.
# A smaller C allows a wider margin with more
# training errors.

# 29. What is a Kernel in SVM?
# A kernel helps SVM model decision boundaries
# in different feature spaces.

# 30. What are common SVM kernels?
# - Linear
# - Polynomial
# - Radial Basis Function (RBF)
# - Sigmoid

# 31. Which kernel is used in this practical?
# Linear kernel through LinearSVC.
# It is suitable for high-dimensional text classification.

# 32. What is Accuracy?
# The percentage of correctly classified emails.
#
# Accuracy = (TP + TN) / Total Predictions

# 33. What is Precision?
# Out of all emails predicted as Spam,
# how many are actually Spam?
#
# Precision = TP / (TP + FP)

# 34. What is Recall?
# Out of all actual Spam emails,
# how many are correctly identified?
#
# Recall = TP / (TP + FN)

# 35. What is F1 Score?
# It is the harmonic mean of Precision and Recall.
# It balances both metrics.

# 36. What is Error Rate?
# The percentage of incorrectly classified emails.
#
# Error Rate = (FP + FN) / Total Predictions
# Error Rate = 1 - Accuracy

# 37. What is a Confusion Matrix?
# A table used to compare actual and predicted classes.
#
# TN = Not Spam correctly classified.
# FP = Not Spam incorrectly classified as Spam.
# FN = Spam incorrectly classified as Not Spam.
# TP = Spam correctly classified.

# 38. What is a False Positive in Spam Detection?
# A legitimate email is incorrectly classified as Spam.
# This may cause an important email to be missed.

# 39. What is a False Negative in Spam Detection?
# A Spam email is incorrectly classified as Not Spam.
# This allows unwanted email into the inbox.

# 40. Which metric is important for Spam Detection?
# Precision is important to avoid marking legitimate
# messages as Spam.
# Recall is important to detect as many Spam messages
# as possible.
# The best choice depends on application requirements.

# 41. What is Overfitting?
# When the model performs well on training data
# but poorly on unseen testing data.

# 42. What is Underfitting?
# When the model fails to learn the important patterns
# in the training dataset.

# 43. What is the difference between Training and Testing?
# Training data is used to build the model.
# Testing data evaluates its performance on unseen emails.

# 44. Why do we use an 80:20 split?
# 80% of records are used for training
# and 20% for testing.
# It is a commonly used experimental split.

# 45. What is random_state=42?
# It makes the train-test split reproducible.

# 46. What is ngram_range=(1, 2)?
# It extracts both single words (unigrams)
# and consecutive two-word combinations (bigrams).

# Example:
# "free offer"
# Unigrams: free, offer
# Bigram: free offer

# 47. What is max_features=5000?
# It limits the TF-IDF vocabulary to 5000 features.

# 48. What is weights="distance" in KNN?
# It gives closer neighbors more influence
# than farther neighbors when voting.

# 49. What is the main advantage of KNN?
# It is simple, intuitive and does not require
# a complex model-building process.

# 50. What is the main advantage of SVM?
# It can perform effectively in high-dimensional
# feature spaces and is suitable for text classification.

# 51. What is the main disadvantage of KNN?
# Prediction may become slow when the training
# dataset contains many records.

# 52. What is the main disadvantage of SVM?
# Model training and parameter selection can become
# computationally expensive on very large datasets.

# 53. Why do we compare two algorithms?
# To determine which model provides better
# classification performance on the same test data.

# 54. Which model is better?
# The model with better Precision, Recall and F1 Score
# may be preferred depending on application requirements.
# There is no guaranteed winner before evaluation.

# 55. Is Spam Detection an NLP problem?
# Yes. It uses Natural Language Processing techniques
# to extract useful information from email text.

# 56. What is NLP?
# Natural Language Processing is a branch of AI
# that enables computers to process and understand
# human language.

# 57. Why is the same TF-IDF vectorizer used for
# training and testing?
# Both datasets must have the same feature vocabulary
# and feature ordering.
# The vocabulary is learned from training data only.

# 58. What is Data Leakage?
# It happens when information from testing data
# improperly influences model training.
# TF-IDF is fitted only on training emails in this practical.

# ==============================================================
# COMPLEXITY ANALYSIS
# ==============================================================

# Let:
# n = Number of training emails.
# d = Number of TF-IDF features.
# k = Number of neighbors.

# 1. TF-IDF FEATURE EXTRACTION:
#
# Complexity depends on the number of words/tokens
# in the email dataset and the vocabulary size.
# Sparse matrices are used to reduce memory requirements.

# 2. KNN TRAINING:
#
# Training mainly stores the feature vectors.
# Approximate model-building work is O(1),
# excluding data storage.
#
# KNN PREDICTION:
# Approximately O(n * d) per query for brute-force search.
#
# KNN SPACE COMPLEXITY:
# O(n * d) for storing training feature vectors,
# with the actual sparse storage depending on
# the number of non-zero TF-IDF values.

# 3. SVM TRAINING:
#
# Training complexity depends on the number of samples,
# feature dimensions, optimization method and convergence.
# There is no single exact complexity for all datasets.
#
# Linear SVM is generally efficient for sparse,
# high-dimensional text data.
#
# SVM PREDICTION:
# Approximately O(d) per email for a linear model,
# depending on the number of non-zero model weights.

# 4. EVALUATION:
#
# Confusion Matrix: O(m)
# Accuracy: O(m)
# Precision: O(m)
# Recall: O(m)
# F1 Score: O(m)
#
# m = Number of testing emails.

# 5. SPACE COMPLEXITY:
#
# TF-IDF:
# Depends on vocabulary and number of non-zero entries.
#
# KNN:
# Stores training email feature vectors.
#
# SVM:
# Stores learned model weights.
#
# ==============================================================
# FINAL PRACTICAL SUMMARY
# ==============================================================

# Experiment:
# Email Spam Detection

# Dataset:
# spam.csv

# Learning Type:
# Supervised Machine Learning

# Problem Type:
# Binary Classification

# Classes:
# 0 = Normal / Not Spam
# 1 = Abnormal / Spam

# Text Preprocessing:
# Missing Value Removal,
# Duplicate Removal and Text Cleaning.

# Feature Extraction:
# TF-IDF Vectorization

# Algorithm 1:
# K-Nearest Neighbors (KNN)

# Algorithm 2:
# Linear Support Vector Machine (SVM)

# Distance Metric:
# Cosine Distance for KNN

# SVM:
# Linear Decision Boundary

# Data Splitting:
# 80% Training
# 20% Testing

# Evaluation Metrics:
# Accuracy, Error Rate, Precision,
# Recall, F1 Score and Confusion Matrix.

# Visualizations:
# KNN Confusion Matrix,
# SVM Confusion Matrix,
# Performance Comparison Bar Graph.

# Final Output:
# Spam or Not Spam classification using both models.

# IMPORTANT EXTERNAL VIVA POINT:
# KNN classifies emails using the nearest training
# email vectors, whereas SVM learns an optimal
# separating hyperplane between Spam and Not Spam.
#
# TF-IDF converts email text into numerical features
# before supplying it to either algorithm.
#
# Both models are evaluated using the same
# training and testing split for a fair comparison.
#
# ==============================================================
# END OF PRACTICAL
# ==============================================================
