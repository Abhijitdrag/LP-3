
# ==========================================================
# EMAIL SPAM DETECTION USING KNN AND SVM
# ==========================================================

# CELL 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# CELL 2: LOAD DATASET

df = pd.read_csv("emails.csv")

print("First Five Records:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
print(df.info())


# CELL 3: DATA PREPROCESSING

# Remove duplicate and missing records
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Separate features and target
X = df.drop(columns=["Email No.", "Prediction"])
y = df["Prediction"].astype(int)

print("\nClass Distribution:")
print(y.value_counts())

print("\n0 = Normal (Not Spam)")
print("1 = Abnormal (Spam)")


# CELL 4: SPLIT DATASET

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# CELL 5: IMPLEMENT KNN

knn = KNeighborsClassifier(
    n_neighbors=5,
    metric="cosine",
    algorithm="brute",
    weights="distance"
)

knn.fit(X_train, y_train)

knn_predictions = knn.predict(X_test)

print("\nKNN Model Trained Successfully!")


# CELL 6: IMPLEMENT SVM

svm = LinearSVC(
    C=1.0,
    max_iter=5000,
    random_state=42
)

svm.fit(X_train, y_train)

svm_predictions = svm.predict(X_test)

print("\nSVM Model Trained Successfully!")


# CELL 7: EVALUATE KNN

knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions)
knn_recall = recall_score(y_test, knn_predictions)
knn_f1 = f1_score(y_test, knn_predictions)
knn_error = 1 - knn_accuracy

print("\n========== KNN PERFORMANCE ==========")
print("Accuracy:", round(knn_accuracy * 100, 2), "%")
print("Error Rate:", round(knn_error * 100, 2), "%")
print("Precision:", round(knn_precision, 4))
print("Recall:", round(knn_recall, 4))
print("F1 Score:", round(knn_f1, 4))

print("\nKNN Classification Report:")
print(classification_report(y_test, knn_predictions))


# CELL 8: KNN CONFUSION MATRIX

knn_cm = confusion_matrix(
    y_test, knn_predictions, labels=[0, 1]
)

print("\nKNN Confusion Matrix:")
print(knn_cm)

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
plt.show()


# CELL 9: EVALUATE SVM

svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions)
svm_recall = recall_score(y_test, svm_predictions)
svm_f1 = f1_score(y_test, svm_predictions)
svm_error = 1 - svm_accuracy

print("\n========== SVM PERFORMANCE ==========")
print("Accuracy:", round(svm_accuracy * 100, 2), "%")
print("Error Rate:", round(svm_error * 100, 2), "%")
print("Precision:", round(svm_precision, 4))
print("Recall:", round(svm_recall, 4))
print("F1 Score:", round(svm_f1, 4))

print("\nSVM Classification Report:")
print(classification_report(y_test, svm_predictions))


# CELL 10: SVM CONFUSION MATRIX

svm_cm = confusion_matrix(
    y_test, svm_predictions, labels=[0, 1]
)

print("\nSVM Confusion Matrix:")
print(svm_cm)

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
plt.show()


# CELL 11: MODEL COMPARISON

comparison = pd.DataFrame({
    "Metric": [
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

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

if svm_f1 > knn_f1:
    print("\nBetter Performing Model: SVM")
else:
    print("\nBetter Performing Model: KNN")


# CELL 12: PERFORMANCE COMPARISON GRAPH

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
plt.title("KNN vs SVM Performance Comparison")
plt.legend()
plt.show()


# CELL 13: CLASSIFY A NEW EMAIL

def predict_email(email):

    # Convert input email into lowercase tokens
    words = re.findall(r"[a-z0-9']+", email.lower())

    # Count words that exist in the dataset vocabulary
    word_counts = {
        word: words.count(word)
        for word in set(words)
        if word in X.columns
    }

    # Create a feature vector in the same column order
    email_vector = pd.DataFrame(
        [[word_counts.get(col, 0) for col in X.columns]],
        columns=X.columns
    )

    # Check whether any known vocabulary words were found
    if email_vector.to_numpy().sum() == 0:
        print("\nWarning: No known dataset words found.")
        print("Prediction may not be meaningful.")
        return

    # Predict using both algorithms
    knn_result = knn.predict(email_vector)[0]
    svm_result = svm.predict(email_vector)[0]

    print("\nKNN Prediction:")
    if knn_result == 1:
        print("Abnormal State: SPAM")
    else:
        print("Normal State: NOT SPAM")

    print("\nSVM Prediction:")
    if svm_result == 1:
        print("Abnormal State: SPAM")
    else:
        print("Normal State: NOT SPAM")


# Enter email
new_email = input("\nEnter an email message to classify: ")

if new_email.strip():
    predict_email(new_email)
else:
    print("Please enter a valid email.")
