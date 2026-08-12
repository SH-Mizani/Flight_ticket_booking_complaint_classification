# Trains multiple classical ML models
#1. Load TF-IDF features 
#2. Train test split
#3. Training methods: Logistic Regression, Linear SVC
#4. Evaluate each with accuracy and F1-score
#5. Save a comparison table

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)

#Paths
features_path = Path("data/processed/features.csv")
results = Path("data/processed")
models_path = Path("models")
models_path.mkdir(parents=True, exist_ok=True)

#Load features
df = pd.read_csv(features_path)
x = df.drop(columns=["Category"])
y = df["Category"]

print(f"Loaded features: {x.shape[0]} rows, {x.shape[1]} columns")
print("\nClass distribution:\n", y.value_counts())

#Train test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 0.2,
    random_state = 42,
    stratify = y,)

#Models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter = 1000,
        class_weight="balanced",
    ),
    "Linear SVM": LinearSVC(
        class_weight="balanced",
        max_iter = 5000,
    ),}

