#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_rf.py

Train a Random Forest classifier for the Chess Opening Predictor.
"""

from outputs.models.common.preprocessing import load_dataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib

def main():

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = load_dataset(
        "data/processed/final_dataset.csv"
    )

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth = None,
        max_features = 250,
        class_weight="balanced"
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    model.fit(X_train, y_train)

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------

    predictions = model.predict(X_test)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(y_test, predictions)

    print("=" * 60)
    print("Random Forest Results")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report\n")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix\n")
    print(confusion_matrix(y_test, predictions))

    joblib.dump(
        model,
        "outputs/models/random_forest/best_model/final_model.pkl"
)


if __name__ == "__main__":
    main()