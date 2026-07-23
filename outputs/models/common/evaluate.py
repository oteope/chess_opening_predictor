#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate.py

Common evaluation metrics for any classification model.
"""

import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from outputs.models.common.preprocessing import load_dataset
from outputs.models.mlp.model import ChessMLP
import torch


def evaluate(model, X_test, y_test):
    """
    Evaluate a PyTorch or Scikit-Learn classifier.
    """

    # -----------------------------
    # PyTorch model
    # -----------------------------
    if isinstance(model, torch.nn.Module):

        model.eval()

        with torch.no_grad():
            outputs = model(X_test)

            predictions = torch.argmax(
                outputs,
                dim=1
            ).numpy()

        y_true = y_test.numpy()

    # -----------------------------
    # Scikit-Learn model
    # -----------------------------
    else:

        predictions = model.predict(X_test)
        y_true = y_test

    # -----------------------------
    # Metrics
    # -----------------------------
    accuracy = accuracy_score(
        y_true,
        predictions
    )

    print("=" * 60)
    print("Model Results")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report\n")

    print(
        classification_report(
            y_true,
            predictions
        )
    )

    print("\nConfusion Matrix\n")

    print(
        confusion_matrix(
            y_true,
            predictions
        )
    )

def main():

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------

    _, X_test, _, y_test = load_dataset(
        "data/processed/final_dataset.csv"
    )

    # --------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------

    X_test = torch.tensor(
        X_test.values,
        dtype=torch.float32
    )

    y_test = torch.tensor(
        y_test.values,
        dtype=torch.long
    )

    # --------------------------------------------------
    # Create Model
    # --------------------------------------------------

    model = ChessMLP()

    # --------------------------------------------------
    # Load Trained Weights
    # --------------------------------------------------

    model.load_state_dict(
        torch.load(
            "outputs/models/mlp/experiments/model_v9.pth",
            weights_only=True
        )
    )

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    evaluate(
        model,
        X_test,
        y_test
    )


if __name__ == "__main__":
    main()