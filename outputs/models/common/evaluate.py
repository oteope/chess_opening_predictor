#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate.py

Evaluate a trained Multi-Layer Perceptron.
"""

import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from outputs.models.common.preprocessing import load_dataset
from outputs.models.mlp.model import ChessMLP


def evaluate(model, X_test, y_test):
    """
    Evaluate a trained model.
    """

    model.eval()

    with torch.no_grad():

        outputs = model(X_test)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

    predictions = predictions.numpy()
    y_test = y_test.numpy()

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("=" * 60)
    print("MLP Results")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report\n")
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print("\nConfusion Matrix\n")
    print(
        confusion_matrix(
            y_test,
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
            "outputs/models/mlp/experiments/model_v4.pth",
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