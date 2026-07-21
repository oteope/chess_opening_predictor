#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_mlp.py

Train a Multi-Layer Perceptron (PyTorch)
for the Chess Opening Predictor.
"""

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader,
)

from torch.optim import Adam

from outputs.models.common.preprocessing import load_dataset
from outputs.models.mlp.model import ChessMLP


def main():

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = load_dataset(
        "data/processed/final_dataset.csv"
    )

    # --------------------------------------------------
    # Convert DataFrames to PyTorch tensors
    # --------------------------------------------------

    X_train = torch.tensor(
        X_train.values,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test.values,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train.values,
        dtype=torch.long
    )

    y_test = torch.tensor(
        y_test.values,
        dtype=torch.long
    )

    # --------------------------------------------------
    # Create Dataset & DataLoader
    # --------------------------------------------------

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True
    )

    # --------------------------------------------------
    # Create Model
    # --------------------------------------------------

    model = ChessMLP()

    # --------------------------------------------------
    # Loss Function
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = Adam(
        model.parameters(),
        lr=0.001
    )

    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------

    epochs = 100

    for epoch in range(epochs):

        model.train()

        for batch_X, batch_y in train_loader:

            # Forward Pass
            outputs = model(batch_X)

            # Compute Loss
            loss = criterion(outputs, batch_y)

            # Reset Previous Gradients
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # Update Weights
            optimizer.step()

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {loss.item():.4f}"
        )


if __name__ == "__main__":
    main()