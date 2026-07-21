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
import os
from outputs.models.common.preprocessing import load_dataset
from outputs.models.mlp.model import ChessMLP
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def main():

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = load_dataset(
        "data/processed/final_dataset.csv"
    )

    # --------------------------------------------------
    # Compute class weights
    # --------------------------------------------------

    weights = torch.tensor(
        [4.0, 1.0, 1.0],
        dtype=torch.float32
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

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

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
    epochs = 50
    
    for epoch in range(epochs):

        model.train()

        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:

            outputs = model(batch_X)

            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        print(
        f"Epoch {epoch+1:03d}/{epochs} | Loss: {epoch_loss:.4f}"
        )


# --------------------------------------------------
# Save Model
# --------------------------------------------------
    os.makedirs(
    "outputs/models/mlp/experiments",
    exist_ok=True
)
    torch.save(
    model.state_dict(),
    "outputs/models/mlp/experiments/model_v4.pth"
)   
if __name__ == "__main__":
    main()