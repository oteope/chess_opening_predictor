#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocessing.py

Loads the final dataset and prepares it for Machine Learning.

Returns:
    X_train
    X_test
    y_train
    y_test
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_dataset(
    dataset_path,
    test_size=0.2,
    random_state=42
):
    """
    Loads the dataset and prepares it for training.
    """

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    df = pd.read_csv(dataset_path)

    # --------------------------------------------------
    # Encode categorical columns
    # --------------------------------------------------

    opening_encoder = LabelEncoder()
    eco_encoder = LabelEncoder()
    source_encoder = LabelEncoder()

    df["Opening_Name"] = opening_encoder.fit_transform(df["Opening_Name"])
    df["ECO_Code"] = eco_encoder.fit_transform(df["ECO_Code"])
    df["Source"] = source_encoder.fit_transform(df["Source"])

    # --------------------------------------------------
    # Features and target
    # --------------------------------------------------

    X = df.drop(columns=["Result"])

    y = df["Result"]

    # --------------------------------------------------
    # Train / Test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test