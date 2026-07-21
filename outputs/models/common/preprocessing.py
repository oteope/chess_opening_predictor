#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocessing.py

Common preprocessing pipeline for all Machine Learning models.

Returns
-------
X_train
X_test
y_train
y_test
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_dataset(
    dataset_path,
    test_size=0.2,
    random_state=42,
    scale=False
):
    """
    Loads and preprocesses the dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the processed dataset.

    test_size : float
        Percentage used for testing.

    random_state : int
        Random seed.

    scale : bool
        Whether to apply StandardScaler.
        False -> Random Forest
        True  -> Neural Networks
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
    # Features / Target
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

    # --------------------------------------------------
    # Optional scaling
    # --------------------------------------------------

    if scale:

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)

        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test, scaler

    return X_train, X_test, y_train, y_test