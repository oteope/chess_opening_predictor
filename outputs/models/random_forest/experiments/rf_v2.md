# Random Forest v2

## Date

2026-07-20

## Hyperparameters

| Parameter | Value |
|-----------|------:|
| n_estimators | 100 |
| max_depth | 20 |
| max_features | sqrt (default) |
| min_samples_leaf | 1 (default) |
| random_state | 42 |
| n_jobs | -1 |

## Results
============================================================
Random Forest Results
============================================================

Accuracy: 0.6834

Classification Report

              precision    recall  f1-score   support

           0       0.75      0.01      0.01       411
           1       0.68      0.76      0.72      3263
           2       0.69      0.69      0.69      3035

    accuracy                           0.68      6709
   macro avg       0.71      0.49      0.47      6709
weighted avg       0.69      0.68      0.66      6709


Confusion Matrix

[[   3  237  171]
 [   1 2491  771]
 [   0  944 2091]]