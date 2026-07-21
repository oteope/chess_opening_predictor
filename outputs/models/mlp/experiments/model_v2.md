# MLP v2

## Date

2026-07-20

## Architecture

771 → 512 → 256 → 3

ReLU

Dropout = 0.3

## Hyperparameters

| Parameter | Value |
|-----------|------:|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 256 |
| Epochs | 50 |
| Loss | CrossEntropyLoss |
| Class Weight | balanced |

============================================================
MLP Results
============================================================

Accuracy: 0.6019

Classification Report

              precision    recall  f1-score   support

           0       0.12      0.37      0.18       411
           1       0.74      0.59      0.66      3263
           2       0.69      0.64      0.67      3035

    accuracy                           0.60      6709
   macro avg       0.52      0.54      0.50      6709
weighted avg       0.68      0.60      0.63      6709


Confusion Matrix

[[ 152  130  129]
 [ 572 1930  761]
 [ 516  563 1956]]

## Notes

- Added class weighting to mitigate dataset imbalance.
- Draw recall increased from 0.00 to 0.37.
- Overall accuracy decreased due to a higher number of draw predictions.
- The network now recognizes draws but overestimates this class, reducing performance on White Wins and Black Wins.