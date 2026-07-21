# Random Forest v1

## Date

2026-07-20

---

## Model

Random Forest Classifier

---

## Hyperparameters

| Parameter | Value |
|-----------|------:|
| n_estimators | 100 |
| max_depth | None |
| max_features | sqrt (default) |
| min_samples_leaf | 1 (default) |
| bootstrap | True (default) |
| criterion | gini (default) |
| random_state | 42 |
| n_jobs | -1 |

---

## Dataset

| Metric | Value |
|--------|------:|
| Features | 771 |
| Target | White Win / Draw / Black Win |
| Train/Test Split | 80% / 20% |

---

## Results
============================================================
Random Forest Results
============================================================

Accuracy: 0.6868

Classification Report

              precision    recall  f1-score   support

           0       0.57      0.04      0.07       411
           1       0.69      0.76      0.72      3263
           2       0.69      0.70      0.69      3035

    accuracy                           0.69      6709
   macro avg       0.65      0.50      0.50      6709
weighted avg       0.68      0.69      0.67      6709


Confusion Matrix

[[  16  207  188]
 [   6 2482  775]
 [   6  919 2110]]