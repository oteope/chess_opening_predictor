# Random Forest v5

## Date

2026-07-20

## Hyperparameters

| Parameter | Value |
|-----------|------:|
| n_estimators | 300 |
| max_depth | None |
| max_features | 250 |
| class_weight | balanced |
| min_samples_leaf | 1 (default) |
| random_state | 42 |
| n_jobs | -1 |

## Results
============================================================
Random Forest Results
============================================================

Accuracy: 0.6798

Classification Report

              precision    recall  f1-score   support

           0       0.27      0.12      0.16       411
           1       0.70      0.72      0.71      3263
           2       0.68      0.71      0.70      3035

    accuracy                           0.68      6709
   macro avg       0.55      0.52      0.52      6709
weighted avg       0.67      0.68      0.67      6709


Confusion Matrix

[[  48  191  172]
 [  77 2364  822]
 [  55  831 2149]]

## Notes

- Introducing `class_weight="balanced"` significantly improved the model's ability to detect **Draws**.
- Draw recall increased from **5%** to **12%**, more than doubling the number of correctly classified draws.
- This improvement came at the cost of a lower overall accuracy (69.00% → 67.98%).
- The classifier sacrifices some performance on White and Black wins in order to reduce the bias toward the majority classes.
- This experiment highlights the trade-off between maximizing overall accuracy and improving minority-class recognition.