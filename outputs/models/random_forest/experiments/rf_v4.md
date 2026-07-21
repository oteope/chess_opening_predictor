# Random Forest v4

## Date

2026-07-20

## Hyperparameters

| Parameter | Value |
|-----------|------:|
| n_estimators | 300 |
| max_depth | None |
| max_features | 250 |
| min_samples_leaf | 1 (default) |
| random_state | 42 |
| n_jobs | -1 |

## Results
============================================================
Random Forest Results
============================================================

Accuracy: 0.6900

Classification Report

              precision    recall  f1-score   support

           0       0.46      0.05      0.09       411
           1       0.69      0.76      0.72      3263
           2       0.69      0.70      0.70      3035

    accuracy                           0.69      6709
   macro avg       0.61      0.50      0.50      6709
weighted avg       0.68      0.69      0.67      6709


Confusion Matrix

[[  21  210  180]
 [  14 2476  773]
 [  11  892 2132]]

 ## Notes

- Best Random Forest configuration evaluated so far.
- Increasing the number of trees from **100** to **300** and limiting each split to **250 randomly selected features** slightly improved overall accuracy.
- The model continues to distinguish **White Wins** and **Black Wins** reasonably well.
- Prediction of **Draws** remains the main limitation, with only **5% recall**.
- Despite the overall improvement, the model still struggles to correctly identify draw positions, suggesting that the current feature representation may not contain enough information for this class.