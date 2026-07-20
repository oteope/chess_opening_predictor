# Random Forest v3

## Date

2026-07-20

## Hyperparameters

| Parameter | Value |
|-----------|------:|
| n_estimators | 100 |
| max_depth | 15 |
| max_features | 350 |
| min_samples_leaf | 1 (default) |
| random_state | 42 |
| n_jobs | -1 |

## Results
============================================================
Random Forest Results
============================================================

Accuracy: 0.6806

Classification Report

              precision    recall  f1-score   support

           0       0.34      0.03      0.05       411
           1       0.68      0.75      0.72      3263
           2       0.69      0.69      0.69      3035

    accuracy                           0.68      6709
   macro avg       0.57      0.49      0.48      6709
weighted avg       0.66      0.68      0.66      6709


Confusion Matrix

[[  11  227  173]
 [  11 2463  789]
 [  10  933 2092]]

 ## Notes

- Overall performance slightly decreased compared to the previous experiments.
- Limiting the trees to **350 features** and **max_depth = 15** did not improve generalization.
- The model's ability to predict **draws** remains extremely poor (3% recall).
- Precision for the **White Win** class dropped noticeably compared to previous configurations, suggesting that this hyperparameter combination negatively affected the discrimination of White victories.