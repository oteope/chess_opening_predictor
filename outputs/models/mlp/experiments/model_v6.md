# Experiment V6 — Batch Normalization

## Objective

Improve training stability by adding Batch Normalization after each Linear layer.

---

## Changes

Added BatchNorm1d after each hidden Linear layer.

Architecture:

Linear
↓
BatchNorm1d
↓
ReLU
↓
Dropout

Applied to both hidden layers.

---

## Results

============================================================
MLP Results
============================================================

Accuracy: 0.6594

Classification Report

              precision    recall  f1-score   support

           0       0.19      0.15      0.17       411
           1       0.66      0.78      0.72      3263
           2       0.71      0.59      0.65      3035

    accuracy                           0.66      6709
   macro avg       0.52      0.51      0.51      6709
weighted avg       0.66      0.66      0.65      6709


Confusion Matrix

[[  60  233  118]
 [  91 2560  612]
 [ 158 1073 1804]]

## Observations

Compared to V5:

• Overall accuracy decreased.

• The model finally started predicting the minority class.

• Recall for class 0 increased from 0.00 to 0.15.

• This came at the cost of lower performance on classes 1 and 2.

Batch Normalization improved representation learning but reduced overall classification accuracy.

---

## Conclusion

Batch Normalization alone is not a net improvement for this dataset.

Future experiments should investigate learning-rate scheduling, deeper architectures, or early stopping.