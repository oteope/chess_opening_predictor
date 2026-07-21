# Experiment V5

## Objective

Increase the capacity of the neural network by adding one extra hidden layer.

---

## Changes

Architecture changed from:

771 → 512 → 256 → 3

to

771 → 512 → 256 → 128 → 3

ReLU activation after every hidden layer.

Dropout kept at 0.30.

Optimizer:
- Adam
- Learning rate = 0.001

Epochs:
- 50

Loss:
- CrossEntropyLoss

---

## Results

 
============================================================
MLP Results
============================================================

Accuracy: 0.6694

Classification Report
 
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       411
           1       0.66      0.77      0.71      3263
           2       0.68      0.65      0.67      3035

    accuracy                           0.67      6709
   macro avg       0.45      0.47      0.46      6709
weighted avg       0.63      0.67      0.65      6709


Confusion Matrix

[[   0  250  161]
 [   0 2515  748]
 [   0 1059 1976]]

## Conclusion

Adding an additional hidden layer did not improve performance.

The model still completely ignores class 0.

The architecture does not appear to be the limiting factor.

Future experiments should focus on:
- better feature engineering
- different loss functions
- resampling techniques
- alternative algorithms