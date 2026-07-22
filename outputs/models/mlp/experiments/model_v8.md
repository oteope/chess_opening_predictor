# Experiment v8 — Larger Architecture

## Objective
Increase the capacity of the neural network by adding more neurons and an additional hidden layer.

---

## Changes

Architecture changed from:

774 → 512 → 256 → 3

to

774 → 1024 → 512 → 256 → 3

Batch Normalization and ReLU were kept after every hidden layer.

---

## Results

============================================================
MLP Results
============================================================

Accuracy: 0.6614

Classification Report

              precision    recall  f1-score   support

           0       0.19      0.16      0.17       411
           1       0.68      0.74      0.71      3263
           2       0.69      0.65      0.67      3035

    accuracy                           0.66      6709
   macro avg       0.52      0.51      0.52      6709
weighted avg       0.66      0.66      0.66      6709


Confusion Matrix

[[  65  201  145]
 [ 134 2405  724]
 [ 141  927 1967]]

## Conclusion

Increasing the network capacity did not improve performance.

The model converged correctly, but test accuracy remained very similar to previous experiments.

This suggests the previous architecture already had enough capacity and that the main limitation is likely the available feature set rather than model size.

The larger network also introduces more parameters, increasing computational cost without improving generalization.