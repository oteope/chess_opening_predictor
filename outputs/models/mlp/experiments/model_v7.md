# Experiment V7 — Learning Rate Scheduler

## Goal

Evaluate whether decreasing the learning rate during training improves convergence and generalization.

---

## Changes

Added a StepLR scheduler.

Configuration:

- Initial Learning Rate: 0.001
- Step Size: 10 epochs
- Gamma: 0.5

Learning rate schedule:

Epoch 1–10   → 0.001

Epoch 11–20  → 0.0005

Epoch 21–30  → 0.00025

Epoch 31–40  → 0.000125

Epoch 41–50  → 0.0000625

---

## Results

============================================================
MLP Results
============================================================

Accuracy: 0.6630

Classification Report

              precision    recall  f1-score   support

           0       0.21      0.17      0.19       411
           1       0.68      0.74      0.71      3263
           2       0.70      0.64      0.67      3035

    accuracy                           0.66      6709
   macro avg       0.53      0.52      0.52      6709
weighted avg       0.66      0.66      0.66      6709


Confusion Matrix

[[  71  206  134]
 [ 116 2424  723]
 [ 146  936 1953]]

## Conclusion

The scheduler reduced the training loss from approximately 0.90 to 0.89, showing a smoother optimization process.

However, the reduction in loss did not translate into better test accuracy.

Compared with previous experiments, performance remained very similar while requiring additional training complexity.

The learning rate schedule alone is insufficient to improve the current architecture.

Future improvements should focus on increasing model capacity instead of further optimizer tuning.