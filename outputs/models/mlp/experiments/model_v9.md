# Experiment v9 - Deeper Architecture

## Objective

Evaluate whether increasing the network depth improves the model's ability to learn more complex feature representations.

---

## Changes

Architecture changed from:

Input → 512 → 256 → 128 → Output

to:

Input (774) → 512 → 256 → 128 → 64 → Output

Batch Normalization and ReLU were kept after every hidden layer.

---

## Results

### Training Loss

Final Loss: **0.8752**

---

### Test Accuracy

**65.87%**

---

### Classification Report

============================================================
MLP Results
============================================================

Accuracy: 0.6587

Classification Report

              precision    recall  f1-score   support

           0       0.18      0.17      0.18       411
           1       0.69      0.73      0.71      3263
           2       0.69      0.65      0.67      3035

    accuracy                           0.66      6709
   macro avg       0.52      0.52      0.52      6709
weighted avg       0.66      0.66      0.66      6709


Confusion Matrix

[[  71  200  140]
 [ 145 2380  738]
 [ 175  892 1968]]

---

## Conclusion

Adding an extra hidden layer did **not improve** the model.

Compared to previous experiments:

- Training remained stable.
- Loss decreased correctly.
- Accuracy slightly decreased.
- Draw prediction remained weak.
- Win predictions remained almost identical.

The additional capacity did not translate into better generalization.

This suggests that the current limitation is likely the information contained in the features rather than the network depth.