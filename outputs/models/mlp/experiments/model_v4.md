# MLP v4

## Date

2026-07-20

## Architecture

```
774 → 512 → 256 → 3
```

Activation:
- ReLU

Regularization:
- Dropout = 0.30

---

## Hyperparameters

| Parameter | Value |
|-----------|------:|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 256 |
| Epochs | 50 |
| Loss Function | CrossEntropyLoss |
| Class Weights | [4.0, 1.0, 1.0] |

---

## Results

Accuracy: **0.6727**

### Classification Report

| Class | Precision | Recall | F1-score |
|------|----------:|-------:|---------:|
| Draw | 0.00 | 0.00 | 0.00 |
| White Win | 0.66 | 0.79 | 0.72 |
| Black Win | 0.69 | 0.64 | 0.67 |

---

## Confusion Matrix

```
[[   0  247  164]
 [   0 2564  699]
 [   0 1086 1949]]
```

---

## Notes

- Increased the manual class weight for the Draw class from **2.0** to **4.0**.
- The model still failed to predict any Draw positions.
- Increasing the draw weight shifted the decision boundary toward predicting more White Wins.
- White Win recall improved noticeably (79%), while Black Win recall decreased.
- Results suggest that manual class weighting alone is insufficient to solve the class imbalance problem.