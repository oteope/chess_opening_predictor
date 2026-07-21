# MLP v1

## Date

2026-07-20

## Architecture

```
771 → 512 → 256 → 3
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
| Class Weights | None |

---

## Results

Accuracy: **0.6760**

### Classification Report

| Class | Precision | Recall | F1-score |
|------|----------:|-------:|---------:|
| Draw | 0.00 | 0.00 | 0.00 |
| White Win | 0.71 | 0.68 | 0.69 |
| Black Win | 0.65 | 0.77 | 0.70 |

---

## Confusion Matrix

```
[[   0  198  213]
 [   0 2211 1052]
 [   0  711 2324]]
```

---

## Notes

- Baseline MLP architecture.
- The network successfully learned to distinguish White Wins and Black Wins.
- The model completely ignored the Draw class.
- Dataset imbalance appears to strongly influence the optimization process.
- This experiment serves as the baseline for future MLP improvements.