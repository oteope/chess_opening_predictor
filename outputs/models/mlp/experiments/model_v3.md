# MLP v3

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
| Class Weights | [2.0, 1.0, 1.0] |

---

## Results

Accuracy: **0.6739**

### Classification Report

| Class | Precision | Recall | F1-score |
|------|----------:|-------:|---------:|
| Draw | 0.00 | 0.00 | 0.00 |
| White Win | 0.70 | 0.69 | 0.70 |
| Black Win | 0.65 | 0.75 | 0.70 |

---

## Confusion Matrix

```
[[   0  198  213]
 [   0 2251 1012]
 [   0  765 2270]]
```

---

## Notes

- Tested manually defined class weights instead of automatic balancing.
- The selected weights were not strong enough to encourage the network to learn the Draw class.
- The model reverted to behavior very similar to the baseline.
- Performance remains almost identical to MLP v1, suggesting stronger weighting or different imbalance handling strategies are required.