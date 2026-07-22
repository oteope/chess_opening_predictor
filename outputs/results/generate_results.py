#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_results.py

Final script that loads the two trained models (Random Forest & MLP),
evaluates them on the same test split, and produces every table, figure,
and markdown report required for the research question.

Usage
-----
    python outputs/results/generate_results.py
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# Setup paths so that imports from the outputs/ package work correctly
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent                     # outputs/results
OUTPUTS_DIR = SCRIPT_DIR.parent                                 # outputs
PROJECT_ROOT = OUTPUTS_DIR.parent                               # parent of outputs

if str(OUTPUTS_DIR) not in sys.path:
    sys.path.insert(0, str(OUTPUTS_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.common.preprocessing import load_dataset

# ---------------------------------------------------------------------------
# Paths to trained models and processed dataset
# ---------------------------------------------------------------------------
DATASET_PATH = str(PROJECT_ROOT / "data" / "processed" / "final_dataset.csv")
RF_MODEL_PATH = str(PROJECT_ROOT / "outputs" / "models" / "random_forest" / "best_model" / "final_model.pkl")
MLP_MODEL_PATH = str(PROJECT_ROOT / "outputs" / "models" / "mlp" / "best_model" / "model_v9.pth")
RESULTS_DIR = SCRIPT_DIR

FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
REPORT_DIR = RESULTS_DIR / "report"

# ---------------------------------------------------------------------------
# Helper: ensure directories exist
# ---------------------------------------------------------------------------
for d in [FIGURES_DIR, TABLES_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load dataset (reuse existing preprocessing)
# ---------------------------------------------------------------------------
print("[info] Loading dataset (Random Forest)")
X_train_rf, X_test_rf, y_train_rf, y_test_rf = load_dataset(
    DATASET_PATH, scale=False
)

print("[info] Loading dataset (MLP)")
X_train_mlp, X_test_mlp, y_train_mlp, y_test_mlp, scaler = load_dataset(
    DATASET_PATH, scale=True
)

feature_names = X_train_rf.columns.tolist()

# ---------------------------------------------------------------------------
# 2. Load Random Forest model
# ---------------------------------------------------------------------------
print("[info] Loading Random Forest model")
rf_model = joblib.load(RF_MODEL_PATH)

# ---------------------------------------------------------------------------
# 3. Load MLP model
# ---------------------------------------------------------------------------
print("[info] Loading MLP model")

# Attempt to import the model class; assume it exists in the project
try:
    from models.mlp.model import ChessMLP
except ImportError:
    # fallback: define a minimal wrapper (should not happen)
    from torch import nn
    class ChessMLP(nn.Module):
        def __init__(self, input_size=774):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 3),
            )
        def forward(self, x):
            return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load(MLP_MODEL_PATH, map_location=device)

# Determine input_size from state_dict keys
input_size = state_dict["network.0.weight"].shape[1]
mlp_model = ChessMLP(input_size=input_size).to(device)
mlp_model.load_state_dict(state_dict)
mlp_model.eval()

# ---------------------------------------------------------------------------
# 4. Helper to compute metrics from predictions
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    """Return dictionary with accuracy, precision, recall, f1 (weighted)."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {"Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4)}

# ---------------------------------------------------------------------------
# 5. Evaluate both models
# ---------------------------------------------------------------------------
print("[info] Evaluating Random Forest")
y_pred_rf = rf_model.predict(X_test_rf)
rf_metrics = compute_metrics(y_test_rf, y_pred_rf)
print("    RF:", rf_metrics)

print("[info] Evaluating MLP")
X_test_mlp_tensor = torch.tensor(X_test_mlp, dtype=torch.float32).to(device)
with torch.no_grad():
    logits = mlp_model(X_test_mlp_tensor)
    y_pred_mlp = torch.argmax(logits, dim=1).cpu().numpy()
mlp_metrics = compute_metrics(y_test_mlp, y_pred_mlp)
print("    MLP:", mlp_metrics)

# ---------------------------------------------------------------------------
# 6. Save comparison table
# ---------------------------------------------------------------------------
comparison_rows = [
    ["Random Forest"] + list(rf_metrics.values()),
    ["MLP"] + list(mlp_metrics.values()),
]
comparison_df = pd.DataFrame(
    comparison_rows,
    columns=["Model"] + list(rf_metrics.keys()),
)
comparison_df.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
print("[info] Saved table: model_comparison.csv")

# ---------------------------------------------------------------------------
# 7. Confusion matrices (matplotlib only)
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    # Place text inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

plot_confusion_matrix(y_test_rf, y_pred_rf,
                      "Confusion Matrix - Random Forest",
                      FIGURES_DIR / "rf_confusion_matrix.png")
plot_confusion_matrix(y_test_mlp, y_pred_mlp,
                      "Confusion Matrix - MLP",
                      FIGURES_DIR / "mlp_confusion_matrix.png")
print("[info] Saved confusion matrices")

# ---------------------------------------------------------------------------
# 8. Feature importance (Random Forest)
# ---------------------------------------------------------------------------
print("[info] Computing feature importance")
importances = rf_model.feature_importances_
indices = np.argsort(importances)[-20:][::-1]   # top 20 descending

top_20_names = [feature_names[i] for i in indices]
top_20_vals = importances[indices]

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(top_20_names))
ax.barh(y_pos, top_20_vals, align="center")
ax.set_yticks(y_pos)
ax.set_yticklabels(top_20_names)
ax.invert_yaxis()  # largest on top
ax.set_xlabel("Importance score", fontsize=12)
ax.set_title("Top 20 Features – Random Forest", fontsize=14)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
plt.close(fig)
print("[info] Saved feature_importance.png")

# ---------------------------------------------------------------------------
# 9. Accuracy comparison bar chart
# ---------------------------------------------------------------------------
models_names = ["Random Forest", "MLP"]
accuracies = [rf_metrics["Accuracy"], mlp_metrics["Accuracy"]]
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(models_names, accuracies, color=["steelblue", "darkorange"])
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy Comparison", fontsize=14)
ax.set_ylim(0, 1)
for i, acc in enumerate(accuracies):
    ax.text(i, acc + 0.015, f"{acc:.3f}", ha="center", fontsize=10)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "accuracy_comparison.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 10-12. Per-class precision / recall / f1 comparison
# ---------------------------------------------------------------------------
def per_class_values(y_true, y_pred, metric_name):
    if metric_name == "precision":
        arr = precision_score(y_true, y_pred, average=None, zero_division=0)
    elif metric_name == "recall":
        arr = recall_score(y_true, y_pred, average=None, zero_division=0)
    elif metric_name == "f1":
        arr = f1_score(y_true, y_pred, average=None, zero_division=0)
    else:
        raise ValueError("Unknown metric_name")
    return arr

classes = sorted(set(y_test_rf))
class_labels = [f"Class {c}" for c in classes]

rf_prec = per_class_values(y_test_rf, y_pred_rf, "precision")
mlp_prec = per_class_values(y_test_mlp, y_pred_mlp, "precision")
rf_rec = per_class_values(y_test_rf, y_pred_rf, "recall")
mlp_rec = per_class_values(y_test_mlp, y_pred_mlp, "recall")
rf_f1 = per_class_values(y_test_rf, y_pred_rf, "f1")
mlp_f1 = per_class_values(y_test_mlp, y_pred_mlp, "f1")

def plot_per_class(values_rf, values_mlp, class_labels, metric_name, save_path):
    x = np.arange(len(class_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width / 2, values_rf, width, label="Random Forest")
    bars2 = ax.bar(x + width / 2, values_mlp, width, label="MLP")
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(f"{metric_name.capitalize()} per Class", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

plot_per_class(rf_prec, mlp_prec, class_labels, "precision",
               FIGURES_DIR / "precision_per_class.png")
plot_per_class(rf_rec, mlp_rec, class_labels, "recall",
               FIGURES_DIR / "recall_per_class.png")
plot_per_class(rf_f1, mlp_f1, class_labels, "f1",
               FIGURES_DIR / "f1_per_class.png")
print("[info] Saved precision/recall/f1 per‑class figures")

# ---------------------------------------------------------------------------
# 13. Generate markdown report
# ---------------------------------------------------------------------------
def _fmt(d):
    """Format a metric dictionary to a table row."""
    return "| {Model} | {Accuracy} | {Precision} | {Recall} | {F1} |".format(
        Model=f"{'RF':>8}",
        Accuracy=f"{d['Accuracy']:.4f}",
        Precision=f"{d['Precision']:.4f}",
        Recall=f"{d['Recall']:.4f}",
        F1=f"{d['F1']:.4f}",
    )

report_lines = []
report_lines.append("# Results\n")
report_lines.append("## Random Forest\n")
report_lines.append("| Model | Accuracy | Precision | Recall | F1 |")
report_lines.append("|------|----------|-----------|--------|-----|")
report_lines.append(_fmt(rf_metrics))
report_lines.append("")
report_lines.append("## MLP\n")
report_lines.append("| Model | Accuracy | Precision | Recall | F1 |")
report_lines.append("|------|----------|-----------|--------|-----|")
report_lines.append(_fmt(mlp_metrics))
report_lines.append("")
report_lines.append("## Comparison\n")
report_lines.append("| Model | Accuracy | Precision | Recall | F1 |")
report_lines.append("|------|----------|-----------|--------|-----|")
report_lines.append(_fmt(rf_metrics))
report_lines.append(_fmt(mlp_metrics))
report_lines.append("")
report_lines.append("## Research Question\n")
report_lines.append(
    "This project aims to investigate whether the opening phase of a chess game "
    "contains predictive information about the final outcome. "
    "The experimental results presented above will be analyzed and discussed "
    "in the accompanying report."
)
report_lines.append("")

report_path = REPORT_DIR / "results.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print("[info] Generated markdown report: results.md")

print("=" * 60)
print("All evaluation artifacts have been created successfully.")
print("=" * 60)
