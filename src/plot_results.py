import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)


def main():
    prediction_dir = "results/predictions"
    figure_dir = "results/figures"

    os.makedirs(figure_dir, exist_ok=True)

    y_true = np.load(os.path.join(prediction_dir, "y_true.npy"))
    baseline_preds = np.load(
        os.path.join(prediction_dir, "baseline_preds.npy")
    )
    baseline_probs = np.load(
        os.path.join(prediction_dir, "baseline_probs.npy")
    )
    snn_preds = np.load(
        os.path.join(prediction_dir, "snn_preds.npy")
    )
    snn_probs = np.load(
        os.path.join(prediction_dir, "snn_probs.npy")
    )

    baseline_cm = confusion_matrix(
        y_true,
        baseline_preds,
        normalize="true",
    )

    snn_cm = confusion_matrix(
        y_true,
        snn_preds,
        normalize="true",
    )

    baseline_fpr, baseline_tpr, _ = roc_curve(
        y_true,
        baseline_probs,
    )
    snn_fpr, snn_tpr, _ = roc_curve(
        y_true,
        snn_probs,
    )

    baseline_auc = roc_auc_score(y_true, baseline_probs)
    snn_auc = roc_auc_score(y_true, snn_probs)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ConfusionMatrixDisplay(
        confusion_matrix=baseline_cm,
        display_labels=["Normal", "Attack"],
    ).plot(
        ax=axes[0, 0],
        cmap="Blues",
        values_format=".2f",
        colorbar=False,
    )

    axes[0, 0].set_title("Confusion Matrix - Baseline MLP")
    axes[0, 0].set_xlabel("Predicted label")
    axes[0, 0].set_ylabel("True label")

    ConfusionMatrixDisplay(
        confusion_matrix=snn_cm,
        display_labels=["Normal", "Attack"],
    ).plot(
        ax=axes[0, 1],
        cmap="Oranges",
        values_format=".2f",
        colorbar=False,
    )

    axes[0, 1].set_title("Confusion Matrix - EdgeThreat-SNN")
    axes[0, 1].set_xlabel("Predicted label")
    axes[0, 1].set_ylabel("True label")

    axes[1, 0].plot(
        baseline_fpr,
        baseline_tpr,
        color="steelblue",
        linewidth=2,
        label=f"Baseline MLP (AUC = {baseline_auc:.3f})",
    )

    axes[1, 0].plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random",
    )

    axes[1, 0].set_title("ROC Curve - Baseline MLP")
    axes[1, 0].set_xlabel("False positive rate")
    axes[1, 0].set_ylabel("True positive rate")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(
        snn_fpr,
        snn_tpr,
        color="darkorange",
        linewidth=2,
        label=f"EdgeThreat-SNN (AUC = {snn_auc:.3f})",
    )

    axes[1, 1].plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random",
    )

    axes[1, 1].set_title("ROC Curve - EdgeThreat-SNN")
    axes[1, 1].set_xlabel("False positive rate")
    axes[1, 1].set_ylabel("True positive rate")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)

    plt.tight_layout()

    output_path = os.path.join(
        figure_dir,
        "model_comparison.png",
    )

    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"Saved corrected figure to: {output_path}")


if __name__ == "__main__":
    main()
