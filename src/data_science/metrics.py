# metrics.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


def regression_errors(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    rss = np.sum((y_true - y_pred) ** 2)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - rss / tss if tss > 0 else np.nan
    rse_residual_std_error = np.sqrt(
        rss / (len(y_true) - 2)
    )  # if 1 predictor; adjust as needed
    rse_relative_squared_error = rss / tss if tss > 0 else np.nan  # = 1 - R^2
    rmsle = np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
    mape = (
        np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100
    )
    smape = (
        np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100
    )
    return dict(
        MAE=mae,
        MSE=mse,
        RMSE=rmse,
        RSE_residual=rse_residual_std_error,
        RSE_relative=rse_relative_squared_error,
        RMSLE=rmsle,
        MAPE=mape,
        sMAPE=smape,
        R2=r2,
    )


def mase(y_true, y_pred, y_naive):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_naive = np.asarray(y_naive)
    mae_model = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(y_true - y_naive))
    return mae_model / mae_naive if mae_naive != 0 else np.nan


def classification_metrics(y_true, y_score, threshold: Optional[float] = None):
    """If threshold None, compute threshold-free AUCs. If set, compute confusion-based metrics too.
    y_score is probability for class 1."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    out = {
        "ROC_AUC": roc_auc_score(y_true, y_score),
        "PR_AUC": average_precision_score(y_true, y_score),
        "Brier": brier_score_loss(y_true, y_score),
    }
    if threshold is not None:
        y_pred = (y_score >= threshold).astype(int)
        out.update(
            {
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred, zero_division=0),
                "F1": f1_score(y_true, y_pred, zero_division=0),
                "ConfusionMatrix": confusion_matrix(y_true, y_pred).tolist(),
            }
        )
    return out


def plot_confusion_matrix(y_true, y_pred, labels=("Neg", "Pos")):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig, ax


def plot_roc_pr(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    fig1, ax1 = plt.subplots()
    ax1.plot(fpr, tpr)
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.set_title("ROC Curve")
    fig1.tight_layout()

    fig2, ax2 = plt.subplots()
    ax2.plot(rec, prec)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("PR Curve")
    fig2.tight_layout()
    return (fig1, ax1), (fig2, ax2)
