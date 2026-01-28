from __future__ import annotations
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def classification_metrics(y_true, y_prob):
    y_true = (y_true).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    bs = brier_score_loss(y_true, y_prob)
    return {"auc": auc, "average_precision": ap, "brier": bs}
