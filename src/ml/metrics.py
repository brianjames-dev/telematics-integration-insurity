from __future__ import annotations
import numpy as np

def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Robust ROC AUC:
      AUC = ( P(score_pos > score_neg) + 0.5 * P(score_pos == score_neg) )
    Pairwise O(n_pos * n_neg) — fine for our small test folds.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]

    # Broadcast to compare all positives to all negatives
    diff = pos_scores[:, None] - neg_scores[None, :]
    gt = (diff > 0).sum()
    eq = (diff == 0).sum()

    auc = (gt + 0.5 * eq) / (n_pos * n_neg)
    return float(auc)

def decile_lift(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n = len(y_true)
    if n == 0:
        return {"base_rate": 0.0, "lifts": [0.0]*10}
    idx = np.argsort(-y_score)
    cuts = [int(n * k / 10) for k in range(11)]
    base_rate = float(y_true.mean())
    lifts = []
    for d in range(10):
        sl = slice(cuts[d], cuts[d+1] if d < 9 else None)
        denom = cuts[d+1] - cuts[d]
        rate = float(y_true[idx][sl].mean()) if denom > 0 else 0.0
        lifts.append(rate / base_rate if base_rate > 0 else 0.0)
    return {"base_rate": base_rate, "lifts": lifts}

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_pred - y_true)))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    yt = np.maximum(y_true, 1e-6)
    return float(np.mean(np.abs((y_pred - yt) / yt)))
