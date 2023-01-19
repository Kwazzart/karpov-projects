from typing import Tuple

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score


def roc_auc_ci(
    classifier: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    conf: float = 0.95,
    n_bootstraps: int = 10_000,
) -> Tuple[float, float]:
    """Returns confidence bounds of the ROC-AUC"""
    roc_auc_scores = []
    data = np.concatenate([X, y.reshape(-1,1)], axis = 1)
    N_SAMPLES = data.shape[0]
    alpha = 1 - conf
    for i in range(n_bootstraps):
        indx = np.random.randint(0, N_SAMPLES, N_SAMPLES)
        if len(np.unique(data[indx, -1])) == 1:
            continue
        bootstrapped_data = np.zeros_like(data)
        for j in range(len(data[0])):
            bootstrapped_data[:, j] = data[indx, j]
        x_temp = bootstrapped_data[:, :-1]
        y_temp = bootstrapped_data[:, -1]
        roc_auc = roc_auc_score(y_temp, classifier.predict(x_temp))
        roc_auc_scores.append(roc_auc)
        print(i)
    lcb, ucb = np.quantile(roc_auc_scores, [(alpha / 2), (1 - alpha / 2)])
    return (lcb, ucb)





