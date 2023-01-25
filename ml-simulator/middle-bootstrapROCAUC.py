from typing import Tuple

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score

def roc_auc_ci(
    classifier: ClassifierMixin,
    regressors: np.ndarray,
    target: np.ndarray,
    conf: float = 0.95,
    n_bootstraps: int = 10_000,
) -> Tuple[float, float]:
    """Returns confidence bounds of the ROC-AUC"""
    
    roc_auc_scores = []
    y_proba = classifier.predict_proba(regressors)[:,1]
    
    for _ in range(n_bootstraps):
        idx = np.random.choice(regressors.shape[0], regressors.shape[0], replace = True)
        while len(np.unique(target[idx])) == 1:
            idx = np.random.choice(regressors.shape[0], regressors.shape[0], replace=True)
            
        roc_auc = roc_auc_score(target[idx], y_proba[idx])
        roc_auc_scores.append(roc_auc)
        
    lcb, ucb = np.quantile(roc_auc_scores, [((1 - conf) / 2), (1 - (1 - conf) / 2)])
    return (lcb, ucb)
