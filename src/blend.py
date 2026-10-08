import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from typing import Dict, Tuple, List

def rank_average(pred_arrays: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
    """Computes percentile rank average blend across multiple prediction arrays."""
    n_models = len(pred_arrays)
    if weights is None:
        weights = [1.0 / n_models] * n_models
    weights = np.array(weights) / np.sum(weights)

    n_samples = len(pred_arrays[0])
    blended_rank = np.zeros(n_samples, dtype=np.float64)
    for p, w in zip(pred_arrays, weights):
        rank = rankdata(p) / n_samples
        blended_rank += w * rank
    return blended_rank

def optimize_blend_weights(y_true: np.ndarray, oof_dict: Dict[str, np.ndarray]) -> Tuple[Dict[str, float], float]:
    """Finds optimal continuous ensemble weights maximizing ROC-AUC via SLSQP."""
    model_names = list(oof_dict.keys())
    m_matrix = np.column_stack([oof_dict[name] for name in model_names])

    def loss(w):
        w_norm = w / np.sum(w)
        pred = np.dot(m_matrix, w_norm)
        return -roc_auc_score(y_true, pred)

    n_models = len(model_names)
    initial_w = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    res = minimize(loss, initial_w, method='SLSQP', bounds=bounds, constraints=constraints)
    best_weights = {name: float(w) for name, w in zip(model_names, res.x / np.sum(res.x))}
    best_auc = -res.fun
    return best_weights, best_auc
