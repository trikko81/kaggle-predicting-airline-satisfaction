import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import Optional

def generate_stratified_folds(
    train_df: pd.DataFrame,
    target_col: str = "satisfaction",
    n_splits: int = 10,
    seed: int = 42
) -> np.ndarray:
    """Generate Stratified K-Fold partition array for zero-leakage cross-validation."""
    y = train_df[target_col].copy()
    if y.dtype == object:
        y = (y.astype(str).str.lower().str.strip() == "satisfied").astype(int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.zeros(len(train_df), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(train_df, y)):
        folds[val_idx] = fold_idx
    return folds
