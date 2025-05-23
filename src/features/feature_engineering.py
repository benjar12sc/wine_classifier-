# src/features/feature_engineering.py
"""
Feature engineering for the wine dataset.
"""
import pandas as pd
from typing import List

def select_top_features(X: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """
    Selects the top features from the DataFrame.
    Args:
        X: Feature DataFrame
        feature_names: List of top feature names
    Returns:
        DataFrame with only the selected features
    """
    return X[feature_names]

def get_most_important_features(X: pd.DataFrame, importances: list, top_n: int = 5) -> list:
    """
    Returns the names of the top_n most important features based on importances.
    Args:
        X: Feature DataFrame
        importances: List or array of feature importances (e.g., from a model)
        top_n: Number of top features to return
    Returns:
        List of top_n feature names
    """
    import numpy as np
    indices = np.argsort(importances)[::-1][:top_n]
    return X.columns[indices].tolist()
