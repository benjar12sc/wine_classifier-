# src/models/svm_model.py
"""
SVM model training and persistence for the wine dataset.
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple

def train_svm(X: pd.DataFrame, y: pd.Series) -> Tuple[GridSearchCV, StandardScaler]:
    """
    Trains an SVM classifier with hyperparameter tuning and scaling.
    Returns the fitted GridSearchCV object and scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    svm = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_scaled, y)
    return grid_search, scaler

def save_model(model: GridSearchCV, scaler: StandardScaler, model_path: str, scaler_path: str):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

def load_model(model_path: str, scaler_path: str) -> Tuple[GridSearchCV, StandardScaler]:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler
