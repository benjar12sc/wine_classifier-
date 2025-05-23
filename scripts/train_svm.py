# scripts/train_svm.py
"""
Script to train and persist an SVM model for the wine dataset.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
from data.load_wine import get_wine_data
from features.feature_engineering import select_top_features
from models.svm_model import train_svm, save_model

TOP_FEATURES = [
    'proline', 'od280/od315_of_diluted_wines', 'color_intensity', 'flavanoids', 'alcohol'
]  # Example top features, adjust as needed

MODEL_PATH = str(Path(__file__).resolve().parent.parent / 'models' / 'svm_model.joblib')
SCALER_PATH = str(Path(__file__).resolve().parent.parent / 'models' / 'svm_scaler.joblib')

def main():
    X, y = get_wine_data(as_frame=True)
    X_selected = select_top_features(X, TOP_FEATURES)
    model, scaler = train_svm(X_selected, y)
    save_model(model, scaler, MODEL_PATH, SCALER_PATH)
    print(f"Model saved to {MODEL_PATH}\nScaler saved to {SCALER_PATH}")

if __name__ == "__main__":
    main()
