# scripts/serve_svm_api.py
"""
Script to run the FastAPI server for the SVM model.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
from models.svm_api import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
