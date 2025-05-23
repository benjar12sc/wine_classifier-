# tests/test_svm_api.py
"""
Unit tests for the SVM FastAPI prediction endpoint.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from fastapi.testclient import TestClient
import sqlite3
import pytest
import importlib.util

spec = importlib.util.spec_from_file_location("models.svm_api", str(Path(__file__).resolve().parent.parent / "src" / "models" / "svm_api.py"))
svm_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svm_api)
app = svm_api.app

client = TestClient(app)

# Helper to add a test token to the DB before running tests
def add_test_token(token: str):
    db_path = str(Path(__file__).resolve().parent.parent / 'models' / 'tokens.db')
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO tokens (token) VALUES (?)", (token,))
        conn.commit()

TEST_TOKEN = "testusertoken"
add_test_token(TEST_TOKEN)

valid_headers = {"Authorization": f"Bearer {TEST_TOKEN}"}

# Example valid payload
valid_payload = {
    "proline": 1000,
    "od280_od315_of_diluted_wines": 3.0,
    "color_intensity": 5.0,
    "flavanoids": 2.5,
    "alcohol": 13.0
}

def test_predict_success():
    response = client.post("/predict", json=valid_payload, headers=valid_headers)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probabilities" in data
    assert "class_names" in data
    assert len(data["probabilities"]) == len(data["class_names"])

@pytest.mark.parametrize("missing_field", [
    "proline", "od280_od315_of_diluted_wines", "color_intensity", "flavanoids", "alcohol"
])
def test_predict_missing_field(missing_field):
    payload = valid_payload.copy()
    payload.pop(missing_field)
    response = client.post("/predict", json=payload, headers=valid_headers)
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_unauthorized():
    response = client.post("/predict", json=valid_payload, headers={"Authorization": "Bearer wrongtoken"})
    assert response.status_code == 401
    assert "Invalid or missing authorization token" in response.text

# Ensure the file ends with a newline for pytest discovery
