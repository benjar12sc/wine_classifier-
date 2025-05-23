# src/models/svm_api.py
"""
FastAPI app to serve the trained SVM model for predictions.
"""
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
import numpy as np
import joblib
from pathlib import Path
import sqlite3
from functools import wraps
from functools import lru_cache
import time

# Load model and scaler
MODEL_PATH = str(Path(__file__).resolve().parent.parent.parent / 'models' / 'svm_model.joblib')
SCALER_PATH = str(Path(__file__).resolve().parent.parent.parent / 'models' / 'svm_scaler.joblib')
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Define top features (must match training)
TOP_FEATURES = [
    'proline', 'od280/od315_of_diluted_wines', 'color_intensity', 'flavanoids', 'alcohol'
]

class WineFeatures(BaseModel):
    proline: float
    od280_od315_of_diluted_wines: float
    color_intensity: float
    flavanoids: float
    alcohol: float

    def to_array(self):
        return np.array([
            self.proline,
            self.od280_od315_of_diluted_wines,
            self.color_intensity,
            self.flavanoids,
            self.alcohol
        ]).reshape(1, -1)

app = FastAPI()

DB_PATH = str(Path(__file__).resolve().parent.parent.parent / 'models' / 'tokens.db')
ADMIN_TOKEN = "adminsupersecret"  # In production, use env vars or a secrets manager
RATE_LIMIT_SECONDS = 60  # 1 minute between admin token additions

# Initialize DB if not exists
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS tokens (
            token TEXT PRIMARY KEY,
            last_used REAL DEFAULT 0
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS admin_rate_limit (
            id INTEGER PRIMARY KEY, last_add REAL DEFAULT 0
        )''')
        c.execute('''INSERT OR IGNORE INTO admin_rate_limit (id, last_add) VALUES (1, 0)''')
        conn.commit()
init_db()

def check_token(token: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT token FROM tokens WHERE token=?", (token,))
        return c.fetchone() is not None

def add_token(token: str):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO tokens (token) VALUES (?)", (token,))
        conn.commit()

def check_admin_token(authorization: str = Header(...)):
    if authorization != f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid or missing admin token.")

@lru_cache(maxsize=32)
def rate_limited_cached():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT last_add FROM admin_rate_limit WHERE id=1")
        last_add = c.fetchone()[0]
        now = time.time()
        if now - last_add < RATE_LIMIT_SECONDS:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
        c.execute("UPDATE admin_rate_limit SET last_add=? WHERE id=1", (now,))
        conn.commit()

def rate_limited():
    # Call the cached version, but always clear cache after to ensure up-to-date checks
    try:
        rate_limited_cached()
    finally:
        rate_limited_cached.cache_clear()

def get_user_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid or missing authorization token.")
    token = authorization.split(" ", 1)[1]
    if not check_token(token):
        raise HTTPException(status_code=401, detail="Invalid or missing authorization token.")
    return token

class AddTokenRequest(BaseModel):
    new_token: str
    rate_limit_seconds: int | None = None

@app.post("/predict")
def predict(features: WineFeatures, token: str = Depends(get_user_token)):
    X = features.to_array()
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    return {
        "prediction": int(pred),
        "probabilities": proba.tolist(),
        "class_names": model.classes_.tolist()
    }

@app.post("/admin/add_token")
def admin_add_token(
    req: AddTokenRequest,
    admin: None = Depends(check_admin_token)
):
    if req.rate_limit_seconds is not None:
        global RATE_LIMIT_SECONDS
        RATE_LIMIT_SECONDS = req.rate_limit_seconds
    try:
        rate_limited()
    except HTTPException as e:
        if e.status_code == 429:
            return {"error": "Rate limit exceeded. Try again later."}
        raise
    add_token(req.new_token)
    return {"message": "Token added successfully.", "rate_limit_seconds": RATE_LIMIT_SECONDS}
