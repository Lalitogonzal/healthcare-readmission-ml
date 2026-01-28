from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

MODEL_VERSION = "1.0.0"

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "readmission_model.joblib"
FEATURES_PATH = BASE_DIR / "models" / "feature_columns.joblib"

model = joblib.load(MODEL_PATH)
expected_features = joblib.load(FEATURES_PATH)

app = FastAPI(title="Hospital Readmission ML API", version=MODEL_VERSION)

class PatientInput(BaseModel):
    age: str
    time_in_hospital: int
    num_lab_procedures: int
    num_medications: int
    gender: str = "Unknown"
    race: str = "Unknown"

@app.get("/")
def home():
    return {"status": "ok", "service": "Hospital Readmission ML API", "model_version": MODEL_VERSION}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "expected_features": expected_features
    }

@app.post("/predict")
def predict_readmission(patient: PatientInput):
    try:
        df = pd.DataFrame([patient.model_dump()])
        df = df.reindex(columns=expected_features)

        prob = float(model.predict_proba(df)[0][1])
    except Exception as e:
        # show the real error instead of a silent 500
        raise HTTPException(status_code=500, detail=str(e))

    risk_band = "LOW" if prob < 0.3 else "MEDIUM" if prob < 0.6 else "HIGH"

    return {
        "readmission_probability": round(prob, 3),
        "risk_band": risk_band,
        "model_version": MODEL_VERSION
    }
