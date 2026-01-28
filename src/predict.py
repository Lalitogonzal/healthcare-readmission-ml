import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "readmission_model.joblib"
FEATURES_PATH = BASE_DIR / "models" / "feature_columns.joblib"

model = joblib.load(MODEL_PATH)
expected_features = joblib.load(FEATURES_PATH)

def predict_readmission(patient_data: dict) -> dict:
    df = pd.DataFrame([patient_data])

    # enforce schema order and missing columns
    df = df.reindex(columns=expected_features)

    prob = float(model.predict_proba(df)[0][1])

    return {
        "readmission_probability": round(prob, 3),
        "risk_band": "LOW" if prob < 0.3 else "MEDIUM" if prob < 0.6 else "HIGH"
    }

if __name__ == "__main__":
    test_patient = {
        "age": "[60-70)",
        "time_in_hospital": 4,
        "num_lab_procedures": 45,
        "num_medications": 12,
        "gender": "Male",
        "race": "Caucasian"
    }

    print("Expected features:", expected_features)
    print(predict_readmission(test_patient))
