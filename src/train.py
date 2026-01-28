import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# ------------------
# Paths
# ------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "diabetic_data.csv"
MODEL_PATH = BASE_DIR / "models" / "readmission_model.joblib"
FEATURES_PATH = BASE_DIR / "models" / "feature_columns.joblib"

# ------------------
# Config: lock feature schema
# ------------------
SELECT_FEATURES = [
    "age",
    "time_in_hospital",
    "num_lab_procedures",
    "num_medications",
    "gender",
    "race"
]

# ------------------
# Load data
# ------------------
df = pd.read_csv(DATA_PATH)

# target
df["readmitted_30"] = (df["readmitted"] == "<30").astype(int)

# keep only the features we serve + clean missing markers
X = df[SELECT_FEATURES].replace("?", np.nan)
y = df["readmitted_30"]

# Save feature schema ONCE (this is what the API will enforce)
joblib.dump(SELECT_FEATURES, FEATURES_PATH)

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

# ------------------
# Preprocessing
# ------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ],
    remainder="drop"
)

# ------------------
# Model
# ------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, probs))
print("PR-AUC:", average_precision_score(y_test, probs))
print("Base rate:", y_test.mean())

joblib.dump(model, MODEL_PATH)
print("Model saved:", MODEL_PATH)
print("Features saved:", FEATURES_PATH)
print("Features:", SELECT_FEATURES)
