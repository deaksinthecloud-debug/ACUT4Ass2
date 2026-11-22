import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer          
from sklearn.impute import SimpleImputer               
from sklearn.pipeline import Pipeline                  
from sklearn.preprocessing import OneHotEncoder, StandardScaler  


class ClinicalFeatures(BaseEstimator, TransformerMixin):
    """Adds all five features from list
   
    """

    def fit(self, X, y=None):
        return self                       # nothing to learn

    def transform(self, X):
        X = X.copy()                      # never change the caller's data

        # 1. Pulse pressure: gap between systolic and diastolic (arterial stiffness)
        X["pulse_pressure"] = X["systolic_bp"] - X["diastolic_bp"]

        # 2. Comorbidity count: number of chronic conditions (0, 1 or 2)
        X["comorbidity_count"] = X["diabetes"] + X["hypertension"]

        # 3. Age band: risk rises non-linearly with age; absorbs top-coded age 90
        X["age_band"] = pd.cut(
            X["age"],
            bins=[0, 40, 65, 80, 200],
            labels=["Under 40", "40-64", "65-79", "80+"],
            right=False,
        ).astype(str)

        # 4. High glucose: at or above the ADA diabetes threshold (126 mg/dL)
        X["high_glucose"] = (X["glucose"] >= 126).astype(int)

        # 5. BMI category: WHO adult classes
        X["bmi_category"] = pd.cut(
            X["bmi"],
            bins=[0, 18.5, 25, 30, 35, 40, 1000],
            labels=["Underweight", "Normal", "Overweight", "Obese I", "Obese II", "Obese III"],
            right=False,
        ).astype(str)

        return X

# Numbers: will be median-imputed and scaled
NUMERIC = [
    "age", "bmi", "systolic_bp", "diastolic_bp",
    "glucose", "cholesterol", "creatinine",
    "pulse_pressure", "comorbidity_count",        
]

# 0/1 flags: will be imputed but NOT scaled (already 0 or 1)
BINARY = [
    "diabetes", "hypertension",
    "high_glucose",                               
]

# Text categories: will be one-hot encoded
CATEGORICAL = [
    "sex", "diagnosis",
    "age_band", "bmi_category",                   
]

# The ONLY raw columns the model may receive.
# Deliberately excludes:
#   patient_id      - an identifier, not a patient characteristic
#   mortality       - an outcome recorded after discharge (future information)
#   readmission_30d - the target itself
RAW_INPUTS = [
    "age", "sex", "bmi", "systolic_bp", "diastolic_bp",
    "glucose", "cholesterol", "creatinine",
    "diabetes", "hypertension", "diagnosis",
]
# ---- Preprocessing (Task 3.5) ----

def build_preprocessor():
    """Feature engineering + preprocessing as ONE pipeline. Code Snippet provided by AI

    Imputers, scaler and encoder LEARN from data when .fit() is called,
    so this must only ever be fitted on the training split.
    """
    # Numbers: fill gaps with the median, then scale
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    # Categories: fill gaps with the most common value, then one-hot encode
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Send each list of columns to its own treatment
    columns = ColumnTransformer([
        ("numeric", numeric_steps, NUMERIC),
        ("binary", SimpleImputer(strategy="most_frequent"), BINARY),
        ("categorical", categorical_steps, CATEGORICAL),
    ])

    # Engineer features FIRST, then preprocess them
    return Pipeline([
        ("engineer", ClinicalFeatures()),
        ("columns", columns),
    ])