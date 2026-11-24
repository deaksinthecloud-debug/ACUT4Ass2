"""Automated tests for the readmission pipeline. Run with:  python -m pytest -q"""
import numpy as np
import pandas as pd

from src.features import BINARY, CATEGORICAL, NUMERIC, RAW_INPUTS, ClinicalFeatures, build_preprocessor
from src.ingest import load_config
from src.validate import validate

CFG = load_config()


def sample(n=50, seed=0):
    """A small, realistic, fake dataset - fast, and independent of the real file."""
    r = np.random.default_rng(seed)
    return pd.DataFrame({
        "patient_id": range(n),
        "age": r.integers(20, 90, n),
        "sex": r.choice(["Female", "Male"], n),
        "bmi": r.uniform(18, 40, n).round(1),
        "systolic_bp": r.uniform(110, 160, n).round(),
        "diastolic_bp": r.uniform(60, 95, n).round(),
        "glucose": r.uniform(70, 180, n).round(1),
        "cholesterol": r.uniform(120, 260, n).round(1),
        "creatinine": r.uniform(0.5, 1.8, n).round(2),
        "diabetes": r.integers(0, 2, n),
        "hypertension": r.integers(0, 2, n),
        "diagnosis": r.choice(["Normal", "Pneumonia", "Heart Failure", "Sepsis"], n),
        "readmission_30d": r.integers(0, 2, n),
    })


# ---------- Validation ----------

def test_clean_data_passes():
    assert validate(sample(), CFG)["passed"]


def test_missing_column_fails():
    df = sample().drop(columns=["glucose"])
    report = validate(df, CFG)
    assert not report["passed"]
    assert "glucose" in report["errors"][0]


def test_out_of_range_age_is_flagged():
    df = sample()
    df.loc[0, "age"] = 250
    report = validate(df, CFG)
    assert report["passed"]            # a warning, not an error
    assert 0 in report["bad_rows"]     # but row 0 is marked for removal


def test_duplicate_patient_id_fails():
    df = sample()
    df.loc[1, "patient_id"] = df.loc[0, "patient_id"]
    assert not validate(df, CFG)["passed"]


def test_serving_mode_does_not_need_target():
    df = sample().drop(columns=["readmission_30d"])
    assert validate(df, CFG, mode="serving")["passed"]


# ---------- Features ----------

def test_engineered_features_are_correct():
    X = ClinicalFeatures().transform(sample(5))
    assert (X["pulse_pressure"] == X["systolic_bp"] - X["diastolic_bp"]).all()
    assert (X["comorbidity_count"] == X["diabetes"] + X["hypertension"]).all()
    assert (X["high_glucose"] == (X["glucose"] >= 126).astype(int)).all()


def test_feature_step_does_not_change_input():
    df = sample(5)
    before = df.copy()
    ClinicalFeatures().transform(df)
    pd.testing.assert_frame_equal(df, before)


def test_every_feature_column_is_listed_once():
    out = ClinicalFeatures().transform(sample(5)[RAW_INPUTS])
    listed = NUMERIC + BINARY + CATEGORICAL
    assert sorted(out.columns) == sorted(listed)


# ---------- Leakage controls ----------

def test_excluded_columns_are_not_model_inputs():
    for col in CFG["excluded_columns"]:
        assert col not in RAW_INPUTS, f"{col} must never be a model input"


# ---------- Preprocessing ----------

def test_unseen_category_does_not_crash():
    pre = build_preprocessor().fit(sample()[RAW_INPUTS])
    new = sample(3)[RAW_INPUTS].copy()
    new.loc[0, "diagnosis"] = "Stroke"
    assert pre.transform(new).shape[0] == 3