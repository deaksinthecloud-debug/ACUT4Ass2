"""Prediction: score new patients with the approved model.

Usage:  python -m src.predict --input data/new_patients_example.csv
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.features import RAW_INPUTS
from src.ingest import load_config, load_raw
from src.validate import validate


def predict(df, model_path="models/model.joblib"):
    """Validate new patients, then score them with the saved pipeline."""
    cfg = load_config()

    # 1. Validate in SERVING mode - new patients have no outcome yet
    report = validate(df, cfg, mode="serving")
    if not report["passed"]:
        raise ValueError(f"Input failed validation - not scoring: {report['errors']}")
    for w in report["warnings"]:
        print("WARNING:", w)

    # 2. Load the approved pipeline: features + preprocessing + model, all in one
    if not Path(model_path).exists():
        raise FileNotFoundError(f"No approved model at '{model_path}'. Run: python -m src.train")
    pipe = joblib.load(model_path)

    # 3. Score - using ONLY the raw inputs the model was trained on
    risk = pipe.predict_proba(df[RAW_INPUTS])[:, 1]

    # 4. Build the output a care team would use
    threshold = cfg["decision_threshold"]
    out = pd.DataFrame({
        "patient_id": df[cfg["data"]["id_column"]],
        "readmission_risk": risk.round(3),
        "flag_for_review": (risk >= threshold).astype(int),
        "input_check": "ok",
    })
    out.loc[report["bad_rows"], "input_check"] = "out of range - review manually"
    out.loc[report["bad_rows"], "flag_for_review"] = 1     # never silently clear a bad row
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score new patients for 30-day readmission risk.")
    parser.add_argument("--input", required=True, help="CSV of new patients")
    parser.add_argument("--output", default="artefacts/predictions.csv")
    parser.add_argument("--model", default="models/model.joblib")
    args = parser.parse_args()

    new_patients = load_raw(args.input)
    predictions = predict(new_patients, args.model)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"\nScored {len(predictions)} patients -> {args.output}")
    print(f"Flagged for follow-up or review: {predictions['flag_for_review'].sum()}")
    print(predictions.to_string(index=False))