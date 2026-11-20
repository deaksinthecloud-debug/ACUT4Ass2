"""Validation: check the data against the schema in config/config.yaml.

Two modes:
  training - the target column must be present and is checked too.
  serving  - for new patients at prediction time (no target yet).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.ingest import load_config, load_raw


def validate(df, cfg, mode="training"):
    """Check df against the config. Returns a report dictionary."""
    errors = []                                  # problems that must stop the pipeline
    warnings = []                                # suspicious, logged, but carry on
    row_bad = pd.Series(False, index=df.index)   # True for rows that fail a row-level check

    s = cfg["schema"]
    id_col = cfg["data"]["id_column"]
    target = cfg["data"]["target"]

    # ---- Check 1: missing columns (error) ----
    expected = [id_col] + list(s["numeric"]) + s["binary"] + list(s["categorical"])
    if mode == "training":
        expected.append(target)                  # at serving time there is no target yet

    missing = [c for c in expected if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
        return {"passed": False, "mode": mode, "n_rows": len(df),
                "errors": errors, "warnings": warnings, "bad_rows": []}

    # ---- Check 2: unexpected columns (warning) ----
    extra = [c for c in df.columns if c not in expected]
    if extra:
        warnings.append(f"unexpected columns (not used as features): {extra}")

    # ---- Checks 3-5: numeric columns ----
    for col, rng in s["numeric"].items():
        vals = pd.to_numeric(df[col], errors="coerce")

        # Check 3: wrong data type (error) - present, but not a number
        not_numeric = vals.isna() & df[col].notna()
        if not_numeric.any():
            errors.append(f"{col}: {not_numeric.sum()} non-numeric values")

        # Check 4: missing values (warning)
        if df[col].isna().any():
            warnings.append(f"{col}: {df[col].isna().sum()} missing values")

        # Check 5: out of range (warning) - remember which rows
        out = (vals < rng["min"]) | (vals > rng["max"])
        if out.any():
            warnings.append(f"{col}: {out.sum()} values outside [{rng['min']}, {rng['max']}]")
            row_bad |= out

    # ---- Check 6: 0/1 flags contain anything else (error) ----
    for col in s["binary"]:
        bad = ~df[col].isin([0, 1])
        if bad.any():
            errors.append(f"{col}: {bad.sum()} values that are not 0 or 1")
            row_bad |= bad

    # ---- Check 7: categories outside the allowed list (warning) ----
    for col, allowed in s["categorical"].items():
        bad = ~df[col].isin(allowed) & df[col].notna()
        if bad.any():
            unseen = sorted(df.loc[bad, col].unique())
            warnings.append(f"{col}: unexpected categories {unseen}")

    # ---- Check 8: duplicates ----
    if df.duplicated().any():                     # warning: whole row repeated
        warnings.append(f"{df.duplicated().sum()} fully duplicated rows")
    if df[id_col].duplicated().any():             # error: one ID, two different patients
        errors.append(f"{df[id_col].duplicated().sum()} duplicate {id_col} values")

    # ---- Check 9: target sanity (error) - training only ----
    if mode == "training":
        if not df[target].isin([0, 1]).all():
            errors.append(f"{target} contains values other than 0 and 1")
        rate = df[target].mean()
        if not 0.01 < rate < 0.99:
            errors.append(f"{target} positive rate {rate:.3f} is implausible")

        # ---- Domain check: died AND readmitted (warning) ----
        if "mortality" in df.columns:
            both = ((df["mortality"] == 1) & (df[target] == 1)).sum()
            if both > 0:
                warnings.append(
                    f"{both} rows have mortality=1 AND readmission=1 - "
                    "label definitions need checking with the data owner"
                )

    return {
        "passed": len(errors) == 0,
        "mode": mode,
        "n_rows": len(df),
        "errors": errors,
        "warnings": warnings,
        "bad_rows": df.index[row_bad].tolist(),   # check 10: which rows to review
    }


if __name__ == "__main__":
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Validate a data file against the config.")
    parser.add_argument("--input", default=cfg["data"]["raw_path"], help="CSV file to check")
    parser.add_argument("--mode", default="training", choices=["training", "serving"])
    args = parser.parse_args()

    print(f"Validating: {args.input} (mode: {args.mode})")
    df = load_raw(args.input)
    report = validate(df, cfg, mode=args.mode)

    # Save a short version: the COUNT of bad rows, not the full list
    to_save = {k: v for k, v in report.items() if k != "bad_rows"}
    to_save["n_bad_rows"] = len(report["bad_rows"])

    Path("artefacts").mkdir(exist_ok=True)
    with open("artefacts/validation_report.json", "w") as f:
        json.dump(to_save, f, indent=2)

    print(json.dumps(to_save, indent=2))
    raise SystemExit(0 if report["passed"] else 1)